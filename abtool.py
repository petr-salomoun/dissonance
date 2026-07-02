"""A/B pairwise calibration tool for unpleasantness scorer weights."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from glob import glob
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import minimize

from dissonance.analysis.scorer import UnpleasantnessScorer


FEATURE_NAMES = [
    "roughness",
    "sharpness",
    "dissonance",
    "crest_factor",
    "band_energy_2_4khz",
    "am_energy_70hz",
    "roughness_x_sharpness",
]

WEIGHT_FLOOR = 0.02
PRIOR_BLEND = 0.55
RIDGE_ALPHA = 0.10


def _timestamp_compact() -> str:
    return (
        str(np.datetime64("now", "s"))
        .replace(":", "")
        .replace("-", "")
        .replace("T", "_")
    )


def _resolve_wavs(inputs: list[str]) -> list[Path]:
    files: set[Path] = set()
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for wav in p.glob("*.wav"):
                files.add(wav.resolve())
            for wav in p.glob("*.WAV"):
                files.add(wav.resolve())
            continue
        if p.is_file() and p.suffix.lower() == ".wav":
            files.add(p.resolve())
            continue
        for match in glob(item):
            mp = Path(match)
            if mp.is_file() and mp.suffix.lower() == ".wav":
                files.add(mp.resolve())
    return sorted(files)


def _play_with_fallback(path: Path, no_play: bool) -> None:
    if no_play:
        print(f"[no-play] {path}")
        return

    signal = None
    sr = None
    try:
        signal, sr = sf.read(str(path), dtype="float32", always_2d=False)
        x = np.asarray(signal, dtype=np.float32)
        if x.ndim == 2:
            x = np.mean(x, axis=1, dtype=np.float32)
        signal = x
    except Exception as exc:
        print(f"Warning: failed to decode for playback ({path}): {exc}")

    try:
        import sounddevice as sd  # type: ignore

        if signal is not None and sr is not None:
            sd.play(signal, int(sr))
            sd.wait()
            return
    except Exception as exc:
        print(f"Warning: sounddevice unavailable/failed ({exc}); trying system players.")

    try:
        proc = subprocess.run(["aplay", str(path)], capture_output=True, text=True)
        if proc.returncode == 0:
            return
    except Exception:
        pass

    try:
        proc = subprocess.run(["ffplay", "-nodisp", "-autoexit", str(path)], capture_output=True, text=True)
        if proc.returncode == 0:
            return
    except Exception:
        pass

    print(f"Unable to auto-play: {path}")
    print("Please play this file manually.")


def _analyze_wav_features(wav: Path, scorer: UnpleasantnessScorer) -> dict[str, float]:
    sig, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    x = np.asarray(sig, dtype=np.float32)
    if x.ndim == 2:
        x = np.mean(x, axis=1, dtype=np.float32)
    _, features = scorer.score(x, int(sr))
    return {k: float(v) for k, v in features.items()}


def _feature_vector(features: dict[str, float]) -> np.ndarray:
    rough = float(features.get("roughness", 0.0))
    sharp = float(features.get("sharpness", 0.0))
    return np.array(
        [
            rough,
            sharp,
            float(features.get("dissonance", 0.0)),
            float(features.get("crest_factor", 0.0)),
            float(features.get("band_energy_2_4khz", 0.0)),
            float(features.get("am_energy_70hz", 0.0)),
            rough * sharp,
        ],
        dtype=np.float64,
    )


def _build_pairs(indices: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            pairs.append((indices[i], indices[j]))
    return pairs


def _select_pairs(
    pairs: list[tuple[int, int]],
    scores: np.ndarray,
    strategy: str,
    n_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if not pairs:
        return []
    if strategy == "smart":
        ranked = sorted(pairs, key=lambda ij: abs(float(scores[ij[0]] - scores[ij[1]])))
    else:
        ranked = list(pairs)
        rng.shuffle(ranked)
    return ranked[: min(n_pairs, len(ranked))]


def _parse_pref(raw: str) -> str | None:
    x = raw.strip().lower()
    if x in {"a"}:
        return "A"
    if x in {"b"}:
        return "B"
    if x in {"=", "e", "equal"}:
        return "equal"
    if x in {"s", "skip"}:
        return "skip"
    return None


def _fit_bt(n_items: int, comparisons: list[tuple[int, int, float]]) -> np.ndarray:
    # comparisons are (winner, loser, weight)
    if len(comparisons) < 3:
        raise RuntimeError("Too few comparisons for BT optimization")

    def objective(theta_raw: np.ndarray) -> tuple[float, np.ndarray]:
        theta = theta_raw.copy()
        theta[-1] = 0.0
        nll = 0.0
        grad = np.zeros_like(theta)
        for w, l, wt in comparisons:
            z = theta[w] - theta[l]
            if z >= 0:
                p = 1.0 / (1.0 + np.exp(-z))
            else:
                ez = np.exp(z)
                p = ez / (1.0 + ez)
            p = float(np.clip(p, 1e-10, 1.0 - 1e-10))
            nll -= wt * np.log(p)
            g = wt * (p - 1.0)
            grad[w] += g
            grad[l] -= g
        grad[-1] = 0.0
        return float(nll), grad

    x0 = np.zeros(n_items, dtype=np.float64)
    bounds = [(-6.0, 6.0)] * n_items
    bounds[-1] = (0.0, 0.0)

    def fun(x: np.ndarray) -> float:
        return objective(x)[0]

    def jac(x: np.ndarray) -> np.ndarray:
        return objective(x)[1]

    res = minimize(fun=fun, x0=x0, jac=jac, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(f"BT optimization failed: {res.message}")
    out = np.asarray(res.x, dtype=np.float64)
    out[-1] = 0.0
    out -= float(np.mean(out))
    return out


def _fit_theta_lstsq(n_items: int, comparisons: list[tuple[int, int, float]]) -> np.ndarray:
    if len(comparisons) < 2:
        raise RuntimeError("Too few comparisons for least-squares fallback")
    rows: list[np.ndarray] = []
    y: list[float] = []
    for w, l, wt in comparisons:
        reps = int(max(1.0, round(wt * 2.0)))
        for _ in range(reps):
            r = np.zeros(n_items, dtype=np.float64)
            r[w] = 1.0
            r[l] = -1.0
            rows.append(r)
            y.append(1.0)
    a = np.vstack(rows)
    b = np.array(y, dtype=np.float64)
    theta, *_ = np.linalg.lstsq(a, b, rcond=None)
    theta = np.asarray(theta, dtype=np.float64)
    theta -= float(np.mean(theta))
    return theta


def _fit_weights_from_theta(
    theta: np.ndarray,
    feature_names: list[str],
    xmat: np.ndarray,
    default_weights: dict[str, float],
) -> dict[str, float]:
    """Robust ridge-regularized weight fit with prior blending and floor."""
    n_feat = xmat.shape[1]
    A = xmat.T @ xmat + RIDGE_ALPHA * np.eye(n_feat)
    b = xmat.T @ theta
    w_raw = np.linalg.solve(A, b)

    w_pos = np.clip(w_raw, 0.0, None)
    total = float(np.sum(w_pos))
    if total <= 1e-12:
        return dict(default_weights)
    w_norm = w_pos / total

    prior = np.array([float(default_weights.get(k, 0.0)) for k in feature_names], dtype=np.float64)
    prior_sum = float(np.sum(prior))
    if prior_sum > 1e-12:
        prior = prior / prior_sum
    w_blended = (1.0 - PRIOR_BLEND) * w_norm + PRIOR_BLEND * prior

    w_floored = np.maximum(w_blended, WEIGHT_FLOOR)
    w_final = w_floored / float(np.sum(w_floored))

    return {k: float(v) for k, v in zip(feature_names, w_final)}


def _normalize_weights(raw: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    clipped = np.clip(raw.astype(np.float64), 0.0, None)
    total = float(np.sum(clipped))
    if total <= 1e-12:
        default = UnpleasantnessScorer.DEFAULT_WEIGHTS
        return {k: float(default[k]) for k in feature_names}
    norm = clipped / total
    return {k: float(v) for k, v in zip(feature_names, norm)}


def run_ab_session(
    wavs: list[Path],
    scorer: "UnpleasantnessScorer",
    n_pairs: int = 6,
    strategy: str = "smart",
    no_play: bool = False,
    seed: int = 0,
    accumulated_rows: list[tuple[int, int, float]] | None = None,
    accumulated_items: list[dict[str, float]] | None = None,
) -> tuple[dict[str, float] | None, list[tuple[int, int, float]], list[dict[str, float]]]:
    """Run a mini A-B session and fit on accumulated history.

    Returns (calibrated_weights_or_none, updated_accumulated_rows, updated_accumulated_items).
    """
    rng = np.random.default_rng(seed)
    old_weights = dict(scorer.weights)
    feature_names = list(FEATURE_NAMES)
    all_rows: list[tuple[int, int, float]] = list(accumulated_rows) if accumulated_rows is not None else []
    all_items: list[dict[str, float]] = [dict(it) for it in accumulated_items] if accumulated_items is not None else []
    item_offset = len(all_items)

    features_by_idx: dict[int, dict[str, float]] = {}
    score_vec = np.zeros(len(wavs), dtype=np.float64)
    for idx, wav in enumerate(wavs):
        try:
            sig, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            x = np.asarray(sig, dtype=np.float32)
            if x.ndim == 2:
                x = np.mean(x, axis=1, dtype=np.float32)
            s, f = scorer.score(x, int(sr))
            features_by_idx[idx] = {k: float(v) for k, v in f.items()}
            score_vec[idx] = float(s)
        except Exception as exc:
            print(f"Warning: failed to analyze {wav}: {exc}")
            features_by_idx[idx] = {k: 0.0 for k in feature_names}

    for i in range(len(wavs)):
        all_items.append(dict(features_by_idx[i]))

    all_pairs = _build_pairs(list(range(len(wavs))))
    selected = _select_pairs(all_pairs, score_vec, strategy, n_pairs, rng)

    print(f"\n[A-B calibration] {len(selected)} pairs from {len(wavs)} candidates")

    for idx, (a, b) in enumerate(selected, start=1):
        pa, pb = wavs[a], wavs[b]
        print(f"\nPair {idx}/{len(selected)}:")
        print(f"  [A] {pa.name}  (score: {score_vec[a]:.4f})")
        print(f"  [B] {pb.name}  (score: {score_vec[b]:.4f})")
        print("  Playing A...")
        _play_with_fallback(pa, no_play)
        print("  --- now playing B ---")
        _play_with_fallback(pb, no_play)
        print("  --- done ---")

        pref = None
        if no_play:
            sa = float(score_vec[a])
            sb = float(score_vec[b])
            if np.isclose(sa, sb, atol=1e-6):
                pref = "equal"
            else:
                pref = "A" if sa >= sb else "B"
            print(f"  [auto/no-play] selected: {pref}")
        else:
            while pref is None:
                raw = input("  Less pleasant? [A/b/=/s]: ")
                pref = _parse_pref(raw)
                if pref is None:
                    print("  Invalid. Enter A, b, =, or s.")

        ga = item_offset + a
        gb = item_offset + b
        if pref == "A":
            all_rows.append((ga, gb, 1.0))
        elif pref == "B":
            all_rows.append((gb, ga, 1.0))
        elif pref == "equal":
            all_rows.append((ga, gb, 0.5))
            all_rows.append((gb, ga, 0.5))

    compared = len(all_rows)
    if compared < 3:
        print("[A-B] Too few judgments; keeping current weights.")
        return None, all_rows, all_items

    n_items = len(all_items)
    try:
        theta = _fit_bt(n_items=n_items, comparisons=all_rows)
    except Exception as exc:
        print(f"[A-B] BT failed ({exc}), falling back to lstsq.")
        try:
            theta = _fit_theta_lstsq(n_items=n_items, comparisons=all_rows)
        except Exception as exc2:
            print(f"[A-B] Calibration failed: {exc2}")
            return None, all_rows, all_items

    xmat = np.vstack([_feature_vector(all_items[i]) for i in range(n_items)])
    calibrated = _fit_weights_from_theta(theta, feature_names, xmat, UnpleasantnessScorer.DEFAULT_WEIGHTS)
    print("[A-B] Calibrated weights:")
    for k, v in sorted(calibrated.items()):
        print(f"  {k}: {old_weights.get(k, 0):.4f} -> {v:.4f}")
    return calibrated, all_rows, all_items


def _weights_from_file(path: str | None) -> dict[str, float]:
    if not path:
        return dict(UnpleasantnessScorer.DEFAULT_WEIGHTS)
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("weights-in must be a JSON object")
    weights = dict(UnpleasantnessScorer.DEFAULT_WEIGHTS)
    for k in weights:
        if k in data:
            weights[k] = float(data[k])
    return weights


def _print_summary(
    compared: int,
    judgments: dict[int, float],
    wavs: list[Path],
    old_w: dict[str, float],
    new_w: dict[str, float] | None,
) -> None:
    print("\n========================================")
    print("Calibration Summary")
    print("========================================")
    print(f"Pairs compared: {compared}")
    if judgments:
        ranked = sorted(judgments.items(), key=lambda kv: kv[1], reverse=True)
        most = ranked[0][0]
        least = ranked[-1][0]
        print(f"Most unpleasant by judgments: {wavs[most]}")
        print(f"Least unpleasant by judgments: {wavs[least]}")
    print("\nWeights (old -> new):")
    for k in old_w:
        if new_w is None:
            print(f"  {k}: {old_w[k]:.6f} -> (unchanged)")
        else:
            print(f"  {k}: {old_w[k]:.6f} -> {new_w[k]:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abtool", description="A/B pairwise calibration for unpleasantness scorer")
    parser.add_argument("--wavs", nargs="+", required=True, help="Directory, glob pattern, wav path, or list of these")
    parser.add_argument("--pairs", type=int, default=None, help="Number of pairs to compare")
    parser.add_argument("--weights-in", type=str, default=None, dest="weights_in")
    parser.add_argument("--weights-out", type=str, default="weights_calibrated.json", dest="weights_out")
    parser.add_argument("--strategy", choices=["random", "smart"], default="random")
    parser.add_argument("--no-play", action="store_true", dest="no_play")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ab-interval", type=int, default=0, dest="ab_interval")
    parser.add_argument("--rerun-sweep", action="store_true", dest="rerun_sweep")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    try:
        wavs = _resolve_wavs(args.wavs)
    except Exception as exc:
        print(f"Error while resolving wav files: {exc}")
        sys.exit(1)

    if len(wavs) < 2:
        print("Need at least two WAV files.")
        sys.exit(1)

    try:
        old_weights = _weights_from_file(args.weights_in)
        scorer = UnpleasantnessScorer(weights=old_weights)
    except Exception as exc:
        print(f"Failed to load weights: {exc}")
        sys.exit(1)

    initial_scores = {}
    features_by_idx: dict[int, dict[str, float]] = {}
    score_vec = np.zeros(len(wavs), dtype=np.float64)
    for idx, wav in enumerate(wavs):
        try:
            sig, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            x = np.asarray(sig, dtype=np.float32)
            if x.ndim == 2:
                x = np.mean(x, axis=1, dtype=np.float32)
            s, f = scorer.score(x, int(sr))
            initial_scores[str(wav)] = float(s)
            features_by_idx[idx] = {k: float(v) for k, v in f.items()}
            score_vec[idx] = float(s)
        except Exception as exc:
            print(f"Warning: failed to analyze {wav}: {exc}")
            initial_scores[str(wav)] = 0.0
            features_by_idx[idx] = {k: 0.0 for k in FEATURE_NAMES}

    all_pairs = _build_pairs(list(range(len(wavs))))
    default_n = min(len(all_pairs), 30)
    target_pairs = default_n if args.pairs is None else max(0, min(args.pairs, len(all_pairs)))
    selected = _select_pairs(all_pairs, score_vec, args.strategy, target_pairs, rng)

    print(f"Loaded {len(wavs)} wav files.")
    print(f"Using {len(selected)} pairs (strategy={args.strategy}).")

    pair_records: list[dict[str, str]] = []
    bt_rows: list[tuple[int, int, float]] = []
    judgment_score: dict[int, float] = {i: 0.0 for i in range(len(wavs))}

    for idx, (a, b) in enumerate(selected, start=1):
        pa = wavs[a]
        pb = wavs[b]
        print("\n========================================")
        print(f"Pair {idx}/{len(selected)}: Comparing 2 files")
        print("========================================")
        print(f"\n[A] {pa}  (score: {score_vec[a]:.4f})")
        print(f"[B] {pb}  (score: {score_vec[b]:.4f})")
        print("Playing A...")
        _play_with_fallback(pa, args.no_play)
        print("--- now playing B ---")
        _play_with_fallback(pb, args.no_play)
        print("--- done ---")

        pref = None
        while pref is None:
            raw = input(
                "Which sounds LESS pleasant? [A/b/=/s] "
                "(A=A less pleasant, b=B less pleasant, =equal, s=skip): "
            )
            pref = _parse_pref(raw)
            if pref is None:
                print("Invalid response. Please enter A, B, =/equal, or s/skip.")

        pair_records.append({"a": str(pa), "b": str(pb), "preference": pref})

        if pref == "A":
            bt_rows.append((a, b, 1.0))
            judgment_score[a] += 1.0
            judgment_score[b] -= 1.0
        elif pref == "B":
            bt_rows.append((b, a, 1.0))
            judgment_score[b] += 1.0
            judgment_score[a] -= 1.0
        elif pref == "equal":
            bt_rows.append((a, b, 0.5))
            bt_rows.append((b, a, 0.5))
            judgment_score[a] += 0.0
            judgment_score[b] += 0.0

    compared = sum(1 for r in pair_records if r["preference"] != "skip")

    calibrated_weights: dict[str, float] | None = None
    theta = None
    if compared < 3:
        print("Warning: fewer than 3 non-skipped pairs collected; skipping calibration.")
    else:
        n_items = len(wavs)
        try:
            theta = _fit_bt(n_items=n_items, comparisons=bt_rows)
            print("Bradley-Terry fit succeeded.")
        except Exception as exc:
            print(f"Warning: Bradley-Terry failed ({exc}), falling back to least-squares.")
            try:
                theta = _fit_theta_lstsq(n_items=n_items, comparisons=bt_rows)
            except Exception as exc2:
                print(f"Calibration failed: {exc2}")
                theta = None

        if theta is not None:
            xmat = np.vstack([_feature_vector(features_by_idx[i]) for i in range(len(wavs))])
            calibrated_weights = _fit_weights_from_theta(theta, FEATURE_NAMES, xmat, UnpleasantnessScorer.DEFAULT_WEIGHTS)

    weights_out = Path(args.weights_out)
    try:
        weights_out.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        out_payload = calibrated_weights if calibrated_weights is not None else old_weights
        with weights_out.open("w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2, sort_keys=True)
        print(f"Saved weights to: {weights_out}")
    except Exception as exc:
        print(f"Failed to save weights: {exc}")

    session_path = weights_out.parent / f"ab_session_{_timestamp_compact()}.json"
    try:
        session_payload = {
            "pairs": pair_records,
            "initial_scores": initial_scores,
            "calibrated_weights": calibrated_weights if calibrated_weights is not None else old_weights,
            "timestamp": str(np.datetime64("now", "s")),
        }
        with session_path.open("w", encoding="utf-8") as f:
            json.dump(session_payload, f, indent=2, sort_keys=True)
        print(f"Saved session to: {session_path}")
    except Exception as exc:
        print(f"Warning: failed to save session: {exc}")

    _print_summary(
        compared=compared,
        judgments=judgment_score,
        wavs=wavs,
        old_w=old_weights,
        new_w=calibrated_weights,
    )

    if args.rerun_sweep:
        sweep_script = Path(__file__).parent / "sweep.py"
        cmd = [
            sys.executable,
            str(sweep_script),
            "--out-dir",
            str(weights_out.parent / "sweep_recalibrated"),
        ]
        print(f"\nRe-running sweep with updated weights (external run): {' '.join(cmd)}")
        try:
            subprocess.run(cmd)
        except Exception as exc:
            print(f"Warning: failed to rerun sweep: {exc}")


if __name__ == "__main__":
    main()
