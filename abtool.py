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

from dissonance.analysis.synth_features import compute_synth_features
from dissonance.analysis.scorer import UnpleasantnessScorer


FEATURE_NAMES = [
    "roughness",
    "sharpness",
    "dissonance",
    "crest_factor",
    "band_energy_2_4khz",
    "am_energy_70hz",
    "roughness_x_sharpness",
    *UnpleasantnessScorer.SYNTH_FEATURE_NAMES,
]

DEFAULT_CALIBRATION_WEIGHTS: dict[str, float] = {
    **UnpleasantnessScorer.DEFAULT_WEIGHTS,
    **UnpleasantnessScorer.DEFAULT_SYNTH_WEIGHTS,
}

RIDGE_ALPHA = 14.0
STD_EPS = 1e-3
VAR_EPS = 1e-8
COEF_BOUND = 3.0
INTERCEPT_BOUND = 2.0
MIN_DIRECTIONAL_COMPARISONS = 8
MIN_DISTINCT_ITEMS = 5
MIN_ACTIVE_FEATURES = 3
MIN_FEATURE_CONTRAST = 3
MIN_FEATURE_ITEM_SUPPORT = 3
MIN_FEATURE_VALUE_BUCKETS = 3

ACOUSTIC_FEATURE_NAMES = [
    "roughness",
    "sharpness",
    "dissonance",
    "crest_factor",
    "band_energy_2_4khz",
    "am_energy_70hz",
    "roughness_x_sharpness",
]


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


def _analyze_wav_features(wav: Path, scorer: UnpleasantnessScorer) -> tuple[float, dict[str, float]]:
    sig, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    x = np.asarray(sig, dtype=np.float32)
    if x.ndim == 2:
        x = np.mean(x, axis=1, dtype=np.float32)

    layers: list[dict] | None = None
    duration_s = float(x.shape[0]) / float(int(sr)) if int(sr) > 0 else 5.0
    sidecar = wav.with_name(f"{wav.stem}.params.json")
    if sidecar.exists():
        try:
            with sidecar.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                raw_layers = payload.get("layers", [])
                if isinstance(raw_layers, list):
                    layers = [layer for layer in raw_layers if isinstance(layer, dict)]
                duration_s = float(payload.get("duration_s", duration_s))
        except Exception as exc:
            print(f"Warning: failed to parse sidecar {sidecar}: {exc}")

    score, features = scorer.score(x, int(sr), layers=layers, duration_s=duration_s)
    merged = {**UnpleasantnessScorer.DEFAULT_SYNTH_WEIGHTS}
    merged.update({k: float(v) for k, v in features.items()})
    if layers is not None:
        merged.update(compute_synth_features(layers, duration_s))
    return float(score), merged


def _analyze_wavs_batch(
    wavs: list[Path],
    scorer: UnpleasantnessScorer,
    feature_names: list[str],
) -> tuple[dict[int, dict[str, float]], np.ndarray]:
    features_by_idx: dict[int, dict[str, float]] = {}
    score_vec = np.zeros(len(wavs), dtype=np.float64)
    for idx, wav in enumerate(wavs):
        try:
            s, f = _analyze_wav_features(wav, scorer)
            features_by_idx[idx] = {k: float(v) for k, v in f.items()}
            score_vec[idx] = float(s)
        except Exception as exc:
            print(f"Warning: failed to analyze {wav}: {exc}")
            features_by_idx[idx] = {k: 0.0 for k in feature_names}
    return features_by_idx, score_vec


def _feature_vector(features: dict[str, float]) -> np.ndarray:
    rough = float(features.get("roughness", 0.0))
    sharp = float(features.get("sharpness", 0.0))
    acoustic = [
        rough,
        sharp,
        float(features.get("dissonance", 0.0)),
        float(features.get("crest_factor", 0.0)),
        float(features.get("band_energy_2_4khz", 0.0)),
        float(features.get("am_energy_70hz", 0.0)),
        rough * sharp,
    ]
    synth = [float(features.get(k, 0.0)) for k in UnpleasantnessScorer.SYNTH_FEATURE_NAMES]
    return np.array(acoustic + synth, dtype=np.float64)


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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _comparison_stats(comparisons: list[tuple[int, int, float]]) -> tuple[list[tuple[int, int, float]], int]:
    directional: list[tuple[int, int, float]] = []
    ties = 0
    for w, l, wt in comparisons:
        if wt >= 0.999:
            directional.append((int(w), int(l), float(wt)))
        else:
            ties += 1
    return directional, ties


def _prior_beta_std(
    scorer: UnpleasantnessScorer,
    feature_names: list[str],
    scales: np.ndarray,
) -> tuple[np.ndarray, float]:
    prior = np.zeros(len(feature_names), dtype=np.float64)
    model = getattr(scorer, "preference_model", None)
    if isinstance(model, dict) and isinstance(model.get("beta_std"), dict):
        beta_std = model["beta_std"]
        for i, name in enumerate(feature_names):
            prior[i] = float(beta_std.get(name, 0.0))
        intercept = float(model.get("intercept", 0.0))
        return np.clip(prior, -COEF_BOUND, COEF_BOUND), float(np.clip(intercept, -INTERCEPT_BOUND, INTERCEPT_BOUND))

    for i, name in enumerate(feature_names):
        base_w = float(scorer.weights.get(name, DEFAULT_CALIBRATION_WEIGHTS.get(name, 0.0)))
        prior[i] = base_w * scales[i]
    return np.clip(prior, -COEF_BOUND, COEF_BOUND), 0.0


def _fit_preference_model(
    xmat: np.ndarray,
    feature_names: list[str],
    comparisons: list[tuple[int, int, float]],
    scorer: UnpleasantnessScorer,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    if xmat.ndim != 2 or xmat.shape[1] != len(feature_names):
        raise ValueError("xmat shape does not match feature names")

    directional, ties = _comparison_stats(comparisons)
    if len(directional) < MIN_DIRECTIONAL_COMPARISONS:
        return None, {
            "status": "insufficient_directional",
            "n_directional": len(directional),
            "n_ties": ties,
        }

    compared_items = sorted({i for w, l, _ in directional for i in (w, l)})
    if len(compared_items) < MIN_DISTINCT_ITEMS:
        return None, {
            "status": "insufficient_items",
            "n_directional": len(directional),
            "n_items": len(compared_items),
            "n_ties": ties,
        }

    x_compared = xmat[np.array(compared_items, dtype=np.int64)]
    means = np.mean(x_compared, axis=0)
    stds_raw = np.std(x_compared, axis=0)
    scales = np.where(stds_raw > STD_EPS, stds_raw, 1.0)
    active = stds_raw > STD_EPS
    if int(np.sum(active)) < MIN_ACTIVE_FEATURES:
        return None, {
            "status": "insufficient_diversity",
            "n_directional": len(directional),
            "n_active_features": int(np.sum(active)),
            "n_ties": ties,
        }

    dmat = np.vstack([
        ((xmat[w] - xmat[l]) / scales) for (w, l, _wt) in directional
    ]).astype(np.float64)
    ywt = np.array([wt for (_w, _l, wt) in directional], dtype=np.float64)
    dmat[:, ~active] = 0.0
    contrast_mask = np.abs(dmat) > 1e-6
    contrast_counts = np.sum(contrast_mask, axis=0).astype(np.int64)
    identified_threshold = max(MIN_FEATURE_CONTRAST, int(np.ceil(0.12 * len(directional))))
    support_required = max(MIN_FEATURE_ITEM_SUPPORT, int(np.ceil(0.10 * len(compared_items))))
    value_bucket_required = MIN_FEATURE_VALUE_BUCKETS
    support_counts = np.zeros(len(feature_names), dtype=np.int64)
    distinct_value_counts = np.zeros(len(feature_names), dtype=np.int64)
    support_ok = np.zeros(len(feature_names), dtype=bool)
    for i in range(len(feature_names)):
        vals = np.round(x_compared[:, i], decimals=6)
        unique_vals, counts = np.unique(vals, return_counts=True)
        distinct_value_counts[i] = int(unique_vals.size)
        minority_support = int(len(compared_items) - int(np.max(counts))) if counts.size else 0
        support_counts[i] = minority_support
        support_ok[i] = (minority_support >= support_required) or (distinct_value_counts[i] >= value_bucket_required)

    identified = active & (contrast_counts >= identified_threshold) & support_ok

    prior_beta, prior_intercept = _prior_beta_std(scorer, feature_names, scales)
    x0 = np.concatenate([prior_beta, np.array([prior_intercept], dtype=np.float64)], axis=0)

    def objective(vec: np.ndarray) -> tuple[float, np.ndarray]:
        beta = vec[:-1]
        intercept = vec[-1]
        logits = intercept + dmat @ beta
        loss_terms = np.logaddexp(0.0, -logits)
        loss = float(np.sum(ywt * loss_terms))
        reg = 0.5 * RIDGE_ALPHA * float(np.sum((beta - prior_beta) ** 2))
        reg += 0.5 * RIDGE_ALPHA * float((intercept - prior_intercept) ** 2)
        loss += reg

        probs = _sigmoid(logits)
        grad_z = ywt * (probs - 1.0)
        grad_beta = dmat.T @ grad_z + RIDGE_ALPHA * (beta - prior_beta)
        grad_intercept = float(np.sum(grad_z) + RIDGE_ALPHA * (intercept - prior_intercept))
        grad = np.concatenate([grad_beta, np.array([grad_intercept], dtype=np.float64)], axis=0)
        return loss, grad

    bounds = [(-COEF_BOUND, COEF_BOUND)] * len(feature_names) + [(-INTERCEPT_BOUND, INTERCEPT_BOUND)]
    res = minimize(fun=lambda v: objective(v)[0], x0=x0, jac=lambda v: objective(v)[1], method="L-BFGS-B", bounds=bounds)
    if not res.success:
        return None, {
            "status": "optimization_failed",
            "message": str(res.message),
            "n_directional": len(directional),
            "n_ties": ties,
        }

    learned = np.asarray(res.x, dtype=np.float64)
    beta_hat = np.clip(learned[:-1], -COEF_BOUND, COEF_BOUND)
    intercept_hat = float(np.clip(learned[-1], -INTERCEPT_BOUND, INTERCEPT_BOUND))

    blend = float(np.clip(7.0 / (len(directional) + 7.0), 0.08, 0.50))
    beta = np.clip(prior_beta + blend * (beta_hat - prior_beta), -COEF_BOUND, COEF_BOUND)
    intercept = float(np.clip(prior_intercept + blend * (intercept_hat - prior_intercept), -INTERCEPT_BOUND, INTERCEPT_BOUND))
    beta[~identified] = prior_beta[~identified]

    logits = intercept + dmat @ beta
    pair_log_loss = float(np.mean(np.logaddexp(0.0, -logits)))
    max_delta = float(np.max(np.abs(beta - prior_beta))) if beta.size else 0.0

    feature_coverage: dict[str, dict[str, object]] = {}
    for i, name in enumerate(feature_names):
        status = "identified"
        if not bool(active[i]):
            status = "pending_variance"
        elif not bool(identified[i]):
            status = "pending_contrast"
        feature_coverage[name] = {
            "status": status,
            "active": bool(active[i]),
            "identified": bool(identified[i]),
            "contrast_count": int(contrast_counts[i]),
            "contrast_required": int(identified_threshold),
            "item_support": int(support_counts[i]),
            "item_support_required": int(support_required),
            "distinct_values": int(distinct_value_counts[i]),
            "distinct_values_required": int(value_bucket_required),
            "support_ok": bool(support_ok[i]),
            "std": float(stds_raw[i]),
        }

    model = {
        "feature_names": list(feature_names),
        "means": {name: float(means[i]) for i, name in enumerate(feature_names)},
        "scales": {name: float(max(stds_raw[i], STD_EPS)) for i, name in enumerate(feature_names)},
        "beta_std": {name: float(beta[i]) for i, name in enumerate(feature_names)},
        "intercept": float(intercept),
        "ridge_alpha": float(RIDGE_ALPHA),
        "n_directional": int(len(directional)),
        "n_ties": int(ties),
        "pair_log_loss": pair_log_loss,
        "max_coef_delta": max_delta,
        "blend": blend,
        "active_features": [feature_names[i] for i in np.where(active)[0]],
        "identified_features": [feature_names[i] for i in np.where(identified)[0]],
        "frozen_features": [feature_names[i] for i in np.where(~identified)[0]],
        "feature_coverage": feature_coverage,
    }
    diagnostics: dict[str, object] = {
        "status": "updated",
        "n_directional": len(directional),
        "n_ties": ties,
        "n_items": len(compared_items),
        "n_active_features": int(np.sum(active)),
        "n_identified_features": int(np.sum(identified)),
        "pair_log_loss": pair_log_loss,
        "max_coef_delta": max_delta,
        "blend": blend,
    }
    return model, diagnostics


def _legacy_weights_from_model(feature_names: list[str], model: dict[str, object]) -> dict[str, float]:
    means = model.get("means", {})
    scales = model.get("scales", {})
    beta_std = model.get("beta_std", {})
    weights = dict(DEFAULT_CALIBRATION_WEIGHTS)
    for name in feature_names:
        scale = max(float(scales.get(name, 1.0)), STD_EPS)
        raw = float(beta_std.get(name, 0.0)) / scale
        raw = float(np.clip(raw, -COEF_BOUND, COEF_BOUND))
        if name in ACOUSTIC_FEATURE_NAMES:
            raw = max(0.0, raw)
        weights[name] = raw
    _ = means
    return weights


def run_ab_session(
    wavs: list[Path],
    scorer: "UnpleasantnessScorer",
    n_pairs: int = 6,
    strategy: str = "smart",
    no_play: bool = False,
    seed: int = 0,
    accumulated_rows: list[tuple[int, int, float]] | None = None,
    accumulated_items: list[dict[str, float]] | None = None,
) -> tuple[dict[str, object] | None, list[tuple[int, int, float]], list[dict[str, float]]]:
    """Run a mini A-B session and fit on accumulated history.

    Returns (calibrated_weights_or_none, updated_accumulated_rows, updated_accumulated_items).
    """
    rng = np.random.default_rng(seed)
    old_weights = dict(scorer.weights)
    feature_names = list(FEATURE_NAMES)
    all_rows: list[tuple[int, int, float]] = list(accumulated_rows) if accumulated_rows is not None else []
    all_items: list[dict[str, float]] = [dict(it) for it in accumulated_items] if accumulated_items is not None else []
    item_offset = len(all_items)

    features_by_idx, score_vec = _analyze_wavs_batch(wavs, scorer, feature_names)

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
    if compared < MIN_DIRECTIONAL_COMPARISONS:
        print("[A-B] Too little cumulative evidence; keeping current model.")
        print(f"[A-B] Pending evidence: need >= {MIN_DIRECTIONAL_COMPARISONS} directional comparisons, have {compared}.")
        return None, all_rows, all_items

    n_items = len(all_items)
    xmat = np.vstack([_feature_vector(all_items[i]) for i in range(n_items)])
    model, diag = _fit_preference_model(
        xmat=xmat,
        feature_names=feature_names,
        comparisons=all_rows,
        scorer=scorer,
    )
    if model is None:
        print(f"[A-B] Calibration skipped ({diag.get('status')}).")
        if diag.get("status") == "insufficient_diversity":
            print("[A-B] Pending evidence: increase candidate diversity so more features vary.")
        return None, all_rows, all_items

    calibrated = _legacy_weights_from_model(feature_names, model)
    print("[A-B] Preference model updated:")
    print(
        f"  directional={diag.get('n_directional')} ties={diag.get('n_ties')} "
        f"active_features={diag.get('n_active_features')} identified_features={diag.get('n_identified_features')}"
    )
    print(
        f"  pair_log_loss={float(diag.get('pair_log_loss', 0.0)):.4f} "
        f"max_coef_delta={float(diag.get('max_coef_delta', 0.0)):.4f}"
    )
    identified_set = set(model.get("identified_features", model.get("active_features", [])))
    changed = sorted(
        [
            name
            for name in feature_names
            if abs(float(calibrated.get(name, 0.0)) - float(old_weights.get(name, 0.0))) > 1e-4
            and name in identified_set
        ],
        key=lambda n: abs(float(calibrated.get(n, 0.0)) - float(old_weights.get(n, 0.0))),
        reverse=True,
    )
    print("[A-B] Updated identifiable effects:")
    for name in changed[:10]:
        print(f"  {name}: {old_weights.get(name, 0.0):.4f} -> {calibrated.get(name, 0.0):.4f}")
    coverage = model.get("feature_coverage", {})
    if isinstance(coverage, dict):
        pending = [name for name, meta in coverage.items() if isinstance(meta, dict) and meta.get("status") != "identified"]
        if pending:
            print(f"[A-B] Pending evidence features: {len(pending)}")
            synth_pending = [name for name in UnpleasantnessScorer.SYNTH_FEATURE_NAMES if name in set(pending)]
            if synth_pending:
                print("[A-B] Pending synth evidence:")
                for name in synth_pending:
                    status = str(coverage.get(name, {}).get("status", "pending"))
                    print(f"  {name}: {status}")

    payload: dict[str, object] = {
        "version": 2,
        "weights": calibrated,
        "preference_model": model,
    }
    return payload, all_rows, all_items


def _payload_from_file(path: str | None) -> dict[str, object]:
    if not path:
        return {
            "version": 2,
            "weights": dict(DEFAULT_CALIBRATION_WEIGHTS),
            "preference_model": None,
        }
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("weights-in must be a JSON object")
    if "weights" in data or "preference_model" in data:
        weights = dict(DEFAULT_CALIBRATION_WEIGHTS)
        raw = data.get("weights", {})
        if isinstance(raw, dict):
            for k in weights:
                if k in raw:
                    weights[k] = float(raw[k])
        return {
            "version": int(data.get("version", 2)),
            "weights": weights,
            "preference_model": data.get("preference_model") if isinstance(data.get("preference_model"), dict) else None,
        }
    weights = dict(DEFAULT_CALIBRATION_WEIGHTS)
    for k in weights:
        if k in data:
            weights[k] = float(data[k])
    return {
        "version": 1,
        "weights": weights,
        "preference_model": None,
    }


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
    print("\nTop coefficient deltas (old -> new):")
    if new_w is None:
        print("  (unchanged)")
        return
    deltas = sorted(
        ((k, old_w.get(k, 0.0), new_w.get(k, 0.0), abs(new_w.get(k, 0.0) - old_w.get(k, 0.0))) for k in old_w),
        key=lambda x: x[3],
        reverse=True,
    )
    shown = 0
    for k, ov, nv, dv in deltas:
        if dv < 1e-5:
            continue
        print(f"  {k}: {ov:.6f} -> {nv:.6f}")
        shown += 1
        if shown >= 10:
            break
    if shown == 0:
        print("  no material changes")


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
        loaded = _payload_from_file(args.weights_in)
        old_weights = dict(loaded["weights"])
        scorer = UnpleasantnessScorer(
            weights=old_weights,
            preference_model=loaded.get("preference_model") if isinstance(loaded, dict) else None,
        )
    except Exception as exc:
        print(f"Failed to load weights: {exc}")
        sys.exit(1)

    features_by_idx, score_vec = _analyze_wavs_batch(wavs, scorer, FEATURE_NAMES)
    initial_scores = {str(wavs[i]): float(score_vec[i]) for i in range(len(wavs))}

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

    calibrated_payload: dict[str, object] | None = None
    calibrated_weights: dict[str, float] | None = None
    if compared < MIN_DIRECTIONAL_COMPARISONS:
        print(
            f"Warning: fewer than {MIN_DIRECTIONAL_COMPARISONS} non-skipped directional pairs collected; skipping calibration."
        )
    else:
        xmat = np.vstack([_feature_vector(features_by_idx[i]) for i in range(len(wavs))])
        model, diag = _fit_preference_model(
            xmat=xmat,
            feature_names=FEATURE_NAMES,
            comparisons=bt_rows,
            scorer=scorer,
        )
        if model is None:
            print(f"Calibration skipped: {diag.get('status')}")
        else:
            calibrated_weights = _legacy_weights_from_model(FEATURE_NAMES, model)
            calibrated_payload = {
                "version": 2,
                "weights": calibrated_weights,
                "preference_model": model,
            }
            print(
                "Calibration diagnostics: "
                f"pair_log_loss={float(diag.get('pair_log_loss', 0.0)):.4f}, "
                f"max_coef_delta={float(diag.get('max_coef_delta', 0.0)):.4f}"
            )

    weights_out = Path(args.weights_out)
    try:
        weights_out.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        out_payload: dict[str, object] = (
            calibrated_payload
            if calibrated_payload is not None
            else {
                "version": int(loaded.get("version", 2)) if isinstance(loaded, dict) else 2,
                "weights": old_weights,
                "preference_model": loaded.get("preference_model") if isinstance(loaded, dict) else None,
            }
        )
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
            "payload_version": int((calibrated_payload or loaded).get("version", 1)),
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
