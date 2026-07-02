"""Parameter sweep for maximizing unpleasantness score."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from abtool import run_ab_session
from dissonance.analysis.scorer import UnpleasantnessScorer
from dissonance.core.mixer import mix


WORKER_SEED: int = 42


ROUGH_SPACE: dict[str, list[Any]] = {
    "carrier_hz": [1500, 2000, 2500, 3000, 3500, 4000],
    "n_partials": [4, 8, 12, 16],
    "partial_spread_bark": [0.1, 0.25, 0.5, 0.75],
    "am_rate_hz": [40, 60, 70, 80, 100, 120],
    "am_depth": [0.5, 0.7, 0.9, 1.0],
}

STICKSLIP_SPACE: dict[str, list[Any]] = {
    "ioi_mean_ms": [2.0, 4.0, 6.0, 8.0],
    "ioi_jitter": [0.3, 0.5, 0.7, 0.9],
    "resonance_hz": [
        [2400, 3300, 4100],
        [2000, 3000, 4000],
        [2200, 3100, 4400],
        [2800, 3600, 4500],
    ],
}

FM_INSTAB_SPACE: dict[str, list[Any]] = {
    "carrier_hz": [2500, 3000, 3500, 4000],
    "mod_rate_hz": [8, 12, 16, 20],
    "mod_index": [4, 8, 12, 16],
    "mod_chaos": [0.3, 0.5, 0.7, 1.0],
}

INHARMONIC_SPACE: dict[str, list[Any]] = {
    "base_hz": [150, 200, 300, 400],
    "n_partials": [8, 12, 16, 20],
    "inharmonicity_B": [0.02, 0.06, 0.12, 0.2],
    "random_detune": [0.2, 0.5, 0.8, 1.0],
}

BEATING_SPACE: dict[str, list[Any]] = {
    "base_hz": [150, 220, 330, 440],
    "n_beaters": [2, 3, 4, 5],
    "beat_rate_hz": [5, 7, 10, 14],
    "beat_jitter": [0.1, 0.3, 0.6, 0.9],
}

NOISE_SHAPED_SPACE: dict[str, list[Any]] = {
    "center_hz": [2500, 3150, 4000, 5000],
    "bandwidth_hz": [1000, 2000, 3000],
    "modulation_rate_hz": [40, 60, 70, 100],
    "modulation_depth": [0.4, 0.6, 0.8, 1.0],
}

GLOBAL_SPACE: dict[str, list[Any]] = {
    "hump_2_4khz_db": [6, 9, 12, 15],
    "highpass_hz": [400, 800, 1200],
}


@dataclass(slots=True)
class EvalResult:
    """Single parameter evaluation result."""

    score: float
    params: dict[str, Any]
    features: dict[str, Any]


def init_worker(seed: int) -> None:
    """Initialize worker RNG state."""
    global WORKER_SEED
    WORKER_SEED = int(seed)
    np.random.seed(WORKER_SEED + os.getpid())


def to_serializable(obj: Any) -> Any:
    """Convert nested structures to JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def build_layers(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Build mixer layers from parameter dictionary."""
    rough = {"type": "rough", "gain_db": -6.0, **params["rough"]}
    stickslip = {"type": "stickslip", "gain_db": -3.0, **params["stickslip"]}
    fm_instab = {"type": "fm_instab", "gain_db": -9.0, **params["fm_instab"]}
    inharmonic = {"type": "inharmonic", "gain_db": -9.0, **params["inharmonic"]}
    beating = {"type": "beating", "gain_db": -6.0, **params["beating"]}
    noise_shaped = {"type": "noise_shaped", "gain_db": -12.0, **params["noise_shaped"]}
    return [rough, stickslip, fm_instab, inharmonic, beating, noise_shaped]


def evaluate_params(
    params: dict[str, Any],
    sr: int,
    duration_s: float,
    scorer: UnpleasantnessScorer | None = None,
) -> EvalResult:
    """Evaluate one parameter set, guarding against runtime errors."""
    try:
        layers = build_layers(params)
        signal = mix(
            layers=layers,
            sr=sr,
            duration_s=duration_s,
            global_params=params["global"],
        )
        _scorer = scorer if scorer is not None else UnpleasantnessScorer()
        score, features = _scorer.score(signal, sr)
        return EvalResult(float(score), params, dict(features))
    except Exception:
        return EvalResult(0.0, params, {})


def evaluate_worker(task: tuple[dict[str, Any], int, float]) -> EvalResult:
    """Top-level worker entrypoint for ProcessPoolExecutor."""
    params, sr, duration_s = task
    return evaluate_params(params=params, sr=sr, duration_s=duration_s)


def sample_one(rng: np.random.Generator) -> dict[str, Any]:
    """Sample one parameter combination uniformly from discrete search space."""
    def _sample_group(space: dict[str, list[Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in space.items():
            val = rng.choice(v)
            out[k] = copy.deepcopy(val.item() if isinstance(val, np.generic) else val)
        return out

    return {
        "rough": _sample_group(ROUGH_SPACE),
        "stickslip": _sample_group(STICKSLIP_SPACE),
        "fm_instab": _sample_group(FM_INSTAB_SPACE),
        "inharmonic": _sample_group(INHARMONIC_SPACE),
        "beating": _sample_group(BEATING_SPACE),
        "noise_shaped": _sample_group(NOISE_SHAPED_SPACE),
        "global": _sample_group(GLOBAL_SPACE),
    }


def format_score_line(done: int, total: int, score: float, best: float) -> str:
    """Format progress line."""
    return f"[{done:4d}/{total}] score={score:.3f}  best={best:.3f}"


def print_top_results(results: list[EvalResult], count: int = 5) -> None:
    """Print compact top-N table."""
    print("\nPhase 1 top results:")
    print("rank  score   rough(carrier/am)  stick(ioi/jitter)  fm(carrier/index)  global(hump/hp)")
    for idx, r in enumerate(results[:count], start=1):
        p = r.params
        rough = p.get("rough", {})
        stickslip = p.get("stickslip", {})
        fm_instab = p.get("fm_instab", {})
        global_p = p.get("global", {})
        print(
            f"{idx:>4d}  {r.score:>6.3f}  "
            f"{rough.get('carrier_hz', '-'):>4}/{rough.get('am_rate_hz', '-'):>3}  "
            f"{stickslip.get('ioi_mean_ms', '-'):>3}/{stickslip.get('ioi_jitter', '-'):<3}  "
            f"{fm_instab.get('carrier_hz', '-'):>4}/{fm_instab.get('mod_index', '-'):>2}  "
            f"{global_p.get('hump_2_4khz_db', '-'):>2}/{global_p.get('highpass_hz', '-'):>4}"
        )


def phase1_random_sampling(
    samples: int,
    sr: int,
    duration_s: float,
    workers: int,
    seed: int,
    ab_interval: int = 0,
    ab_pairs: int = 6,
    ab_no_play: bool = False,
    out_dir: Path | None = None,
) -> tuple[list[EvalResult], UnpleasantnessScorer]:
    """Run parallel random sampling and return all results sorted descending by score."""
    rng = np.random.default_rng(seed)

    results: list[EvalResult] = []
    best = 0.0
    done = 0
    scorer = UnpleasantnessScorer()

    if ab_interval <= 0:
        tasks = [(sample_one(rng), sr, duration_s) for _ in range(samples)]

        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(seed,)) as ex:
            futs = [ex.submit(evaluate_worker, task) for task in tasks]
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                done += 1
                if res.score > best:
                    best = res.score
                if done % 10 == 0 or done == samples:
                    print(format_score_line(done, samples, res.score, best), flush=True)

        results.sort(key=lambda x: x.score, reverse=True)
        return results, scorer

    n_batches = int(math.ceil(samples / float(ab_interval)))
    ab_tmp_dir = out_dir / "ab_tmp" if out_dir is not None else None
    ab_accumulated_rows: list[tuple[int, int, float]] = []
    ab_accumulated_items: list[dict[str, float]] = []

    for batch_idx in range(n_batches):
        batch_size = min(ab_interval, samples - done)
        if batch_size <= 0:
            break
        batch_tasks = [(sample_one(rng), sr, duration_s) for _ in range(batch_size)]

        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(seed,)) as ex:
            futs = [ex.submit(evaluate_worker, task) for task in batch_tasks]
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                done += 1
                if res.score > best:
                    best = res.score
                if done % 10 == 0 or done == samples:
                    print(format_score_line(done, samples, res.score, best), flush=True)

        results.sort(key=lambda x: x.score, reverse=True)

        if ab_tmp_dir is None:
            continue

        topn = min(3, len(results))
        if topn < 2:
            continue
        ab_tmp_dir.mkdir(parents=True, exist_ok=True)
        wavs: list[Path] = []
        for rank in range(topn):
            wav = ab_tmp_dir / f"batch{batch_idx + 1:03d}_top{rank + 1}.wav"
            render_result(results[rank].params, out_wav=wav, sr=sr, duration_s=duration_s)
            wavs.append(wav)

        updated_weights, ab_accumulated_rows, ab_accumulated_items = run_ab_session(
            wavs=wavs,
            scorer=scorer,
            n_pairs=ab_pairs,
            strategy="smart",
            no_play=ab_no_play,
            seed=seed + batch_idx,
            accumulated_rows=ab_accumulated_rows,
            accumulated_items=ab_accumulated_items,
        )
        if updated_weights is None:
            continue

        scorer = UnpleasantnessScorer(weights=updated_weights)
        results = [evaluate_params(r.params, sr=sr, duration_s=duration_s, scorer=scorer) for r in results]
        results.sort(key=lambda x: x.score, reverse=True)
        best = results[0].score if results else 0.0
        print(f"[A-B] Weights updated after batch {batch_idx + 1}", flush=True)

    results.sort(key=lambda x: x.score, reverse=True)
    return results, scorer


def iter_param_alternatives(group: str, key: str, current: Any) -> list[Any]:
    """Return all alternative values for one coordinate."""
    space_map = {
        "rough": ROUGH_SPACE,
        "stickslip": STICKSLIP_SPACE,
        "fm_instab": FM_INSTAB_SPACE,
        "inharmonic": INHARMONIC_SPACE,
        "beating": BEATING_SPACE,
        "noise_shaped": NOISE_SHAPED_SPACE,
        "global": GLOBAL_SPACE,
    }
    values = space_map[group][key]
    def _neq(a: Any, b: Any) -> bool:
        """Safe inequality: handles list/numpy values."""
        try:
            result = a != b
            if isinstance(result, (bool, np.bool_)):
                return bool(result)
            return True  # non-scalar comparison (e.g. list vs list) — include it
        except Exception:
            return True
    return [copy.deepcopy(v) for v in values if _neq(v, current)]


def hill_climb_one(
    start: EvalResult,
    sr: int,
    duration_s: float,
    iters: int,
    scorer: UnpleasantnessScorer | None = None,
) -> EvalResult:
    """Coordinate-descent hill climb from one starting candidate."""
    current = EvalResult(start.score, copy.deepcopy(start.params), copy.deepcopy(start.features))
    groups_order = ["rough", "stickslip", "fm_instab", "inharmonic", "beating", "noise_shaped", "global"]

    for it in range(iters):
        improved_any = False
        for group in groups_order:
            for key in current.params[group].keys():
                best_local = current
                cur_val = current.params[group][key]
                for alt in iter_param_alternatives(group, key, cur_val):
                    trial_params = copy.deepcopy(current.params)
                    trial_params[group][key] = alt
                    trial = evaluate_params(trial_params, sr=sr, duration_s=duration_s, scorer=scorer)
                    if trial.score > best_local.score:
                        best_local = trial

                if best_local.score > current.score:
                    old = current.score
                    current = best_local
                    improved_any = True
                    print(
                        f"  iter {it + 1}: improved {group}.{key} -> {current.params[group][key]} "
                        f"{old:.3f} -> {current.score:.3f}",
                        flush=True,
                    )
        if not improved_any:
            break
    return current


def render_result(params: dict[str, Any], out_wav: Path, sr: int, duration_s: float) -> None:
    """Render a sweep result to a WAV file via direct mix() call."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    signal = mix(
        layers=build_layers(params),
        sr=sr,
        duration_s=duration_s,
        global_params=params["global"],
    )
    sf.write(str(out_wav), signal, samplerate=sr)


def build_result_preset(
    params: dict[str, Any],
    sr: int,
    duration_s: float,
    score: float,
    rank: int,
    features: dict[str, Any],
) -> dict[str, Any]:
    """Build a gen-compatible preset payload for one sweep result."""
    return {
        "duration_s": to_serializable(float(duration_s)),
        "sample_rate": to_serializable(int(sr)),
        "global": to_serializable(params["global"]),
        "layers": to_serializable(build_layers(params)),
        "_sweep_meta": {
            "score": to_serializable(score),
            "rank": to_serializable(rank),
            "features": to_serializable(features),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Sweep generator params for maximum unpleasantness.")
    parser.add_argument("--samples", type=int, default=200, help="Number of random samples for phase 1.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K candidates to hill-climb in phase 2.")
    parser.add_argument("--hill-climb-iters", type=int, default=3, help="Coordinate descent passes per candidate.")
    parser.add_argument("--duration", type=float, default=2.0, help="Audio duration in seconds per evaluation.")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate.")
    parser.add_argument("--out-dir", type=str, default="./sweep_results", help="Output directory.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min((os.cpu_count() or 1), 4),
        help="Parallel workers for phase 1.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-save", action="store_true", help="Skip saving audio; save JSON report only.")
    parser.add_argument(
        "--ab-interval",
        type=int,
        default=0,
        help="Every N samples in phase 1, pause for A-B calibration. 0 = disabled.",
    )
    parser.add_argument(
        "--ab-pairs",
        type=int,
        default=6,
        help="Number of A-B pairs per calibration round.",
    )
    parser.add_argument(
        "--ab-no-play",
        action="store_true",
        dest="ab_no_play",
        help="Pass --no-play to A-B tool (headless).",
    )
    return parser.parse_args()


def main() -> None:
    """Run two-phase sweep and write report."""
    args = parse_args()
    scorer = UnpleasantnessScorer()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 1: random sampling")
    phase1, scorer = phase1_random_sampling(
        samples=args.samples,
        sr=args.sr,
        duration_s=args.duration,
        workers=max(1, args.workers),
        seed=args.seed,
        ab_interval=args.ab_interval,
        ab_pairs=args.ab_pairs,
        ab_no_play=args.ab_no_play,
        out_dir=out_dir,
    )
    print_top_results(phase1, count=min(5, len(phase1)))

    k = min(args.top_k, len(phase1))
    starts = phase1[:k]
    improved: list[EvalResult] = []

    print("\nPhase 2: local hill-climbing")
    for idx, start in enumerate(starts, start=1):
        print(f"Start #{idx}: seed score={start.score:.3f}")
        best_local = hill_climb_one(
            start=start,
            sr=args.sr,
            duration_s=args.duration,
            iters=args.hill_climb_iters,
            scorer=scorer,
        )
        improved.append(best_local)
        print(f"End #{idx}: best score={best_local.score:.3f}\n")

    final_pool = improved if improved else phase1[:1]
    final_pool_sorted = sorted(final_pool, key=lambda x: x.score, reverse=True)
    best = final_pool_sorted[0]

    print("Final best result:")
    print(f"score={best.score:.6f}")
    print("features:")
    for kf, vf in sorted(best.features.items()):
        print(f"  {kf}: {vf}")

    report = {
        "best_score": best.score,
        "best_params": to_serializable(best.params),
        "best_features": to_serializable(best.features),
        "top_k_results": [
            {
                "score": r.score,
                "params": to_serializable(r.params),
                "features": to_serializable(r.features),
            }
            for r in final_pool_sorted[:k]
        ],
        "sweep_config": {
            "samples": args.samples,
            "top_k": args.top_k,
            "hill_climb_iters": args.hill_climb_iters,
            "duration": args.duration,
            "sr": args.sr,
            "out_dir": str(out_dir),
            "workers": args.workers,
            "seed": args.seed,
            "no_save": args.no_save,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not args.no_save:
        for rank, result in enumerate(final_pool_sorted[:k], start=1):
            tag = "best" if rank == 1 else f"top{rank}"
            out_wav = out_dir / f"{tag}.wav"
            out_json = out_dir / f"{tag}.json"
            render_result(result.params, out_wav=out_wav, sr=args.sr, duration_s=args.duration)
            preset_payload = build_result_preset(
                params=result.params,
                sr=args.sr,
                duration_s=args.duration,
                score=result.score,
                rank=rank,
                features=result.features,
            )
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(preset_payload, f, indent=2)
            print(f"Saved rank-{rank} audio (score={result.score:.4f}): {out_wav} + {out_json.name}")

    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
