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

from abtool import FEATURE_NAMES, _feature_vector, run_ab_session
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

SCREAM_CHAOS_SPACE: dict[str, list[Any]] = {
    "carrier_hz": [500, 700, 900, 1200],
    "subharmonic_gain": [0.2, 0.4, 0.6, 0.8],
    "chaos_amount": [0.3, 0.5, 0.7, 0.9],
    "pitch_jump_rate_hz": [0.2, 0.5, 0.8, 1.2],
    "biphonation_ratio": [1.31, 1.52, 1.67, 1.83],
    "biphonation_gain": [0.1, 0.2, 0.3, 0.5],
}

DREAD_SWELL_SPACE: dict[str, list[Any]] = {
    "start_hz": [120, 160, 200, 260],
    "end_hz": [900, 1200, 1600, 2200],
    "rise_shape": ["linear", "exponential"],
    "roughness_rise": [True, False],
    "loudness_rise": [True, False],
}

SHEPARD_ASCENT_SPACE: dict[str, list[Any]] = {
    "n_voices": [6, 8, 10, 12],
    "octave_spread": [3.0, 4.0, 5.0],
    "rise_rate_hz_per_s": [0.25, 0.5, 0.75],
    "roughness_am_hz": [0.0, 20.0, 40.0, 70.0],
}

PULSE_PANIC_SPACE: dict[str, list[Any]] = {
    "start_rate_hz": [0.5, 1.0, 2.0],
    "end_rate_hz": [5.0, 8.0, 12.0],
    "acceleration": ["linear", "exponential"],
    "roughness_am_hz": [30.0, 50.0, 70.0, 90.0],
    "burst_decay_ms": [5.0, 8.0, 12.0, 20.0],
}

DOOM_THROB_SPACE: dict[str, list[Any]] = {
    "center_hz": [20.0, 30.0, 40.0, 60.0],
    "detune_hz_start": [0.2, 0.5, 1.0],
    "detune_hz_end": [1.5, 3.0, 4.5, 6.0],
    "am_rate_hz": [0.15, 0.3, 0.5],
}

WOBBLE_DRIFT_SPACE: dict[str, list[Any]] = {
    "base_hz": [120.0, 200.0, 300.0, 500.0],
    "detune_start_hz": [0.1, 0.5, 1.0],
    "detune_end_hz": [6.0, 12.0, 20.0, 30.0],
    "drift_shape": ["linear", "exponential"],
    "n_harmonics": [2, 4, 6, 8],
}

UNCANNY_MORPH_SPACE: dict[str, list[Any]] = {
    "base_hz": [120.0, 200.0, 300.0],
    "n_partials": [8, 12, 16, 20],
    "inharmonicity_start": [0.0, 0.02, 0.05],
    "inharmonicity_end": [0.1, 0.2, 0.3, 0.4],
    "formant_sweep": [True, False],
}

GLOBAL_SPACE: dict[str, list[Any]] = {
    "hump_2_4khz_db": [6, 9, 12, 15],
    "highpass_hz": [400, 800, 1200],
}

TEMPORAL_GROUPS: tuple[str, ...] = (
    "scream_chaos",
    "dread_swell",
    "shepard_ascent",
    "pulse_panic",
    "doom_throb",
    "wobble_drift",
    "uncanny_morph",
)

TEMPORAL_LAYER_GAINS: dict[str, float] = {
    "scream_chaos": -8.0,
    "dread_swell": -8.0,
    "shepard_ascent": -10.0,
    "pulse_panic": -9.0,
    "doom_throb": -6.0,
    "wobble_drift": -8.0,
    "uncanny_morph": -9.0,
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
    layers = [
        rough,
        stickslip,
        fm_instab,
        inharmonic,
        beating,
        noise_shaped,
    ]
    for temporal_group in TEMPORAL_GROUPS:
        group_params = params.get(temporal_group)
        if not isinstance(group_params, dict):
            continue
        layers.append({
            "type": temporal_group,
            "gain_db": float(TEMPORAL_LAYER_GAINS[temporal_group]),
            **group_params,
        })
    return layers


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
        score, features = _scorer.score(signal, sr, layers=layers, duration_s=duration_s)
        return EvalResult(float(score), params, dict(features))
    except Exception:
        return EvalResult(0.0, params, {})


def evaluate_worker(task: tuple[dict[str, Any], int, float]) -> EvalResult:
    """Top-level worker entrypoint for ProcessPoolExecutor."""
    params, sr, duration_s = task
    return evaluate_params(params=params, sr=sr, duration_s=duration_s)


def _sample_group(rng: np.random.Generator, space: dict[str, list[Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in space.items():
        val = rng.choice(v)
        out[k] = copy.deepcopy(val.item() if isinstance(val, np.generic) else val)
    return out


def _sample_active_temporal_groups(
    rng: np.random.Generator,
    min_active: int,
    max_active: int,
    activation_p: float,
) -> list[str]:
    groups = list(TEMPORAL_GROUPS)
    n_groups = len(groups)
    lo = max(0, min(int(min_active), n_groups))
    hi = max(lo, min(int(max_active), n_groups))
    p = float(np.clip(float(activation_p), 0.0, 1.0))

    active = [g for g in groups if rng.random() < p]
    inactive = [g for g in groups if g not in active]

    if len(active) < lo and inactive:
        need = lo - len(active)
        add_idx = rng.permutation(len(inactive))[:need]
        active.extend(inactive[int(i)] for i in add_idx)
    if len(active) > hi:
        keep_idx = rng.permutation(len(active))[:hi]
        active = [active[int(i)] for i in keep_idx]
    return sorted(active)


def sample_one(
    rng: np.random.Generator,
    temporal_min_active: int = 0,
    temporal_max_active: int = 3,
    temporal_activation_p: float = 0.45,
) -> dict[str, Any]:
    """Sample one parameter combination uniformly from discrete search space."""
    sampled = {
        "rough": _sample_group(rng, ROUGH_SPACE),
        "stickslip": _sample_group(rng, STICKSLIP_SPACE),
        "fm_instab": _sample_group(rng, FM_INSTAB_SPACE),
        "inharmonic": _sample_group(rng, INHARMONIC_SPACE),
        "beating": _sample_group(rng, BEATING_SPACE),
        "noise_shaped": _sample_group(rng, NOISE_SHAPED_SPACE),
        "global": _sample_group(rng, GLOBAL_SPACE),
    }
    temporal_spaces: dict[str, dict[str, list[Any]]] = {
        "scream_chaos": SCREAM_CHAOS_SPACE,
        "dread_swell": DREAD_SWELL_SPACE,
        "shepard_ascent": SHEPARD_ASCENT_SPACE,
        "pulse_panic": PULSE_PANIC_SPACE,
        "doom_throb": DOOM_THROB_SPACE,
        "wobble_drift": WOBBLE_DRIFT_SPACE,
        "uncanny_morph": UNCANNY_MORPH_SPACE,
    }
    active_temporal = _sample_active_temporal_groups(
        rng=rng,
        min_active=temporal_min_active,
        max_active=temporal_max_active,
        activation_p=temporal_activation_p,
    )
    for group in active_temporal:
        sampled[group] = _sample_group(rng, temporal_spaces[group])
    return sampled


def format_score_line(done: int, total: int, score: float, best: float) -> str:
    """Format progress line."""
    return f"[{done:4d}/{total}] score={score:.3f}  best={best:.3f}"


def _select_ab_probe_indices(
    results: list[EvalResult],
    scorer: UnpleasantnessScorer,
    rng: np.random.Generator,
    pick_count: int = 6,
) -> list[int]:
    if len(results) <= pick_count:
        return list(range(len(results)))

    top_pool = min(len(results), max(48, pick_count * 10))
    pool = results[:top_pool]
    feats = np.vstack([_feature_vector(r.features) for r in pool])
    stds = np.std(feats, axis=0)
    active_idx = np.where(stds > 1e-3)[0]
    if active_idx.size == 0:
        return list(range(min(pick_count, len(pool))))

    z = feats[:, active_idx]
    z = (z - np.mean(z, axis=0)) / np.maximum(np.std(z, axis=0), 1e-3)

    uncertainty = np.zeros(len(pool), dtype=np.float64)
    pending_synth: list[str] = []
    model = getattr(scorer, "preference_model", None)
    if isinstance(model, dict) and isinstance(model.get("beta_std"), dict):
        beta_map = model["beta_std"]
        beta = np.array([float(beta_map.get(FEATURE_NAMES[i], 0.0)) for i in active_idx], dtype=np.float64)
        logits = z @ beta
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))
        uncertainty = 1.0 - np.abs(probs - 0.5) * 2.0
        coverage = model.get("feature_coverage")
        if isinstance(coverage, dict):
            pending_ranked: list[tuple[int, str]] = []
            for name in UnpleasantnessScorer.SYNTH_FEATURE_NAMES:
                meta = coverage.get(name)
                if not isinstance(meta, dict):
                    continue
                if str(meta.get("status", "")) == "identified":
                    continue
                contrast_count = int(meta.get("contrast_count", 0))
                pending_ranked.append((contrast_count, name))
            pending_ranked.sort(key=lambda x: x[0])
            pending_synth = [name for _, name in pending_ranked]

    selected: list[int] = [0]
    if pending_synth:
        used = set(selected)
        score_vals = np.array([float(r.score) for r in pool], dtype=np.float64)
        denom = max(float(np.std(score_vals)), 1e-6)
        score_z = (score_vals - float(np.mean(score_vals))) / denom
        for synth_name in pending_synth:
            if len(selected) >= min(pick_count, len(pool)):
                break
            if synth_name not in UnpleasantnessScorer.SYNTH_FEATURE_NAMES:
                continue
            hi_i = -1
            hi_val = -1e12
            lo_i = -1
            lo_val = 1e12
            for i, res in enumerate(pool):
                if i in used:
                    continue
                v = float(res.features.get(synth_name, 0.0))
                merit_hi = v + 0.08 * float(score_z[i]) + 0.04 * float(uncertainty[i])
                merit_lo = v - 0.08 * float(score_z[i])
                if merit_hi > hi_val:
                    hi_val = merit_hi
                    hi_i = i
                if merit_lo < lo_val:
                    lo_val = merit_lo
                    lo_i = i
            if hi_i >= 0:
                selected.append(hi_i)
                used.add(hi_i)
            if len(selected) >= min(pick_count, len(pool)):
                break
            if lo_i >= 0 and lo_i not in used:
                selected.append(lo_i)
                used.add(lo_i)

    while len(selected) < min(pick_count, len(pool)):
        best_i = -1
        best_score = -1e12
        for i in range(len(pool)):
            if i in selected:
                continue
            min_dist = min(float(np.linalg.norm(z[i] - z[j])) for j in selected)
            score = min_dist + 0.35 * float(uncertainty[i]) + 0.05 * float(rng.random())
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            break
        selected.append(best_i)
    return selected


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
    temporal_min_active: int = 0,
    temporal_max_active: int = 3,
    temporal_activation_p: float = 0.45,
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

    def _sample_task() -> tuple[dict[str, Any], int, float]:
        return (
            sample_one(
                rng,
                temporal_min_active=temporal_min_active,
                temporal_max_active=temporal_max_active,
                temporal_activation_p=temporal_activation_p,
            ),
            sr,
            duration_s,
        )

    if ab_interval <= 0:
        tasks = [_sample_task() for _ in range(samples)]

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
        batch_tasks = [_sample_task() for _ in range(batch_size)]

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

        probe_indices = _select_ab_probe_indices(results, scorer=scorer, rng=rng, pick_count=6)
        topn = len(probe_indices)
        if topn < 2:
            continue
        ab_tmp_dir.mkdir(parents=True, exist_ok=True)
        wavs: list[Path] = []
        for rank, res_idx in enumerate(probe_indices, start=1):
            wav = ab_tmp_dir / f"batch{batch_idx + 1:03d}_probe{rank:02d}.wav"
            render_result(results[res_idx].params, out_wav=wav, sr=sr, duration_s=duration_s)
            write_params_sidecar(results[res_idx].params, out_wav=wav, sr=sr, duration_s=duration_s)
            wavs.append(wav)

        updated_payload, ab_accumulated_rows, ab_accumulated_items = run_ab_session(
            wavs=wavs,
            scorer=scorer,
            n_pairs=ab_pairs,
            strategy="smart",
            no_play=ab_no_play,
            seed=seed + batch_idx,
            accumulated_rows=ab_accumulated_rows,
            accumulated_items=ab_accumulated_items,
        )
        if updated_payload is None:
            continue

        payload_weights = updated_payload.get("weights") if isinstance(updated_payload, dict) else None
        payload_model = updated_payload.get("preference_model") if isinstance(updated_payload, dict) else None
        scorer = UnpleasantnessScorer(
            weights=payload_weights if isinstance(payload_weights, dict) else scorer.weights,
            preference_model=payload_model if isinstance(payload_model, dict) else None,
        )
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
        "scream_chaos": SCREAM_CHAOS_SPACE,
        "dread_swell": DREAD_SWELL_SPACE,
        "shepard_ascent": SHEPARD_ASCENT_SPACE,
        "pulse_panic": PULSE_PANIC_SPACE,
        "doom_throb": DOOM_THROB_SPACE,
        "wobble_drift": WOBBLE_DRIFT_SPACE,
        "uncanny_morph": UNCANNY_MORPH_SPACE,
        "global": GLOBAL_SPACE,
    }
    group_space = space_map.get(group)
    if group_space is None or key not in group_space:
        return []
    values = group_space[key]
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
    groups_order = [
        "rough",
        "stickslip",
        "fm_instab",
        "inharmonic",
        "beating",
        "noise_shaped",
        "scream_chaos",
        "dread_swell",
        "shepard_ascent",
        "pulse_panic",
        "doom_throb",
        "wobble_drift",
        "uncanny_morph",
        "global",
    ]

    for it in range(iters):
        improved_any = False
        for group in groups_order:
            group_params = current.params.get(group)
            if not isinstance(group_params, dict):
                continue
            for key in list(group_params.keys()):
                best_local = current
                cur_val = group_params[key]
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


def write_params_sidecar(params: dict[str, Any], out_wav: Path, sr: int, duration_s: float) -> Path:
    """Write analysis sidecar with full layer/global metadata next to a rendered wav."""
    sidecar = out_wav.with_name(f"{out_wav.stem}.params.json")
    payload = {
        "duration_s": float(duration_s),
        "sample_rate": int(sr),
        "global": to_serializable(params.get("global", {})),
        "layers": to_serializable(build_layers(params)),
    }
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return sidecar


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
    parser.add_argument(
        "--temporal-min-active",
        type=int,
        default=0,
        help="Minimum active temporal layers per sampled candidate.",
    )
    parser.add_argument(
        "--temporal-max-active",
        type=int,
        default=3,
        help="Maximum active temporal layers per sampled candidate.",
    )
    parser.add_argument(
        "--temporal-activation-p",
        type=float,
        default=0.45,
        help="Independent activation probability for each temporal layer before min/max constraints.",
    )
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
        temporal_min_active=args.temporal_min_active,
        temporal_max_active=args.temporal_max_active,
        temporal_activation_p=args.temporal_activation_p,
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
            "temporal_min_active": args.temporal_min_active,
            "temporal_max_active": args.temporal_max_active,
            "temporal_activation_p": args.temporal_activation_p,
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
