"""Command-line interface for generation, scoring, analysis, and parameter sweep."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from dissonance.analysis.features import compute_features
from dissonance.ab_candidates import generate_ab_candidates
from dissonance.analysis.scorer import UnpleasantnessScorer
from dissonance.io.presets import PRESET_DEFAULT_GEN, get_preset
from dissonance.io.render import render_from_params


def _load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    signal, sr = sf.read(path, dtype="float32", always_2d=False)
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim == 2:
        x = np.mean(x, axis=1, dtype=np.float32)
    elif x.ndim != 1:
        raise ValueError(f"Unsupported audio shape: {x.shape}")
    return x, int(sr)


def _print_feature_table(features: dict[str, float]) -> None:
    items = list(features.items())
    key_w = max(len("Feature"), *(len(k) for k, _ in items)) if items else len("Feature")
    val_w = len("Value")
    border = f"+-{'-' * key_w}-+-{'-' * val_w}-+"
    print(border)
    print(f"| {'Feature'.ljust(key_w)} | {'Value'.rjust(val_w)} |")
    print(border)
    for key, value in items:
        print(f"| {key.ljust(key_w)} | {f'{float(value):.6f}'.rjust(val_w)} |")
    print(border)


def _cmd_gen(args: argparse.Namespace) -> None:
    if args.preset:
        preset_path = Path(args.preset)
        if preset_path.suffix.lower() == ".json" or preset_path.is_file():
            with preset_path.open("r", encoding="utf-8") as f:
                params = json.load(f)
        else:
            params = get_preset(args.preset)
    else:
        params = copy.deepcopy(PRESET_DEFAULT_GEN)

    if args.duration is not None:
        params["duration_s"] = float(args.duration)
    if args.sr is not None:
        params["sample_rate"] = int(args.sr)

    out_path = args.out
    render_from_params(params, out_path)
    duration = float(params.get("duration_s", 4.0))
    sr = int(params.get("sample_rate", 48000))
    print(f"Generated: {out_path} ({duration}s, {sr}Hz)")




def _cmd_score(args: argparse.Namespace) -> None:
    signal, sr = _load_wav_mono(args.input)
    final_score, features = UnpleasantnessScorer().score(signal, sr)
    _print_feature_table(features)
    print(f"\nFINAL UNPLEASANTNESS SCORE: {final_score:.6f}")


def _cmd_analyze(args: argparse.Namespace) -> None:
    signal, sr = _load_wav_mono(args.input)
    features = compute_features(signal, sr)
    _print_feature_table(features)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(features, f, indent=2, sort_keys=True)


def _cmd_sweep(args: argparse.Namespace) -> None:
    """Delegate to sweep.py, forwarding all arguments."""
    sweep_script = Path(__file__).parent.parent / "sweep.py"
    cmd = [
        sys.executable, str(sweep_script),
        "--samples", str(args.samples),
        "--top-k", str(args.top_k),
        "--hill-climb-iters", str(args.hill_climb_iters),
        "--duration", str(args.duration),
        "--sr", str(args.sr),
        "--out-dir", str(args.out_dir),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--temporal-min-active", str(args.temporal_min_active),
        "--temporal-max-active", str(args.temporal_max_active),
        "--temporal-activation-p", str(args.temporal_activation_p),
        "--ab-interval", str(args.ab_interval),
        "--ab-pairs", str(args.ab_pairs),
    ]
    if getattr(args, "no_save", False):
        cmd.append("--no-save")
    if getattr(args, "ab_no_play", False):
        cmd.append("--ab-no-play")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def _cmd_ab_candidates(args: argparse.Namespace) -> None:
    """Generate deterministic A/B calibration candidates that isolate temporal synth features."""
    rendered = generate_ab_candidates(
        out_dir=args.out_dir,
        duration_s=args.duration,
        sample_rate=args.sr,
        seed=args.seed,
        repeats=args.repeats,
    )
    print(f"Generated {len(rendered)} files in {Path(args.out_dir).resolve()}")





def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dissonance", description="Unpleasant sound engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("gen", help="Generate unpleasant sound (MODE A)")
    gen.add_argument("--preset", type=str, default=None)
    gen.add_argument("--out", type=str, default="generated.wav")
    gen.add_argument("--duration", type=float, default=None)
    gen.add_argument("--sr", type=int, default=None)
    gen.set_defaults(func=_cmd_gen)

    # Distortion mode removed

    score = subparsers.add_parser("score", help="Score unpleasantness of a WAV file")
    score.add_argument("--in", dest="input", type=str, required=True)
    score.set_defaults(func=_cmd_score)

    analyze = subparsers.add_parser("analyze", help="Full analysis report")
    analyze.add_argument("--in", dest="input", type=str, required=True)
    analyze.add_argument("--report", type=str, default=None)
    analyze.set_defaults(func=_cmd_analyze)

    sweep = subparsers.add_parser("sweep", help="Sweep generator params to find most unpleasant combo")
    sweep.add_argument("--samples", type=int, default=200)
    sweep.add_argument("--top-k", type=int, default=5, dest="top_k")
    sweep.add_argument("--hill-climb-iters", type=int, default=3, dest="hill_climb_iters")
    sweep.add_argument("--duration", type=float, default=2.0)
    sweep.add_argument("--sr", type=int, default=22050)
    sweep.add_argument("--out-dir", type=str, default="./sweep_results", dest="out_dir")
    sweep.add_argument("--workers", type=int, default=min((os.cpu_count() or 1), 4))
    sweep.add_argument("--seed", type=int, default=42)
    sweep.add_argument("--temporal-min-active", type=int, default=0, dest="temporal_min_active")
    sweep.add_argument("--temporal-max-active", type=int, default=3, dest="temporal_max_active")
    sweep.add_argument("--temporal-activation-p", type=float, default=0.45, dest="temporal_activation_p")
    sweep.add_argument("--ab-interval", type=int, default=0, dest="ab_interval")
    sweep.add_argument("--ab-pairs", type=int, default=6, dest="ab_pairs")
    sweep.add_argument("--ab-no-play", action="store_true", dest="ab_no_play")
    sweep.add_argument("--no-save", action="store_true", dest="no_save")
    sweep.set_defaults(func=_cmd_sweep)

    ab_candidates = subparsers.add_parser(
        "ab-candidates",
        help="Generate balanced A/B WAV+sidecar candidates isolating one temporal synth feature at a time",
    )
    ab_candidates.add_argument("--out-dir", type=str, default="./ab_candidates", dest="out_dir")
    ab_candidates.add_argument("--duration", type=float, default=2.0)
    ab_candidates.add_argument("--sr", type=int, default=22050)
    ab_candidates.add_argument("--seed", type=int, default=42)
    ab_candidates.add_argument("--repeats", type=int, default=1)
    ab_candidates.set_defaults(func=_cmd_ab_candidates)

    # AB tool removed

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
