from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

import abtool
import dissonance.cli as cli
from dissonance.ab_candidates import TEMPORAL_CALIBRATION_SPECS, generate_ab_candidates
from sweep import TEMPORAL_GROUPS, _select_ab_probe_indices, sample_one, write_params_sidecar


def _defaults() -> dict[str, float]:
    return {**abtool.UnpleasantnessScorer.DEFAULT_WEIGHTS, **abtool.UnpleasantnessScorer.DEFAULT_SYNTH_WEIGHTS}


def _make_scorer(model: dict[str, object] | None = None) -> abtool.UnpleasantnessScorer:
    return abtool.UnpleasantnessScorer(weights=_defaults(), preference_model=model)


def test_pairwise_logistic_converges_and_stays_bounded() -> None:
    names = list(abtool.FEATURE_NAMES)
    rng = np.random.default_rng(123)
    n_items = 36
    xmat = rng.normal(0.0, 1.0, size=(n_items, len(names)))

    i_rough = names.index("roughness")
    i_synth = names.index("layer_scream_chaos")
    i_synth2 = names.index("shepard_n_voices")
    beta_true = np.zeros(len(names), dtype=np.float64)
    beta_true[i_rough] = 1.2
    beta_true[i_synth] = -0.9
    beta_true[i_synth2] = 0.8

    utilities = xmat @ beta_true
    pairs = []
    while len(pairs) < 48:
        a = int(rng.integers(0, n_items))
        b = int(rng.integers(0, n_items))
        if a == b:
            continue
        if utilities[a] >= utilities[b]:
            pairs.append((a, b, 1.0))
        else:
            pairs.append((b, a, 1.0))

    batches = [pairs[:16], pairs[:32], pairs]
    scorer = _make_scorer()
    last_model: dict[str, object] | None = None
    deltas: list[float] = []
    blends: list[float] = []
    for batch in batches:
        model, diag = abtool._fit_preference_model(xmat=xmat, feature_names=names, comparisons=batch, scorer=scorer)
        assert model is not None
        beta_std = model["beta_std"]
        assert isinstance(beta_std, dict)
        max_abs = max(abs(float(v)) for v in beta_std.values())
        assert max_abs <= 3.0 + 1e-8
        deltas.append(float(diag["max_coef_delta"]))
        blends.append(float(diag["blend"]))
        last_model = model
        scorer = _make_scorer(last_model)

    assert blends[-1] < blends[0]
    assert deltas[-1] < 0.4
    assert last_model is not None
    beta_last = last_model["beta_std"]
    assert float(beta_last["roughness"]) > 0.0
    assert float(beta_last["layer_scream_chaos"]) < 0.0
    assert float(beta_last["shepard_n_voices"]) > 0.0


def test_zero_variance_synth_feature_stays_frozen() -> None:
    names = list(abtool.FEATURE_NAMES)
    xmat = np.zeros((18, len(names)), dtype=np.float64)
    rng = np.random.default_rng(10)

    i_const = names.index("synth_n_layers")
    i_var = names.index("pulse_panic_rate")
    xmat[:, i_const] = 0.42
    xmat[:, i_var] = np.linspace(0.0, 1.0, xmat.shape[0])
    xmat[:, names.index("roughness")] = rng.normal(0.0, 1.0, xmat.shape[0])
    xmat[:, names.index("sharpness")] = rng.normal(0.0, 1.0, xmat.shape[0])

    comparisons: list[tuple[int, int, float]] = []
    utility = xmat[:, i_var] + 0.2 * xmat[:, names.index("roughness")]
    for _ in range(40):
        a = int(rng.integers(0, xmat.shape[0]))
        b = int(rng.integers(0, xmat.shape[0]))
        if a == b:
            continue
        comparisons.append((a, b, 1.0) if utility[a] >= utility[b] else (b, a, 1.0))

    model, _diag = abtool._fit_preference_model(xmat=xmat, feature_names=names, comparisons=comparisons, scorer=_make_scorer())
    assert model is not None
    assert "synth_n_layers" in set(model["frozen_features"])
    assert model["feature_coverage"]["synth_n_layers"]["status"] == "pending_variance"
    assert float(model["beta_std"]["synth_n_layers"]) == 0.0


def test_low_contrast_feature_is_pending_not_silently_updated() -> None:
    names = list(abtool.FEATURE_NAMES)
    rng = np.random.default_rng(202)
    n_items = 28
    xmat = rng.normal(0.0, 1.0, size=(n_items, len(names)))

    idx_sparse = names.index("layer_uncanny_morph")
    xmat[:, idx_sparse] = 0.0
    xmat[:2, idx_sparse] = 1.0

    idx_main = names.index("roughness")
    utility = 1.2 * xmat[:, idx_main] + 0.8 * xmat[:, names.index("sharpness")]

    comparisons: list[tuple[int, int, float]] = []
    while len(comparisons) < 56:
        a = int(rng.integers(0, n_items))
        b = int(rng.integers(0, n_items))
        if a == b:
            continue
        comparisons.append((a, b, 1.0) if utility[a] >= utility[b] else (b, a, 1.0))

    model, diag = abtool._fit_preference_model(xmat=xmat, feature_names=names, comparisons=comparisons, scorer=_make_scorer())
    assert model is not None
    cov = model["feature_coverage"]["layer_uncanny_morph"]
    assert cov["status"] == "pending_contrast"
    assert int(cov["item_support"]) < int(cov["item_support_required"])
    assert not bool(cov["support_ok"])
    assert "layer_uncanny_morph" in set(model["frozen_features"])
    assert int(diag["n_identified_features"]) <= int(diag["n_active_features"])


def test_scorer_blends_model_with_legacy_when_partial_identified() -> None:
    base = _defaults()
    base["roughness"] = 0.3
    base["sharpness"] = 0.2
    model = {
        "feature_names": ["roughness", "layer_scream_chaos", "shepard_n_voices"],
        "identified_features": ["layer_scream_chaos"],
        "means": {"roughness": 0.0, "layer_scream_chaos": 0.0, "shepard_n_voices": 0.0},
        "scales": {"roughness": 1.0, "layer_scream_chaos": 1.0, "shepard_n_voices": 1.0},
        "beta_std": {"roughness": 2.0, "layer_scream_chaos": 3.0, "shepard_n_voices": -2.0},
        "intercept": 0.0,
        "n_directional": 12,
    }
    scorer = abtool.UnpleasantnessScorer(weights=base, preference_model=model)
    score, feats = scorer.score(np.zeros(1600, dtype=np.float32), 16000, layers=[], duration_s=1.0)
    assert 0.0 <= score <= 1.0

    acoustic_only = float(np.clip(base["roughness"] * float(feats["roughness"]) + base["sharpness"] * float(feats["sharpness"]), 0.0, 1.0))
    assert abs(score - acoustic_only) < 0.25


def test_probe_selection_targets_pending_synth_features() -> None:
    class R:
        def __init__(self, score: float, features: dict[str, float]) -> None:
            self.score = score
            self.params = {}
            self.features = features

    synth = list(abtool.UnpleasantnessScorer.SYNTH_FEATURE_NAMES)
    names = list(abtool.FEATURE_NAMES)
    rng = np.random.default_rng(99)
    results = []
    for i in range(24):
        f = {name: 0.0 for name in names}
        f["roughness"] = float(rng.normal(0.0, 1.0))
        f["sharpness"] = float(rng.normal(0.0, 1.0))
        for s_idx, s_name in enumerate(synth):
            f[s_name] = float((i + s_idx) % 4) / 3.0
        results.append(R(score=float(1.0 - i / 30.0), features=f))

    coverage = {name: {"status": "pending_contrast", "contrast_count": 0} for name in synth}
    model = {
        "beta_std": {name: 0.0 for name in names},
        "feature_coverage": coverage,
    }
    scorer = abtool.UnpleasantnessScorer(weights=_defaults(), preference_model=model)
    chosen = _select_ab_probe_indices(results=results, scorer=scorer, rng=np.random.default_rng(3), pick_count=10)
    assert len(chosen) >= 8
    selected_feats = [results[i].features for i in chosen]
    varied_synth = 0
    for s_name in synth:
        vals = {float(f[s_name]) for f in selected_feats}
        if len(vals) >= 2:
            varied_synth += 1
    assert varied_synth == len(synth)


def test_sweep_probe_sidecar_enables_temporal_feature_observation(tmp_path: Path) -> None:
    wav = tmp_path / "probe.wav"
    sf.write(wav, np.zeros(2205, dtype=np.float32), 22050)

    params = {
        "rough": {
            "carrier_hz": 3000,
            "n_partials": 8,
            "partial_spread_bark": 0.25,
            "am_rate_hz": 70,
            "am_depth": 0.9,
        },
        "stickslip": {"ioi_mean_ms": 4.0, "ioi_jitter": 0.5, "resonance_hz": [2400, 3300, 4100]},
        "fm_instab": {"carrier_hz": 3000, "mod_rate_hz": 12, "mod_index": 8, "mod_chaos": 0.7},
        "inharmonic": {"base_hz": 200, "n_partials": 12, "inharmonicity_B": 0.12, "random_detune": 0.5},
        "beating": {"base_hz": 220, "n_beaters": 3, "beat_rate_hz": 7, "beat_jitter": 0.3},
        "noise_shaped": {"center_hz": 3150, "bandwidth_hz": 2000, "modulation_rate_hz": 70, "modulation_depth": 0.6},
        "global": {"hump_2_4khz_db": 9, "highpass_hz": 800},
        "scream_chaos": {
            "carrier_hz": 700,
            "subharmonic_gain": 0.4,
            "chaos_amount": 0.6,
            "pitch_jump_rate_hz": 0.8,
            "biphonation_ratio": 1.67,
            "biphonation_gain": 0.2,
        },
        "uncanny_morph": {
            "base_hz": 200,
            "n_partials": 12,
            "inharmonicity_start": 0.02,
            "inharmonicity_end": 0.3,
            "formant_sweep": True,
        },
    }
    sidecar = write_params_sidecar(params=params, out_wav=wav, sr=22050, duration_s=0.1)
    assert sidecar.exists()

    score, feats = abtool._analyze_wav_features(wav, _make_scorer())
    assert isinstance(score, float)
    assert feats["layer_scream_chaos"] == 1.0
    assert feats["layer_uncanny_morph"] == 1.0
    assert feats["layer_dread_swell"] == 0.0


def test_run_ab_session_accumulates_and_skips_until_enough_evidence(tmp_path: Path) -> None:
    class DummyScorer:
        def __init__(self) -> None:
            self.weights = _defaults()
            self.preference_model = None

        def score(self, signal, sr, layers=None, duration_s=5.0):
            x = np.asarray(signal, dtype=np.float64)
            m = float(np.mean(np.abs(x)))
            return m, {
                "roughness": m,
                "sharpness": m * m,
                "dissonance": m * 0.5,
                "crest_factor": 0.0,
                "band_energy_2_4khz": 0.0,
                "am_energy_70hz": 0.0,
            }

    def _mk_wav(path: Path, level: float) -> Path:
        sig = np.full(800, level, dtype=np.float32)
        sf.write(path, sig, 16000)
        return path

    scorer = DummyScorer()
    batch1 = [_mk_wav(tmp_path / "a.wav", 0.1), _mk_wav(tmp_path / "b.wav", 0.1)]
    payload1, rows1, items1 = abtool.run_ab_session(
        wavs=batch1,
        scorer=scorer,
        n_pairs=1,
        strategy="smart",
        no_play=True,
        seed=1,
    )
    assert payload1 is None
    assert rows1
    assert len(items1) == 2

    batch2 = [
        _mk_wav(tmp_path / "c.wav", 0.15),
        _mk_wav(tmp_path / "d.wav", 0.25),
        _mk_wav(tmp_path / "e.wav", 0.35),
        _mk_wav(tmp_path / "f.wav", 0.45),
        _mk_wav(tmp_path / "g.wav", 0.55),
    ]
    payload2, rows2, items2 = abtool.run_ab_session(
        wavs=batch2,
        scorer=scorer,
        n_pairs=10,
        strategy="smart",
        no_play=True,
        seed=2,
        accumulated_rows=rows1,
        accumulated_items=items1,
    )
    assert len(rows2) > len(rows1)
    assert len(items2) == len(items1) + len(batch2)
    assert payload2 is not None
    assert isinstance(payload2.get("preference_model"), dict)


def test_main_analysis_path_loads_params_sidecar(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    sidecar_path = tmp_path / "sample.params.json"
    sf.write(wav_path, np.zeros(1600, dtype=np.float32), 16000)
    sidecar_path.write_text(
        """
        {
          "duration_s": 1.23,
          "layers": [{"type": "scream_chaos", "biphonation_ratio": 0.8}]
        }
        """.strip(),
        encoding="utf-8",
    )

    class CaptureScorer:
        def __init__(self) -> None:
            self.layers = None
            self.duration_s = None

        def score(self, signal, sr, layers=None, duration_s=5.0):
            self.layers = layers
            self.duration_s = duration_s
            return 0.0, {}

    scorer = CaptureScorer()
    features_by_idx, _ = abtool._analyze_wavs_batch([wav_path], scorer, list(abtool.FEATURE_NAMES))

    assert isinstance(scorer.layers, list)
    assert scorer.layers and scorer.layers[0]["type"] == "scream_chaos"
    assert abs(float(scorer.duration_s) - 1.23) < 1e-9
    assert features_by_idx[0]["layer_scream_chaos"] == 1.0


def test_ab_candidate_generator_sidecars_cover_presence_and_param_ranges(tmp_path: Path) -> None:
    out_dir = tmp_path / "ab_candidates"
    wavs = generate_ab_candidates(out_dir=out_dir, duration_s=0.02, sample_rate=22050, seed=7, repeats=1)

    assert wavs
    assert len(wavs) == 2 * len(TEMPORAL_CALIBRATION_SPECS) * 3

    layer_present: dict[str, set[int]] = {spec.layer_type: set() for spec in TEMPORAL_CALIBRATION_SPECS}
    param_values_seen: dict[str, set[float]] = {spec.layer_type: set() for spec in TEMPORAL_CALIBRATION_SPECS}

    for wav in wavs:
        sidecar = wav.with_name(f"{wav.stem}.params.json")
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        layers = payload.get("layers", [])
        assert isinstance(layers, list)
        by_type = {
            str(layer.get("type")): layer
            for layer in layers
            if isinstance(layer, dict) and isinstance(layer.get("type"), str)
        }

        for spec in TEMPORAL_CALIBRATION_SPECS:
            present = 1 if spec.layer_type in by_type else 0
            layer_present[spec.layer_type].add(present)
            if present:
                param_values_seen[spec.layer_type].add(float(by_type[spec.layer_type][spec.param_key]))

    for spec in TEMPORAL_CALIBRATION_SPECS:
        assert layer_present[spec.layer_type] == {0, 1}
        assert len(param_values_seen[spec.layer_type]) >= 3


def test_sweep_sampling_varies_active_temporal_layers() -> None:
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(17)
    draws = [
        sample_one(rng_a, temporal_min_active=0, temporal_max_active=3, temporal_activation_p=0.45),
        sample_one(rng_a, temporal_min_active=0, temporal_max_active=3, temporal_activation_p=0.45),
        sample_one(rng_b, temporal_min_active=0, temporal_max_active=3, temporal_activation_p=0.45),
        sample_one(rng_b, temporal_min_active=0, temporal_max_active=3, temporal_activation_p=0.45),
    ]

    active_sets = []
    for params in draws:
        active = {name for name in TEMPORAL_GROUPS if isinstance(params.get(name), dict)}
        assert len(active) <= 3
        active_sets.append(frozenset(active))

    has_missing = any(len(s) < len(TEMPORAL_GROUPS) for s in active_sets)
    has_some_active = any(len(s) > 0 for s in active_sets)
    assert has_missing
    assert has_some_active


def test_cli_sweep_forwards_ab_calibration_flags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd):
        captured["cmd"] = cmd

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    args = cli.argparse.Namespace(
        samples=7,
        top_k=2,
        hill_climb_iters=4,
        duration=1.5,
        sr=16000,
        out_dir="./out",
        workers=1,
        seed=9,
        temporal_min_active=1,
        temporal_max_active=2,
        temporal_activation_p=0.2,
        ab_interval=11,
        ab_pairs=3,
        ab_no_play=True,
        no_save=True,
    )

    try:
        cli._cmd_sweep(args)
    except SystemExit as exc:
        assert exc.code == 0

    cmd = captured["cmd"]
    assert "--ab-interval" in cmd and "11" in cmd
    assert "--ab-pairs" in cmd and "3" in cmd
    assert "--ab-no-play" in cmd
