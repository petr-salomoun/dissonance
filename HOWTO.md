# HOWTO — using dissonance

## Installation

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.11, numpy, scipy, soundfile, librosa. Optional: sounddevice (for audio playback during A/B sessions).

---

## CLI commands

### Generate a sound

Generate a single unpleasant sound using the default preset:

```bash
dissonance gen --out my_sound.wav
```

Options:
- `--out PATH` — output WAV path (default: `out.wav`)
- `--preset PATH` — load parameters from a JSON preset file
- `--duration SECS` — duration in seconds (default: 5.0)
- `--sr HZ` — sample rate (default: 44100)

### Score a file

Score the unpleasantness of any WAV file:

```bash
dissonance score --in my_sound.wav
```

Prints a score between 0 and 1, plus the individual feature values.

### Full analysis

Print a detailed breakdown of all acoustic features:

```bash
dissonance analyze --in my_sound.wav
```

### Run the optimizer

Search the parameter space for the most unpleasant combination:

```bash
dissonance sweep --samples 200 --top-k 5 --hill-climb-iters 5 --out-dir ./results
```

Options:
- `--samples N` — number of random candidates to evaluate in phase 1 (default: 100)
- `--top-k K` — how many top candidates to hill-climb from in phase 2 (default: 5)
- `--hill-climb-iters N` — hill-climb iterations per seed (default: 5)
- `--out-dir PATH` — where to save results (WAVs + JSON presets)

Each result is saved as a `.wav` + `.json` pair so any result can be regenerated exactly:

```bash
dissonance gen --preset results/best.json --out repro.wav
```

---

## A/B calibration

The scoring weights can be calibrated from your own listening preferences.

### Standalone A/B session

Run a pairwise listening test directly:

```bash
dissonance sweep --samples 200 --out-dir ./results
# then:
python abtool.py --dir results --pairs 10
```

You will hear each pair played back-to-back (A then B) and be prompted to pick which one sounds worse. After the session, calibrated weights are printed and can be passed back to the scorer.

### Mid-sweep calibration

The sweep command can pause every N samples, run a mini listening session on the current top candidates, update the scoring weights, and continue. This lets the optimizer adapt to your preferences as it runs:

```bash
dissonance sweep --samples 200 --ab-interval 20 --ab-pairs 6 --out-dir ./results
```

Options:
- `--ab-interval N` — pause every N samples for a calibration round (0 = disabled)
- `--ab-pairs N` — number of pairwise comparisons per calibration round
- `--ab-no-play` — run calibration without audio playback (for testing/automation)

### How the calibration works

1. Your pairwise verdicts are fit to a **Bradley-Terry model**, assigning a latent unpleasantness strength to each candidate.
2. Those strengths are regressed onto the acoustic feature matrix using **ridge regularization** (prevents weight collapse to a single feature).
3. The result is **blended with the default weights** (prior blend), so one noisy session doesn't destabilize everything.
4. All comparisons **accumulate** across rounds, so weights converge over time.

---

## Project layout

```
dissonance/
  core/
    synth/
      rough.py          roughness-oriented partial cluster
      stickslip.py      jittered impulse / screech generator
      fm_instab.py      chaotic FM instability tone
      inharmonic.py     stretched, detuned partial stack
      beating.py        close-tone beating synthesizer
      noise_shaped.py   critical-band harsh noise
    mixer.py            combines layers, applies global EQ
    dsp/                low-level DSP (filters, envelopes, Bark scale)
  io/
    render.py           params dict → WAV file
    presets.py          built-in parameter sets
  analysis/
    features.py         acoustic feature extraction
    scorer.py           weighted unpleasantness scoring
  cli.py                gen / score / analyze / sweep commands
sweep.py                optimizer (phase 1 random + phase 2 hill-climb + A/B hooks)
abtool.py               pairwise calibration tool
```

---

## Preset JSON format

Presets are plain JSON files with a `layers` list and an optional `global` block:

```json
{
  "duration_s": 5.0,
  "sample_rate": 44100,
  "global": {
    "hump_2_4khz_db": 12,
    "highpass_hz": 1200
  },
  "layers": [
    {
      "type": "rough",
      "carrier_hz": 3000,
      "n_partials": 4,
      "partial_spread_bark": 0.1,
      "am_rate_hz": 40,
      "am_depth": 0.5,
      "gain_db": -6.0
    },
    {
      "type": "stickslip",
      "ioi_mean_ms": 6.0,
      "ioi_jitter": 0.3,
      "resonance_hz": [2400, 3300, 4100],
      "gain_db": -3.0
    }
  ]
}
```

Layer types: `rough`, `stickslip`, `fm_instab`, `inharmonic`, `beating`, `noise_shaped`.

The `_sweep_meta` key (added by the optimizer) is ignored by the generator and can be left in or stripped.
