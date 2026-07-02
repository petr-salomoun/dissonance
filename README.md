# dissonance — engineering the most unpleasant sound possible

Psychoacoustics has well-studied principles for what makes sound grate on us. This project turns those principles into synthesis algorithms, then runs an optimizer to find the combination that scores highest on a weighted unpleasantness metric. It can also calibrate the scoring weights from your own A/B listening judgments.

---

## What makes sound unpleasant?

The literature points to a handful of perceptual properties — roughness, sharpness, inharmonicity, tonal instability. Each is a distinct mechanism. We implement each as its own synthesis layer so we can study and combine them independently.

---

### Roughness

When two frequency components sit close together inside a single *critical band* — the auditory system's native spectral resolution unit — they beat against each other at 20–200 Hz, producing the sensation psychoacousticians call **roughness**. It is the sound of two slightly detuned strings, scaled up to dozens of partials.

<audio controls src="examples/01_roughness.wav"></audio><br>
▶ [examples/01_roughness.wav](examples/01_roughness.wav?raw=true) — unpleasantness score: **0.99**

---

### Sharpness

Energy concentrated above ~3 kHz feels *piercing* in a way that goes beyond loudness. The German psychoacoustician Zwicker formalized this as **sharpness** (Schärfe), weighted toward the top of the audible range. Think dentist drill, not thunder.

<audio controls src="examples/02_sharpness.wav"></audio><br>
▶ [examples/02_sharpness.wav](examples/02_sharpness.wav?raw=true) — unpleasantness score: **0.52**

---

### FM instability

A stable tone is easy for the auditory system to track and suppress. Chaotic pitch fluctuation — where the modulator itself drifts randomly — prevents that adaptation. The brain keeps reaching for a stable percept and finding none.

<audio controls src="examples/03_fm_instability.wav"></audio><br>
▶ [examples/03_fm_instability.wav](examples/03_fm_instability.wav?raw=true) — unpleasantness score: **0.33**

---

### Stick-slip / screech

The chalk-on-blackboard family. Irregular micro-bursts create stochastic amplitude modulation in the roughness band, but with random timing that *blocks habituation* — the ear never gets to stop noticing it. This mechanism is at work in fingernails on glass, squealing brakes, and alarmed primates.

<audio controls src="examples/04_stickslip.wav"></audio><br>
▶ [examples/04_stickslip.wav](examples/04_stickslip.wav?raw=true) — unpleasantness score: **0.90**

---

### Inharmonic partials

Natural sounds — voices, strings, wind instruments — have overtones at integer multiples of the fundamental. Stretch or compress those ratios and the result sounds broken: metallic but not in a good way, like a bell hit by another bell that disagrees with it. Inharmonicity also generates roughness, because non-integer-ratio partials inevitably land inside the same critical band as other partials at odd intervals.

<audio controls src="examples/05_inharmonic.wav"></audio><br>
▶ [examples/05_inharmonic.wav](examples/05_inharmonic.wav?raw=true) — unpleasantness score: **0.31**

---

### Beating tones

Two pure tones a few Hz apart create a slow amplitude oscillation — **beats** — as they cycle in and out of phase. One or two beats can sound like vibrato. Five simultaneous beaters at slightly different beat rates creates something woozy, unmoored, mildly nauseating.

<audio controls src="examples/06_beating.wav"></audio><br>
▶ [examples/06_beating.wav](examples/06_beating.wav?raw=true) — unpleasantness score: **0.17**

---

### Shaped critical-band noise

Broadband noise becomes nastier when focused on the most sensitive hearing range, roughly 2–6 kHz (the frequency range of speech consonants and, evolutionarily, predator/infant cries). Concentrating noise there simultaneously maximizes roughness and sharpness. Adding 70 Hz AM puts temporal flutter on top.

<audio controls src="examples/07_noise_shaped.wav"></audio><br>
▶ [examples/07_noise_shaped.wav](examples/07_noise_shaped.wav?raw=true) — unpleasantness score: **0.98**

---

## Optimizing for maximum unpleasantness

Each synthesis method has knobs: carrier frequency, number of partials, AM rate, FM chaos depth, and so on. The optimizer searches this joint space to find the combination that scores highest.

**The unpleasantness score** is a weighted sum of normalized acoustic features:

| Feature | What it captures |
|---|---|
| roughness | fast AM beating within critical bands |
| sharpness | high-frequency energy concentration |
| dissonance | spectral dissonance between partial pairs |
| crest factor | transient spikiness / impulsiveness |
| band energy 2–4 kHz | energy in the most aversive frequency region |
| AM energy at 70 Hz | strength of roughness-rate modulation |
| roughness × sharpness | interaction (rough *and* sharp is worse than either alone) |

The search runs in two phases. First, a wide random sample across the full parameter space to find promising regions. Then a local hill-climb from the top candidates, nudging each parameter and keeping mutations that improve the score.

---

## Best results

The optimizer consistently lands near a score of **0.855** when all layers are combined. Here are the top results from the actual optimization run:

**#1 — score 0.855**

<audio controls src="examples/results_best.wav"></audio><br>
▶ [examples/results_best.wav](examples/results_best.wav?raw=true)

**#2 — score 0.855**

<audio controls src="examples/results_top2.wav"></audio><br>
▶ [examples/results_top2.wav](examples/results_top2.wav?raw=true)

**#3 — score 0.853**

<audio controls src="examples/results_top3.wav"></audio><br>
▶ [examples/results_top3.wav](examples/results_top3.wav?raw=true)

**#4 — score 0.853**

<audio controls src="examples/results_top4.wav"></audio><br>
▶ [examples/results_top4.wav](examples/results_top4.wav?raw=true)

**#5 — score 0.852**

<audio controls src="examples/results_top5.wav"></audio><br>
▶ [examples/results_top5.wav](examples/results_top5.wav?raw=true)

What each layer contributes in the winning presets:
- **rough** — anchors the mix in a dense 3 kHz cluster with fast AM pulse
- **stickslip** — irregular transient grit that refuses to let the ear habituate
- **fm_instab** — continuous pitch wobble so the tone never settles
- **inharmonic** — metallic partials that don't line up with anything harmonic
- **beating** — slow queasy oscillation underneath the brighter layers
- **noise_shaped** — fills 2–6 kHz with harsh, modulated noise

---

## A/B calibration

The scoring weights reflect one reasonable guess at what humans find unpleasant. Yours may differ.

The toolkit includes a pairwise listening test mode: two candidates are played back-to-back and you indicate which sounds worse. Your judgments are fit to a Bradley-Terry ranking model, and the resulting latent scores are regressed onto the acoustic features to yield updated weights. Ridge regularization and prior blending keep the weights stable across sessions, and all historical comparisons accumulate so the weights converge over time rather than jumping per-batch.

The optimizer can pause mid-sweep every N samples, run a mini listening session on the current top candidates, update the weights, and continue — so the search adapts to your preferences as it runs.

---

## Usage and installation

See [HOWTO.md](HOWTO.md) for installation, CLI usage, and how to run the A/B calibration.
