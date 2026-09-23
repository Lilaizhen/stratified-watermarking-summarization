# Watermarking for Summarization — reproducible Colab

Laizhen Li · Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

This package accompanies the final analysis report. It replaces the historical
Colab protocol; old notebooks and archives elsewhere in the repository are not
inputs to this package.

## Open in Colab

Open the published notebook link and run its first cell to download the pinned
repository revision automatically. Run the installation cell in a fresh runtime;
restart and rerun from the top if requested. Saved results use CPU; new generation
requires CUDA. Hosted Colab and fresh installation remain unverified.

## Final protocol

BART-large-CNN at temperature 0.8; full-vocabulary multinomial sampling; bias
**after temperature scaling**; pure Stratified Watermarking without extra
special-token protection. Model revisions, data file hash and article IDs are
fixed. Generation batch size is 8 with seed 1729 + batch starting offset.

- `demo`: Unwatermarked, Soft δ=3, Stratified ρ=.98/δ=4, and Hard; a subset of
  evaluation articles. It is a functional demonstration, not the full report.
- `main`: all 500 evaluation articles, 200 calibration articles, 32 configurations:
  Unwatermarked, Hard, 15 Soft and 15 Stratified (ρ=.98).
- `full`: 66 configurations, also covering γ=.1/.25/.5/.75/.9 at δ=2.5 and
  ρ=.90/.95/.98 on the 15-bias grid. This generates 33,000 evaluation summaries
  plus 200 calibration summaries. Expect multiple sessions on limited Colab.
- Probability-mass analysis is a separate optional pass over saved Unwatermarked
  prefixes. Report reconstruction uses 500 articles and 35,795 positions.

Detection uses CUDA color reconstruction, distinct adjacent pairs, and strict
`z > threshold`. Thresholds come only from the independent calibration set.
Unscorable summaries count as non-detections and rank at negative infinity for
ROC calculations. Actual evaluation FPR is measured, not assumed to equal 1%.
Small demos have noisy detection estimates even with the full calibration set.

ROUGE-1/2/L F1, unconditional GPT-2-medium corpus PPL, and AlignScore-base nli_sp
are supported. `QUALITY=False` explicitly skips PPL/AlignScore. No BERTScore is
installed. ROUGE is on 0–100; AlignScore on 0–1; TPR/FPR in CSVs are fractions.
PPL is exp(total negative log-likelihood / total scored tokens), not the mean of
per-summary perplexities. AlignScore uses the documented local inference port.
Empty-summary AlignScore is missing and its count is reported.

## Local execution

Run from the extracted package root:

```bash
python -m pip install -r requirements.txt
python tests/check_methods.py
python src/plots.py
python src/setup_assets.py --quality
python -c 'import sys; sys.path.insert(0,"src"); from workflow import run; run(quality=True)'
```

Each model runs in a separate subprocess. GPU memory is released between stages.
An interrupted generation resumes at the last complete batch, discarding only
an incomplete final batch. Evaluation stages check generation hashes before
reusing outputs. Keep settings unchanged to resume; use another output path to
change sample counts or configuration. Persist long-run output folders to Drive.
Do not change GPU/library/batch-size settings mid-run if exact RNG equivalence
matters. Different hardware may yield different summaries even with fixed seeds.

## Files and provenance

- `Watermark_Analysis.ipynb`: readable workflow, algorithm explanation and source.
- `src/`: generation, independent calibration, evaluation, plots, diagnostic.
- `configs/`: immutable experiment settings and disjoint sample IDs.
- `archived/`: per-article scores for 66 final configurations, calibration scores,
  and probability-mass summary; no source articles or model weights.
- `report.pdf`: current analysis report.
- `tests/check_methods.py`: regional probability conservation and detector math.
- `VERIFICATION.json`: performed checks and unverified environments.
- `MANIFEST.json`: hashes of packaged inputs; portable AlignScore settings are
  hashed without machine-local paths added during setup.

Algorithms reuse Transformers WatermarkLogitsProcessor and the local, verified
regional logit correction. AlignScore source attribution/hashes are retained in
`configs/focused/alignscore.json`. Libraries and checkpoints retain their original
licenses and terms. Models, news data and NLTK resources download separately.

Complete curves include all measured grid points, not continuous interpolation.
The report selects minimum-PPL configurations meeting each target TPR (ties by
smaller bias), so actual TPRs differ. The findings remain descriptive single-seed
results, not evidence of attack robustness or universal superiority.
