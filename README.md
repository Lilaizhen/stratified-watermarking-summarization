# Watermarking for Summarization

Laizhen Li — Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

Apply and analyze soft and hard text watermarking on BART-large-CNN / CNN-DailyMail,
and investigate Stratified Watermarking, which preserves head and tail probability
mass separately while favoring green tokens within each region.

## Run

The Colab URL will be added after publication. `Watermark_Analysis.ipynb` is the
entry point; its bootstrap downloads a pinned code revision automatically.
No ZIP upload is required for the GitHub version.

- Saved analysis: CPU reconstruction from archived per-article records.
- Demo: small GPU experiment with independent calibration.
- Main/full: 32/66 configurations on 500 evaluation articles and 200 calibration articles.

The protocol fixes temperature 0.8, no top-k/top-p truncation, and temperature
before bias. Stratified uses no extra special-token probability protection.
The report is [report.pdf](report.pdf). Detailed setup and metric definitions are
in [USAGE.md](USAGE.md); verification scope is in [VERIFICATION.json](VERIFICATION.json).

## Local quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/check_methods.py
python src/plots.py
```

For new generation, run `python src/setup_assets.py --quality`, then follow the
notebook. Do not commit downloaded data, checkpoints, or generated news text.

## Evidence and scope

Local cached-model tests reproduce all token sequences and metric values for
four configurations on eight articles. The independent 200-article calibration
reproduces the original scores for five green-list fractions. Hosted Colab and
fresh online installation are not yet verified. Full archived results are not
claimed to have been regenerated with this package.

Original algorithm: [Kirchenbauer et al. (2023)](https://arxiv.org/abs/2301.10226).
AlignScore port provenance is retained in `configs/focused/alignscore.json`.
Third-party libraries, data and checkpoints remain subject to their upstream terms.
