# LLM Impact on AI Publications

Reproducible code and derived data for analysing detector-positive textual
traces consistent with LLM-assisted polishing and stylistic convergence in
selected AI-related computer-science conference abstracts.

## Repository structure

- `configs/`: analysis and model configuration files.
- `data/`: data documentation and local data layers; large or restricted data
  are not tracked by Git.
- `code_get_paper_data/`: pipeline of collecting data.
- `code_analysis/`: measurement and plotting figures.
- `detector_adaptation/`

## Reproducibility

The public release should contain scripts for data acquisition and matching,
APT-Det adaptation and evaluation, fixed-effects and DiD-style analyses,
semantic-topic analysis, stylometric analysis, robustness checks, and figure
generation. Exact software versions, random seeds, and execution instructions
will be documented before archival release.

## Data access

Original abstract text is not committed to this repository. See
`data/README.md` for the planned release of de-identified derived data, source
data, and instructions for reconstructing records from their original sources
subject to provider terms.

## Licence and citation

The project licence and archival citation will be added before public release.

