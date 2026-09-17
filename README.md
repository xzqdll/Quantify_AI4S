# LLM Impact on AI Publications

Reproducible code and derived data for analysing detector-positive textual
traces consistent with LLM-assisted polishing and stylistic convergence in
selected AI-related computer-science conference abstracts.

## Repository structure

- `configs/`: analysis and model configuration files.
- `data/`: data documentation and local data layers; large or restricted data
  are not tracked by Git.
- `docs/`: reproducibility, data-governance, and contributor documentation.
- `models/`: model documentation and links to archived weights.
- `notebooks/`: numbered notebooks for exploration and result reproduction.
- `prompts/`: versioned prompt templates used in benchmark generation and topic
  interpretation.
- `results/`: generated figures, tables, and run logs.
- `scripts/`: command-line entry points for end-to-end workflows.
- `src/llm_impact_on_ai_pub/`: reusable Python package.
- `tests/`: automated tests and small synthetic fixtures.

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

