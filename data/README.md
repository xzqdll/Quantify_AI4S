# Data directory

The repository separates data by processing stage:

- `raw/`: locally acquired source records; never committed.
- `external/`: third-party inputs; never committed unless redistribution is
  explicitly permitted.
- `interim/`: intermediate matching and feature-extraction outputs.
- `processed/`: de-identified analysis-ready data.
- `source_data/`: numerical values underlying manuscript figures and tables.

Before public release, document every file's provenance, licence, schema,
checksum, and reconstruction procedure. Do not commit API credentials, author
names linked to detector outputs, or abstract text whose redistribution is not
permitted.

