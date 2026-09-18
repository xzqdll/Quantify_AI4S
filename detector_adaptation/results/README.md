# Detector adaptation results

`comparison/` contains the aggregate comparison of direct CheckGPT,
last-layer adaptation, and full-head adaptation. `selected_full_head_run/`
contains the metrics, validation history and threshold-sensitivity outputs for
run 02, whose checkpoint is distributed as `models/APT-Det_full_head.pth`.

Metrics use the held-out test split. Threshold calibration and sensitivity
files are supplied for auditing; the article's primary classification uses the
default threshold of 0.5.
