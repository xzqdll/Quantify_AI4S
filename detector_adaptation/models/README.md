# Model checkpoints

`CS_Task3.pth` is the pretrained computer-science CheckGPT classification head
used as the starting point. `APT-Det_full_head.pth` is the selected full-head
adaptation used in the article. Both checkpoints contain the CheckGPT
classification head only; `roberta-large` is loaded separately from
Hugging Face Transformers.

| File | Purpose | SHA-256 |
|---|---|---|
| `CS_Task3.pth` | Base CheckGPT computer-science head | `a4ea7fe39ebe99c8d5bf6860937390e012fd5935fc29df2894f2e49e07253653` |
| `APT-Det_full_head.pth` | Adapted head used for corpus inference | `894e2e561a8fe2426aeafb66f451dd6adb0f108781172e112473fc83fd29d460` |

The decision threshold used in the main analysis is 0.5. The accompanying
threshold-calibration JSON is retained for sensitivity auditing but is not used
to replace the default threshold in the main analysis.
