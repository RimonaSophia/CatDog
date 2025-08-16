# Cat vs Dog — Gradio (Spaces-ready)

Upload the following into your Hugging Face Space (SDK: Gradio, Public, CPU Basic):

- `app.py`
- `requirements.txt`
- `runtime.txt`
- `artifacts/best_model_dogcat.pt`
- `artifacts/classes.json`
- optional: `examples/cat1.jpg`, `examples/dog1.jpg`

**Note:** `app.py` does **not** download pretrained weights; it loads your fine-tuned weights from `artifacts/`.
If you trained with normalization, uncomment the `Normalize(...)` line in `TF`.