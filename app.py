import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# ---------- Device ----------
DEVICE = torch.device("cpu")  # Spaces CPU is fine; change if you enable GPU

# ---------- Transforms ----------
# IMPORTANT: match whatever you used at validation/inference time during training.
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # If you trained with ImageNet normalization, uncomment:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------- Model ----------
def build_model(num_classes=2):
    # Use weights API when available; fallback to pretrained=True for older torchvision
    try:
        weights = models.ResNet18_Weights.DEFAULT
        m = models.resnet18(weights=weights)
    except Exception:
        m = models.resnet18(pretrained=True)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# Load classes
with open("artifacts/classes.json") as f:
    CLASSES = json.load(f)  # e.g. ["cats", "dogs"]

# Build + load weights
model = build_model(num_classes=len(CLASSES))
state = torch.load("artifacts/best_model_dogcat.pt", map_location="cpu")  # <- ensure filename matches
model.load_state_dict(state, strict=True)
model.to(DEVICE).eval()

# Optional: warmup to surface any runtime errors early (e.g., missing files)
with torch.inference_mode():
    _ = model(torch.zeros(1, 3, 224, 224))

# ---------- Prediction ----------
@torch.inference_mode()
def predict(img: Image.Image):
    try:
        x = TF(img.convert("RGB")).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
        logits = model(x)  # <-- fixed: use `model`, not `MODEL`
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        idx = int(torch.tensor(probs).argmax().item())
        label = CLASSES[idx]
        # Gradio Label expects {class_name: prob}
        prob_dict = {CLASSES[i]: float(p) for i, p in enumerate(probs)}
        return f"{label} ({probs[idx]:.3f})", prob_dict
    except Exception as e:
        # Return readable error in the UI instead of crashing
        return f"Error: {type(e).__name__}: {e}", {}

# --- built-in demo images users can click ---
EXAMPLES = [
    "examples/2.jpg",
    "examples/10.jpg",
    "examples/12.jpg",
    "examples/20.jpg",
    "examples/21.jpg",
]


# ---------- Gradio UI ----------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a cat or dog"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(num_top_classes=len(CLASSES), label="Class probabilities"),
    ],
    title="Cat vs Dog Classifier",
    examples=EXAMPLES,  
    flagging_mode="never",
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
