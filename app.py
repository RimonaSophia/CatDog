import os, json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr


ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")

# MUST match your validation/inference preprocessing.
# If you trained with ImageNet normalization, uncomment the Normalize line below.
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),  # <-- uncomment if used in training
])

def build_model(num_classes=2):
    # Load a ResNet18 backbone with ImageNet weights and replace the final layer.
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)  # older torchvision fallback
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Load classes and weights
CLASSES_PATH = os.path.join(ROOT, "artifacts", "classes.json")
WEIGHTS_PATH = os.path.join(ROOT, "artifacts", "best_model_catdog.pt")

if not os.path.exists(CLASSES_PATH) or not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        "Missing artifacts. Expected files:
"
        f" - {CLASSES_PATH}
"
        f" - {WEIGHTS_PATH}
"
        "Place your model and classes.json in the artifacts/ folder."
    )

with open(CLASSES_PATH) as f:
    CLASSES = json.load(f)

MODEL = build_model(num_classes=len(CLASSES))
state = torch.load(WEIGHTS_PATH, map_location="cpu")
MODEL.load_state_dict(state)
MODEL.eval().to(DEVICE)

@torch.inference_mode()
def predict(img: Image.Image):
    x = TF(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = MODEL(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    label = CLASSES[idx]
    return f"{label} ({probs[idx]:.3f})", {CLASSES[i]: float(p) for i, p in enumerate(probs)}

# Build clickable example list only for files that actually exist
def existing_examples():
    candidates = [
        os.path.join(ROOT, "examples", "cat1.jpg"),
        os.path.join(ROOT, "examples", "dog1.jpg"),
    ]
    return [[p] for p in candidates if os.path.exists(p)]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a cat or dog"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(num_top_classes=2, label="Class probabilities")
    ],
    title="Cat vs Dog Classifier",
    flagging_mode="never",
    examples=existing_examples()
)

# Spaces will look for this variable
app = demo

if __name__ == "__main__":
    demo.launch()