import os, json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Paths & device
ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")

# Inference transform (match your validation/train if you used Normalize)
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # If you trained with normalization, uncomment this:
    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def build_model(num_classes=2):
    # IMPORTANT: Do NOT request pretrained weights from the internet.
    # We load our fine-tuned weights from artifacts/ below.
    model = models.resnet18(weights=None)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ----- Load artifacts -----
classes_path = os.path.join(ROOT, "artifacts", "classes.json")
weights_path = os.path.join(ROOT, "artifacts", "best_model_dogcat.pt")

if not os.path.exists(classes_path) or not os.path.exists(weights_path):
    raise FileNotFoundError(
        "Missing artifacts. Please upload both files into artifacts/:\n"
        f" - {classes_path}\n"
        f" - {weights_path}"
    )

with open(classes_path, "r") as f:
    CLASSES = json.load(f)

MODEL = build_model(num_classes=len(CLASSES))
state = torch.load(weights_path, map_location="cpu")
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

# Optional examples (add your own images to examples/ in the Space)
def build_examples():
    examples = []
    for name in ["cat1.jpg", "dog1.jpg"]:
        p = os.path.join(ROOT, "examples", name)
        if os.path.exists(p):
            examples.append([p])
    return examples

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a cat or dog"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(num_top_classes=2, label="Class probabilities"),
    ],
    title="Cat vs Dog Classifier",
    flagging_mode="never",
    examples=build_examples(),
)

# Hugging Face Spaces looks for `app`
app = demo

if __name__ == "__main__":
    demo.launch()