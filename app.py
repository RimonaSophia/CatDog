# app.py
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

DEVICE = torch.device("cpu")  # simple and portable

# must match your val transforms
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def build_model(num_classes=2):
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# load once
with open("artifacts/classes.json") as f:
    CLASSES = json.load(f)

MODEL = build_model(num_classes=len(CLASSES))
state = torch.load("artifacts/best_model_catdog.pt", map_location="cpu")
MODEL.load_state_dict(state)
MODEL.eval().to(DEVICE)

@torch.inference_mode()
def predict(img: Image.Image):
    x = TF(img.convert("RGB")).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
    logits = MODEL(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    label = CLASSES[idx]
    # return a friendly top label and a dict for the Label widget
    return f"{label} ({probs[idx]:.3f})", {CLASSES[i]: float(p) for i, p in enumerate(probs)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a cat or dog"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(num_top_classes=len(CLASSES), label="Class probabilities")
    ],
    title="Cat vs Dog Classifier",
    allow_flagging="never",
    examples=None  # you can add sample image paths here later
)

if __name__ == "__main__":
    # share=True gives you a temporary public link
    demo.launch(share=False)
