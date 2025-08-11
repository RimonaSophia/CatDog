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

model = build_model(num_classes=len(CLASSES))
state = torch.load("artifacts/best_model_dogcat.pt", map_location="cpu")
model.load_state_dict(state)
model.eval().to(DEVICE)

@torch.inference_mode()
def predict(img: Image.Image):
    x = TF(img.convert("RGB")).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
    logits = MODEL(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    label = CLASSES[idx]
    return f"{label} ({probs[idx]:.3f})", {CLASSES[i]: float(p) for i, p in enumerate(probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a cat or dog"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(num_top_classes=len(CLASSES), label="Class probabilities")
    ],
    title="Cat vs Dog Classifier",
    flagging_mode="never"          # was allow_flagging
   
)

if __name__ == "__main__":
    demo.launch(share=True)