# export.py — minimal export for Hugging Face
import os
import json
import pathlib
import torch
import torch.nn as nn
from torchvision import models

# Config 
WEIGHTS_SRC = "best_model_dogcat.pt"   # my trained weights (already in repo)
ART_DIR = pathlib.Path("artifacts")
ART_WEIGHTS = ART_DIR / "best_model_dogcat.pt"
ART_CLASSES = ART_DIR / "classes.json"

#  Ensure artifacts/ 
ART_DIR.mkdir(exist_ok=True)

# Classes 
with open("artifacts/classes.json") as f:
    CLASSES = json.load(f)

#  Build model & load weights 
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))

state = torch.load(WEIGHTS_SRC, map_location="cpu")
model.load_state_dict(state, strict=True)
model.eval()

# Save to artifacts
torch.save(model.state_dict(), ART_WEIGHTS)
with open(ART_CLASSES, "w") as f:
    json.dump(classes, f)

print("Exported:")
print("-", ART_WEIGHTS)
print("-", ART_CLASSES)
