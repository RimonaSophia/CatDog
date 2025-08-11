import os, shutil, random, time, json, pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ----------- Config -----------
original_dir = "PATH/TO/YOUR/flat_dataset"   # <- change this: folder with files like catXXXX.jpg, dogXXXX.jpg
output_dir   = "output_data"
train_ratio  = 0.8
batch_size   = 32
epochs       = 10
patience     = 3

pathlib.Path(output_dir).mkdir(exist_ok=True)

def already_split():
    expected = [
        os.path.join(output_dir, 'train', 'cats'),
        os.path.join(output_dir, 'train', 'dogs'),
        os.path.join(output_dir, 'val',   'cats'),
        os.path.join(output_dir, 'val',   'dogs'),
    ]
    return all(os.path.isdir(p) and len(os.listdir(p)) > 0 for p in expected)

if not already_split():
    for split in ['train', 'val']:
        for category in ['cats', 'dogs']:
            os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

    all_images = os.listdir(original_dir)
    cat_images = [f for f in all_images if f.lower().startswith('cat')]
    dog_images = [f for f in all_images if f.lower().startswith('dog')]

    random.seed(42)
    random.shuffle(cat_images)
    random.shuffle(dog_images)

    def split_and_copy(images, category):
        split_index = int(len(images) * train_ratio)
        train_files = images[:split_index]
        val_files   = images[split_index:]
        for file in train_files:
            shutil.copy(os.path.join(original_dir, file), os.path.join(output_dir, 'train', category))
        for file in val_files:
            shutil.copy(os.path.join(original_dir, file), os.path.join(output_dir, 'val', category))

    print("Splitting and copying dataset... (one-time)")
    t0 = time.time()
    split_and_copy(cat_images, 'cats')
    split_and_copy(dog_images, 'dogs')
    print(f"Dataset split complete in {time.time()-t0:.1f}s")

# ----------- Device -----------
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Device:", device)

# ----------- Transforms -----------
# If you trained with Normalize, add it to BOTH train and val and mirror in app.py.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ----------- Datasets & Loaders -----------
train_data = datasets.ImageFolder(os.path.join(output_dir, "train"), transform=train_transform)
val_data   = datasets.ImageFolder(os.path.join(output_dir, "val"),   transform=val_transform)
assert len(train_data) > 0 and len(val_data) > 0, "Empty dataset after split."

num_workers = 0   # safest for notebooks/macOS; increase when running as a script
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False)
val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

# ----------- Model -----------
try:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
except Exception:
    model = models.resnet18(pretrained=True)

for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-4)

best_val = float('inf')
bad_epochs = 0

for epoch in range(1, epochs + 1):
    # Train
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        bs = labels.size(0)
        run_loss += loss.item() * bs
        pred = out.argmax(1)
        correct += (pred == labels).sum().item()
        total += bs
    train_loss = run_loss / total
    train_acc  = correct / total

    # Validate
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            bs = labels.size(0)
            v_loss += loss.item() * bs
            v_correct += (out.argmax(1) == labels).sum().item()
            v_total += bs
    val_loss = v_loss / v_total
    val_acc  = v_correct / v_total

    print(f"Epoch {epoch:02d}/{epochs} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        bad_epochs = 0
        torch.save(model.state_dict(), "best_model_catdog.pt")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss {best_val:.4f})")
            break

print("Training done. Best model saved to best_model_catdog.pt")

# ----------- Export artifacts -----------
pathlib.Path("artifacts").mkdir(exist_ok=True)
shutil.copy("best_model_catdog.pt", "artifacts/best_model_catdog.pt")
with open("artifacts/classes.json", "w") as f:
    json.dump(train_data.classes, f)
print("Wrote artifacts/best_model_catdog.pt and artifacts/classes.json")