# Cat vs Dog Image Classification

## Project Overview

This project focuses on classifying **cat and dog images** using a **ResNet18** pretrained model with **transfer learning** in **PyTorch**. It demonstrates how to adapt pretrained architectures for **binary classification** and deploy them using a **Gradio** app for interactive predictions.

The repository includes:
* Training a fine-tuned ResNet18 model
* Exporting model artifacts
* Deploying an app with Gradio

---

## Dataset Setup & Preprocessing

* **Classes**: 2 → `cat`, `dog`
* **Dataset Split**: Training and validation sets
* **Transforms Applied**:
  * Resize images to **224×224**
  * Convert to tensors
  * Normalize inputs for ResNet18 compatibility
* **Artifacts Generated**:
  * `best_model_dogcat.pt` → trained model weights
  * `classes.json` → label mappings

---

## Model Training

* **Model**: `ResNet18` pretrained on ImageNet
* **Training Configuration**:
  * Epochs: 10
  * Image size: 224×224
  * Batch size: adjustable
  * Optimizer: **Adam**
  * Loss: **CrossEntropyLoss**
  * Device: CPU / GPU

### Training Summary

* Fine-tuned the **fully connected layer** for 2-class classification
* Training and evaluation handled in `Train_DogCat.ipynb`
* Exported the trained model and label mappings into the `artifacts/` directory

---

## Model Export

After training, run the export script to prepare artifacts for deployment:

python Export.py

This will:

* Rebuild the **ResNet18** model  
* Load trained weights  
* Save the packaged model and class mappings into `artifacts/`

---

## Inference & Predictions

Run the Gradio app locally for interactive testing:

python app.py

* Upload an image of a cat or dog
* View predictions with confidence scores
* Preloaded example images are available in the examples/ folder

---

## Model Performance

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 01    | 0.1020    | 95.96%    | 0.0599   | 97.76%  |
| 02    | 0.0961    | 96.39%    | 0.0583   | **97.78%** |
| 03    | 0.0949    | 96.29%    | 0.0636   | 97.52%  |
| 04    | 0.0890    | 96.60%    | 0.0732   | 97.04%  |
| 05    | 0.0919    | 96.25%    | 0.0672   | 97.32%  |

Best validation accuracy: 97.78% (epoch 2)

---

## Next Steps

* Compute **precision**, **recall**, and **F1-score** for deeper evaluation  
* Extend to **multi-class classification** (e.g., different breeds)  
* Add **Grad-CAM visualizations** for explainability    
* Experiment with alternative architectures like **EfficientNet** or **ViT**



