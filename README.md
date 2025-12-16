# Military-Aircraft-Detection-ResNet-18-Image-Classification
 【 ResNet-18 Image Classification Framework with 5-Fold Cross Validation 】

This repository provides a **complete PyTorch-based image classification training and evaluation pipeline** using **ResNet-18** with **5-Fold Cross Validation**.

It is designed for **research, academic projects, and reproducible experiments**, and is suitable for military aircraft recognition, remote sensing imagery, or general multi-class image classification tasks.

This implementation is **primarily developed and validated using the Military Aircraft Detection Dataset from Kaggle**.

---

## 🚀 Features

- ResNet-18 with ImageNet pretrained weights
- Class filtering via JSON configuration
- 5-Fold Cross Validation (K-Fold)
- Extensive data augmentation
- Automatic model checkpointing (per epoch & best model)
- Comprehensive evaluation metrics
- Rich visualizations for analysis

---

## 📁 Project Structure

```bash
.
├── Classifier_ResNet18_model.py   # Main training script
├── specified_classes.json        # Classes to include in training
├── crop/                          # Image dataset (ImageFolder format)
│   ├── class_A/
│   ├── class_B/
│   └── class_C/
└── /home/
    ├── FOLDA/
    ├── FOLDB/
    ├── FOLDC/
    ├── FOLDD/
    └── FOLDE/
```

---

## 🗂️ Primary Dataset

This project is **primarily designed and evaluated using the following Kaggle dataset**:

**Military Aircraft Detection Dataset (Kaggle)**  
<img width="2203" height="371" alt="image" src="https://github.com/user-attachments/assets/6361547c-3d02-4f6f-a2f1-88c90a11e3cf" />

🔗 https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data

### Dataset Description

- Multi-class military aircraft image dataset
- Images collected under diverse backgrounds and viewpoints
- Suitable for fine-grained military aircraft classification

The dataset can be directly adapted to this framework by organizing images into **PyTorch ImageFolder format** (`crop/class_name/*.jpg`).

> ⚠️ **Important**: This repository focuses on **image classification**, not object detection.  
> If the original dataset contains bounding boxes or annotations, images should be cropped or preprocessed beforehand.

---

## 🧠 Model Architecture

- **Backbone**: ResNet-18 (ImageNet pretrained)
- **Classifier Head**: Fully Connected Layer

```python
self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
```

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Learning Rate**: `5e-5`

---

## 🖼️ Dataset Format

### 1. Directory Structure (PyTorch ImageFolder)

```bash
crop/
├── class_1/
│   ├── img001.jpg
│   ├── img002.jpg
├── class_2/
│   ├── img001.jpg
└── class_3/
```

### 2. Class Selection (`specified_classes.json`)

```json
[
  "class_1",
  "class_2",
  "class_3"
]
```

> ⚠️ Only classes listed in this JSON file will be included in training. All other classes will be automatically excluded.

---

## 🔄 Data Preprocessing & Augmentation

```python
transforms.Resize((224, 224))
transforms.RandomHorizontalFlip()
transforms.RandomRotation(15)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
```

✔ Compatible with ImageNet pretrained models

---

## 🔁 Training Strategy

- **Cross Validation**: 5-Fold (KFold, shuffle=True, random_state=42)
- **Epochs**: 15
- **Batch Size**: 64
- **Device**:

```python
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
```

---

## 📊 Evaluation Metrics (per Epoch)

| Metric | Description |
|------|------------|
| Accuracy | Overall classification accuracy |
| Precision | Weighted precision |
| Recall | Weighted recall |
| F1-score | Weighted F1-score |
| Sensitivity | TP / (TP + FN) per class |
| Specificity | TN / (TN + FP) per class |

---

## 📈 Visualization Outputs

For **each fold and each epoch**, the following outputs are generated automatically:

- Confusion Matrix
- ROC Curve (One-vs-Rest + Mean ROC with AUC)
- Training & Validation Loss Curve
- Training & Validation Accuracy Curve

Example filenames:

```bash
confusion_matrix_fold_1_epoch_3.png
roc_curve_fold_1_epoch_3.png
loss_curve_fold_1.png
accuracy_curve_fold_1.png
```

---

## 💾 Model Checkpoints & Logs

### Model Weights

```bash
model_epoch_1.pth
model_epoch_2.pth
...
best_model.pth
```

- `best_model.pth` is selected automatically based on **highest validation accuracy** across all folds and epochs.

### Metrics (JSON)

```json
{
  "epoch": 3,
  "train_loss": 0.42,
  "train_accuracy": 85.3,
  "val_loss": 0.55,
  "val_accuracy": 82.1,
  "precision": 0.83,
  "recall": 0.82,
  "f1_score": 0.82,
  "sensitivity": [...],
  "specificity": [...]
}
```

---

## ▶️ How to Run

```bash
python Classifier_ResNet18_model.py
```

Before running, make sure:

- `crop/` dataset directory exists
- `specified_classes.json` is correctly configured
- Required Python packages are installed

---

## 📦 Requirements

```txt
torch
torchvision
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

Recommended:

- Python ≥ 3.8
- CUDA-enabled GPU for faster training

---

## 🔧 Configurable Parameters

```python
EPOCHS = 15
batch_size = 64
learning_rate = 0.00005
n_splits = 5
```

---

## ⚠️ Notes

- ROC curves are computed using a **One-vs-Rest** strategy
- If a class is missing in a validation fold, ROC computation for that class is skipped
- Weighted metrics are used to mitigate class imbalance

---

## 📜 License

This project is intended for **research and academic use only**.

Please ensure proper evaluation before any production or clinical deployment.

---

## ✉️ Experimental Results

<img width="1555" height="892" alt="image" src="https://github.com/user-attachments/assets/0471f70c-a1e5-4012-84f3-8d3363742418" />
<img width="1789" height="272" alt="image" src="https://github.com/user-attachments/assets/1d254255-d361-4e3e-bf21-d66af46ec69e" />


⭐ If you find this project useful, consider starring the repository!


