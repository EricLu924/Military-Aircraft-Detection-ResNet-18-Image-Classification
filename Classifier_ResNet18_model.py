import os
import json
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 加載指定類別的 JSON 文件
with open('specified_classes.json', 'r') as f:
    specified_classes = json.load(f)
print(f"Loaded specified classes: {specified_classes}")

# 數據處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加載數據集
data_dir = 'crop'
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 過濾數據集中的樣本
filtered_classes_to_idx = {cls: idx for idx, cls in enumerate(specified_classes)}
filtered_samples = []
for s in dataset.samples:
    class_name = os.path.basename(os.path.dirname(s[0]))
    if class_name in filtered_classes_to_idx:
        new_label = filtered_classes_to_idx[class_name]
        filtered_samples.append((s[0], new_label))

# 更新數據集
dataset.samples = filtered_samples
dataset.classes = list(filtered_classes_to_idx.keys())
dataset.class_to_idx = filtered_classes_to_idx

# 確認類別數量
num_classes = len(dataset.classes)
print(f"使用的類別數: {num_classes}")
print(f"類別列表: {dataset.classes}")

# 設置 5 折交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=42)
EPOCHS = 15
batch_size = 64
output_dir = '/home'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

best_accuracy = 0

# 定義模型-ResNet-18
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# 繪製混淆矩陣
def plot_confusion_matrix(cm, classes, fold, epoch, output_dir):
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title(f'Confusion Matrix for Fold {fold+1} - Epoch {epoch+1}', fontsize=18, fontweight='bold')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_fold_{fold+1}_epoch_{epoch+1}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 繪製平均 ROC 曲線
def plot_roc_curve(all_labels, all_preds, classes, fold, epoch, output_dir):
    mean_fpr = np.linspace(0, 1, 100) 
    tprs = []  # 存TPR
    aucs = []  # 存AUC

    plt.figure(figsize=(20, 15)) 

    for i, cls in enumerate(classes):
        # 計算每個類別的 ROC 曲線和 AUC
        binary_labels = [1 if label == i else 0 for label in all_labels]
        binary_preds = [1 if pred == i else 0 for pred in all_preds]
        if len(np.unique(binary_labels)) == 1:
            # 如果某個類別在當前驗證集中只有一種標籤，跳過 ROC 計算
            print(f"Warning: Only one class present in labels for class {cls}. ROC curve is not defined.")
            continue
        fpr, tpr, _ = roc_curve(binary_labels, binary_preds)
        auc_score = roc_auc_score(binary_labels, binary_preds)
        aucs.append(auc_score)

        # 插值到統一的 FPR 範圍
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0 
        tprs.append(tpr_interp)

        # 繪製每個類別的曲線
        plt.plot(fpr, tpr, alpha=0.4, label=f'{cls} (AUC = {auc_score:.2f})')

    # 計算平均 TPR 和標準差
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # 繪製平均 ROC 曲線
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=0.8)

    # 繪製標準差範圍
    std_tpr = np.std(tprs, axis=0)
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='gray', alpha=0.2, label='± 1 std. dev.')

    # 對角線
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Chance', lw=2)

    # 圖表優化
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve for Fold {fold+1} - Epoch {epoch+1}', fontsize=18, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.grid(visible=True, linestyle='--', alpha=0.6)


    plt.savefig(os.path.join(output_dir, f'roc_curve_fold_{fold+1}_epoch_{epoch+1}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 計算敏感度和特異性
def calculate_sensitivity_specificity(cm):
    sensitivity = []
    specificity = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(sum(cm)) - (tp + fn + fp)
        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    return sensitivity, specificity

# 保存評估指標
def save_metrics(metrics, fold_dir, epoch):
    metrics_path = os.path.join(fold_dir, f'metrics_epoch_{epoch+1}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

# 訓練和驗證
for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
    fold_dir = os.path.join(output_dir, f'FOLD{chr(65 + fold)}')
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    
    print(f"\nFold {fold+1}/5")
    print("-" * 20)
    
    train_subset = Subset(dataset, train_idx)
    valid_subset = Subset(dataset, valid_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size)

    model = MyModel(num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                tepoch.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Training - Average Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # 驗證階段
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / len(valid_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Valid - Average Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        # 計算指標
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        sensitivity, specificity = calculate_sensitivity_specificity(cm)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        print(f"Sensitivity: {sensitivity}")
        print(f"Specificity: {specificity}")

        # 保存評估指標
        metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "sensitivity": sensitivity,
            "specificity": specificity
        }
        save_metrics(metrics, fold_dir, epoch)

        # 繪製混淆矩陣
        plot_confusion_matrix(cm, dataset.classes, fold, epoch, fold_dir)

        # 繪製 ROC 曲線
        plot_roc_curve(all_labels, all_preds, dataset.classes, fold, epoch, fold_dir)

        # 保存模型權重
        model_path = os.path.join(fold_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)

        # 更新最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_path = os.path.join(fold_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)

    # 繪製 Loss 和 Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
    plt.title(f'Fold {fold+1} - Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(fold_dir, f'loss_curve_fold_{fold+1}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, EPOCHS+1), val_accuracies, label='Validation Accuracy')
    plt.title(f'Fold {fold+1} - Accuracy Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(fold_dir, f'accuracy_curve_fold_{fold+1}.png'), dpi=300, bbox_inches='tight')
    plt.close()
