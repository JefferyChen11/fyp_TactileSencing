import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# === Parameters ===
# MODEL_FILE     = "svm_model.joblib"       # pretrained SVM model file (optional)
MODEL_FILE     = "bayes_model.joblib"
NEW_DATA_CSV   = "data_rebuild_val_merge_calib_normalize_flatten_crop.csv"  # aligned new data CSV

# === Class, flavor and condition mappings ===
CLASS_NAMES = [
    'bigberry_ds20', 'citrus_ds20', 'rough_ds20', 'smallberry_ds20', 'smooth_ds20', 'strawberry_ds20',
    'bigberry_ds30', 'citrus_ds30', 'rough_ds30', 'smallberry_ds30', 'smooth_ds30', 'strawberry_ds30',
    'bigberry_ef10', 'citrus_ef10', 'rough_ef10', 'smallberry_ef10', 'smooth_ef10', 'strawberry_ef10',
    'bigberry_ef30', 'citrus_ef30', 'rough_ef30', 'smallberry_ef30', 'smooth_ef30', 'strawberry_ef30',
    'bigberry_ef50', 'citrus_ef50', 'rough_ef50', 'smallberry_ef50', 'smooth_ef50', 'strawberry_ef50'
]
texture    = ['smooth', 'strawberry', 'bigberry', 'citrus', 'rough', 'smallberry']
softness   = ['ds20', 'ds30', 'ef10', 'ef30', 'ef50']
flav_map   = {i: texture.index(name.split('_')[0])  for i, name in enumerate(CLASS_NAMES)}
cond_map   = {i: softness.index(name.split('_')[1]) for i, name in enumerate(CLASS_NAMES)}

def validate():
    # 1. Load model
    model = joblib.load(MODEL_FILE)

    # 2. Read new data (assumed aligned), and split features / labels
    df = pd.read_csv(NEW_DATA_CSV)
    X_new  = df.drop(columns=['label']).values
    y_true = df['label'].astype(int).values

    # 3. Predict
    y_pred = model.predict(X_new)
    acc    = np.mean(y_pred == y_true)
    print(f"Validation overall accuracy: {acc:.4f}\n")

    # 4. Classification report
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # 5. Confusion matrix & Softness×Texture accuracy
    cm = confusion_matrix(y_true, y_pred)

    # -- Added: normalize rows to obtain proportion matrix --
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = cm / row_sums
        cm_norm[np.isnan(cm_norm)] = 0.0  # handle division by zero

    # Compute accuracy for each Softness×Texture
    correct = np.zeros((len(texture), len(softness)), dtype=int)
    total   = np.zeros_like(correct)
    for t, p in zip(y_true, y_pred):
        i = flav_map[t]
        j = cond_map[t]
        total[i, j] += 1
        if t == p:
            correct[i, j] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_fc = correct / total
        acc_fc[np.isnan(acc_fc)] = 0.0

    # 6. Visualization (original paired plots retained)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    fig.suptitle("Validation Results", fontsize=20)

    # Confusion matrix (normalized proportions)
    im0 = ax0.imshow(cm_norm, cmap='Blues', vmin=0., vmax=1.)
    ax0.set_title("Confusion Matrix (Normalized)", fontsize=16)
    ticks = np.arange(len(CLASS_NAMES))
    ax0.set_xticks(ticks); ax0.set_yticks(ticks)
    ax0.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=6)
    ax0.set_yticklabels(CLASS_NAMES, fontsize=6)
    thresh = (cm_norm.max() + cm_norm.min()) / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            if val == 0:
                label = "0"
            elif val == 1:
                label = "1"
            else:
                label = f"{val:.2f}"
            color = 'white' if val > thresh else 'black'
            ax0.text(j, i, label, ha='center', va='center', color=color, fontsize=3)
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # Softness × Texture heatmap
    im1 = ax1.imshow(acc_fc, cmap='cividis', vmin=0., vmax=1.)
    ax1.set_title("Softness×Texture Accuracy", fontsize=16)
    ax1.set_xticks(np.arange(len(softness)))
    ax1.set_yticks(np.arange(len(texture)))
    ax1.set_xticklabels(softness, rotation=45, ha='right', fontsize=6)
    ax1.set_yticklabels(texture, fontsize=6)
    for i in range(len(texture)):
        for j in range(len(softness)):
            tc = 'black' if acc_fc[i, j] > 0.5 else 'white'
            ax1.text(j, i, f"{acc_fc[i, j]:.2f}", ha='center', va='center', color=tc, fontsize=8)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # ========= Add: Softness × Softness (normalized proportion) =========
    y_true_s = [cond_map[t] for t in y_true]
    y_pred_s = [cond_map[p] for p in y_pred]
    cm_s = confusion_matrix(y_true_s, y_pred_s, labels=list(range(len(softness))))
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = cm_s.sum(axis=1, keepdims=True)
        cm_s_norm = cm_s / rs
        cm_s_norm[np.isnan(cm_s_norm)] = 0.0

    fig2, ax2 = plt.subplots(figsize=(7, 7), dpi=150)
    im2 = ax2.imshow(cm_s_norm, cmap='Blues', vmin=0., vmax=1., interpolation='nearest', zorder=0)
    ax2.set_title('Softness Confusion Matrix (Proportion)', fontsize=18, pad=14)
    ax2.set_xlabel('Predicted label', fontsize=12)
    ax2.set_ylabel('True label', fontsize=12)

    ax2.set_xticks(np.arange(len(softness)))
    ax2.set_yticks(np.arange(len(softness)))
    ax2.set_xticklabels(softness, rotation=0, ha='center', fontsize=12)  # horizontal
    ax2.set_yticklabels(softness, fontsize=12)

    ax2.set_xticks(np.arange(len(softness)+1) - 0.5, minor=True)
    ax2.set_yticks(np.arange(len(softness)+1) - 0.5, minor=True)
    ax2.set_axisbelow(False)
    ax2.grid(which='minor', color='black', linewidth=1.5, zorder=3)

    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)

    for i in range(len(softness)):
        for j in range(len(softness)):
            v = cm_s_norm[i, j]
            if np.isclose(v, 0):
                txt = "0"
            elif np.isclose(v, 1):
                txt = "1"
            else:
                txt = f"{v:.2f}"
            clr = 'white' if v > 0.5 else 'black'
            ax2.text(j, i, txt, ha='center', va='center', color=clr, fontsize=12)

    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ========= Add: Texture × Texture (normalized proportion) =========
    y_true_t = [flav_map[t] for t in y_true]
    y_pred_t = [flav_map[p] for p in y_pred]
    cm_t = confusion_matrix(y_true_t, y_pred_t, labels=list(range(len(texture))))
    with np.errstate(divide='ignore', invalid='ignore'):
        rt = cm_t.sum(axis=1, keepdims=True)
        cm_t_norm = cm_t / rt
        cm_t_norm[np.isnan(cm_t_norm)] = 0.0

    fig3, ax3 = plt.subplots(figsize=(8.5, 8.5), dpi=150)
    im3 = ax3.imshow(cm_t_norm, cmap='Blues', vmin=0., vmax=1., interpolation='nearest', zorder=0)
    ax3.set_title('Texture Confusion Matrix (Proportion)', fontsize=18, pad=14)
    ax3.set_xlabel('Predicted label', fontsize=12)
    ax3.set_ylabel('True label', fontsize=12)

    ax3.set_xticks(np.arange(len(texture)))
    ax3.set_yticks(np.arange(len(texture)))
    ax3.set_xticklabels(texture, rotation=0, ha='center', fontsize=11)  # horizontal
    ax3.set_yticklabels(texture, fontsize=11)

    ax3.set_xticks(np.arange(len(texture)+1) - 0.5, minor=True)
    ax3.set_yticks(np.arange(len(texture)+1) - 0.5, minor=True)
    ax3.set_axisbelow(False)
    ax3.grid(which='minor', color='black', linewidth=1.5, zorder=3)

    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)

    for i in range(len(texture)):
        for j in range(len(texture)):
            v = cm_t_norm[i, j]
            if np.isclose(v, 0):
                txt = "0"
            elif np.isclose(v, 1):
                txt = "1"
            else:
                txt = f"{v:.2f}"
            clr = 'white' if v > 0.5 else 'black'
            ax3.text(j, i, txt, ha='center', va='center', color=clr, fontsize=11)

    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    validate()
