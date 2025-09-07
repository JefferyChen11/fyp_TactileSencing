import warnings
import numpy as np
import numpy.ma as ma
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score
)
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

# Suppress a specific noisy warning
warnings.filterwarnings("ignore",
    message="The number of unique classes is greater than 50% of the number of samples."
)

# === Parameters ===
FLATTEN_CSV        = "data_rebuild_merge_calib_crop_normalize_flatten.csv"
MODEL_FILE         = "svm_model.joblib"
FEATURE_NAMES_FILE = "feature_names.joblib"
N_SPLITS           = 5
THRESHOLD          = 1e-4  # Mask cells in confusion matrices below this threshold (shown as white)

# === Class list and mappings ===
CLASS_NAMES = [
    'bigberry_ds20', 'citrus_ds20', 'rough_ds20', 'smallberry_ds20', 'smooth_ds20', 'strawberry_ds20',
    'bigberry_ds30', 'citrus_ds30', 'rough_ds30', 'smallberry_ds30', 'smooth_ds30', 'strawberry_ds30',
    'bigberry_ef10', 'citrus_ef10', 'rough_ef10', 'smallberry_ef10', 'smooth_ef10', 'strawberry_ef10',
    'bigberry_ef30', 'citrus_ef30', 'rough_ef30', 'smallberry_ef30', 'smooth_ef30', 'strawberry_ef30',
    'bigberry_ef50', 'citrus_ef50', 'rough_ef50', 'smallberry_ef50', 'smooth_ef50', 'strawberry_ef50'
]
texture  = ['smooth', 'strawberry', 'bigberry', 'citrus', 'rough', 'smallberry']
softness = ['ds20', 'ds30', 'ef10', 'ef30', 'ef50']

flav_map = {i: texture.index(name.split('_')[0]) for i, name in enumerate(CLASS_NAMES)}
cond_map = {i: softness.index(name.split('_')[1]) for i, name in enumerate(CLASS_NAMES)}


def plot_aggregated(y_true, y_pred):
    """Plot aggregated results: proportional confusion matrix + Softness×Texture heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    row_sum = cm.sum(axis=1, keepdims=True)
    ratio = np.divide(cm, row_sum, where=row_sum != 0)
    masked = ma.masked_less(ratio, THRESHOLD)
    cmap0 = plt.cm.Blues.copy()
    cmap0.set_bad(color='white')

    # Compute Softness×Texture accuracy grid
    correct = np.zeros((len(texture), len(softness)), dtype=int)
    total   = np.zeros_like(correct)
    for t, p in zip(y_true, y_pred):
        i, j = flav_map[t], cond_map[t]
        total[i, j] += 1
        if t == p:
            correct[i, j] += 1
    acc_fc = np.divide(correct, total, where=total != 0)
    acc_fc = np.nan_to_num(acc_fc)

    # ===== First figure: two panels (confusion matrix proportion + Softness×Texture heatmap) =====
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    fig.suptitle("Cross-Validation Aggregated Results", fontsize=20)

    # Proportional confusion matrix
    ax0.imshow(masked, cmap=cmap0, vmin=THRESHOLD, vmax=1.0, interpolation='nearest')
    ax0.set_title('Confusion Matrix (Proportion)', fontsize=16)
    n = len(CLASS_NAMES)
    ax0.set_xticks(np.arange(n)); ax0.set_yticks(np.arange(n))
    ax0.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=3)
    ax0.set_yticklabels(CLASS_NAMES, fontsize=3)
    ax0.set_xticks(np.arange(n+1)-0.5, minor=True)
    ax0.set_yticks(np.arange(n+1)-0.5, minor=True)
    ax0.grid(which='minor', color='white', linewidth=1)
    for spine in ax0.spines.values(): spine.set_visible(False)
    for i in range(n):
        for j in range(n):
            r = ratio[i, j]
            txt = '0' if r < THRESHOLD else f"{r:.2f}"
            clr = 'white' if r > 0.5 else 'black'
            ax0.text(j, i, txt, ha='center', va='center', color=clr, fontsize=3)
    c0 = plt.colorbar(ax0.images[0], ax=ax0, fraction=0.046, pad=0.4/10)
    c0.ax.tick_params(labelsize=6)

    # Softness × Texture accuracy heatmap
    im1 = ax1.imshow(acc_fc, cmap='cividis', vmin=0.0, vmax=1.0, aspect='equal')
    ax1.set_title('Softness×Texture Accuracy', fontsize=16)
    ax1.set_xticks(np.arange(len(softness))); ax1.set_yticks(np.arange(len(texture)))
    ax1.set_xticklabels(softness, rotation=45, ha='right', fontsize=12)
    ax1.set_yticklabels(texture, fontsize=12)
    ax1.set_xticks(np.arange(len(softness)+1)-0.5, minor=True)
    ax1.set_yticks(np.arange(len(texture)+1)-0.5, minor=True)
    ax1.grid(which='minor', color='white', linewidth=1)
    for spine in ax1.spines.values(): spine.set_visible(False)
    for i in range(len(texture)):
        for j in range(len(softness)):
            v = acc_fc[i, j]
            txt = '0' if v == 0 else f"{v:.2f}"
            clr = 'black' if v > 0.5 else 'white'
            ax1.text(j, i, txt, ha='center', va='center', color=clr, fontsize=12)
    c1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.4/10)
    c1.ax.tick_params(labelsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # ===== Below two new plots: thicker borders, stronger grid, keep titles, make x-axis labels horizontal =====

    # 1) Softness × Softness confusion matrix
    y_true_s = [cond_map[t] for t in y_true]
    y_pred_s = [cond_map[p] for p in y_pred]

    cm_s    = confusion_matrix(y_true_s, y_pred_s)
    sums_s  = cm_s.sum(axis=1, keepdims=True)
    ratio_s = np.divide(cm_s, sums_s, where=sums_s != 0)
    masked_s = ma.masked_less(ratio_s, THRESHOLD)
    cmap_s   = plt.cm.Blues.copy()
    cmap_s.set_bad(color='white')

    fig2, ax2 = plt.subplots(figsize=(7, 7), dpi=150)
    im2 = ax2.imshow(masked_s, cmap=cmap_s, vmin=THRESHOLD, vmax=1.0,
                     interpolation='nearest', zorder=0)

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
            r = ratio_s[i, j]
            txt = '0' if r < THRESHOLD else f"{r:.2f}"
            clr = 'white' if r > 0.5 else 'black'
            ax2.text(j, i, txt, ha='center', va='center', color=clr, fontsize=12)

    c2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    c2.ax.tick_params(labelsize=8)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # 2) Texture × Texture confusion matrix
    y_true_t = [flav_map[t] for t in y_true]
    y_pred_t = [flav_map[p] for p in y_pred]

    cm_t    = confusion_matrix(y_true_t, y_pred_t)
    sums_t  = cm_t.sum(axis=1, keepdims=True)
    ratio_t = np.divide(cm_t, sums_t, where=sums_t != 0)
    masked_t = ma.masked_less(ratio_t, THRESHOLD)
    cmap_t   = plt.cm.Blues.copy()
    cmap_t.set_bad(color='white')

    fig3, ax3 = plt.subplots(figsize=(8.5, 8.5), dpi=150)
    im3 = ax3.imshow(masked_t, cmap=cmap_t, vmin=THRESHOLD, vmax=1.0,
                     interpolation='nearest', zorder=0)

    ax3.set_title('Texture Confusion Matrix (Proportion)', fontsize=18, pad=14)
    ax3.set_xlabel('Predicted label', fontsize=12)
    ax3.set_ylabel('True label', fontsize=12)

    ax3.set_xticks(np.arange(len(texture)))
    ax3.set_yticks(np.arange(len(texture)))
    ax3.set_xticklabels(texture, rotation=0, ha='center', fontsize=11)   # horizontal
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
            r = ratio_t[i, j]
            txt = '0' if r < THRESHOLD else f"{r:.2f}"
            clr = 'white' if r > 0.5 else 'black'
            ax3.text(j, i, txt, ha='center', va='center', color=clr, fontsize=11)

    c3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    c3.ax.tick_params(labelsize=8)
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    # 1. Load data
    df = pd.read_csv(FLATTEN_CSV)
    feature_names = df.drop(columns=['label']).columns.tolist()
    joblib.dump(feature_names, FEATURE_NAMES_FILE)
    X = df[feature_names].values
    y = df['label'].astype(int).values

    # 2. Stratified 5-fold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # 3. Grid Search
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc',    SVC())
    ])
    param_grid = {
        'svc__kernel': ['rbf', 'poly'],
        'svc__C':      [1, 10, 100],
        'svc__gamma':  ['scale', 'auto', 1e-3],
        'svc__degree': [2, 3]
    }
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='accuracy',
        cv=skf,
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X, y)

    # 3.a Print mean accuracy for each hyperparameter combination
    print("\nCross-validation results for each parameter combination:")
    for params, mean_acc in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
        print(f"Params {params} finished, accuracy: {mean_acc:.4f}")

    # 3.b Summarize best parameters and their accuracy and f1
    best_params = grid.best_params_
    best_acc = grid.best_score_
    best_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc',    SVC(
            kernel=best_params['svc__kernel'],
            C=best_params['svc__C'],
            gamma=best_params['svc__gamma'],
            degree=best_params['svc__degree']
        ))
    ])
    f1_scores = cross_val_score(best_pipe, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
    best_f1 = f1_scores.mean()
    print(f"\nBest parameters: {best_params}, accuracy: {best_acc:.4f}, f1 score: {best_f1:.4f}\n")

    # 4. Evaluate with stratified 5-fold using best params
    y_true_all, y_pred_all = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        model = SVC(
            kernel=best_params['svc__kernel'],
            C=best_params['svc__C'],
            gamma=best_params['svc__gamma'],
            degree=best_params['svc__degree']
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        acc   = accuracy_score(y_te, preds)
        prec  = precision_score(y_te, preds, average='macro', zero_division=0)
        rec   = recall_score(y_te, preds,    average='macro', zero_division=0)
        f1    = f1_score(y_te, preds,        average='macro', zero_division=0)

        print(f"fold {fold} — accuracy: {acc:.4f}, "
              f"precision_macro: {prec:.4f}, "
              f"recall_macro: {rec:.4f}, "
              f"f1 score: {f1:.4f}")

        y_true_all.extend(y_te)
        y_pred_all.extend(preds)

    # 5. Compute overall metrics
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    overall_acc  = accuracy_score(y_true_all, y_pred_all)
    overall_prec = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    overall_rec  = recall_score(y_true_all, y_pred_all,    average='macro', zero_division=0)
    overall_f1   = f1_score(y_true_all, y_pred_all,        average='macro', zero_division=0)

    print("\nOverall Results with Best Params:")
    print(f"accuracy: {overall_acc:.4f}, "
          f"precision_macro: {overall_prec:.4f}, "
          f"recall_macro: {overall_rec:.4f}, "
          f"f1 score: {overall_f1:.4f}")

    # 6. Train final model on full data and save
    final_model = SVC(
        kernel=best_params['svc__kernel'],
        C=best_params['svc__C'],
        gamma=best_params['svc__gamma'],
        degree=best_params['svc__degree']
    )
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_FILE)
    print(f"Final model saved to {MODEL_FILE}")

    # 7. Plot aggregated visualizations
    plot_aggregated(y_true_all, y_pred_all)


if __name__ == '__main__':
    main()
