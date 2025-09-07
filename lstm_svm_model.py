import os, sys, faulthandler
faulthandler.enable()

# -- runtime control --
DO_OOF         = True   # whether to run 5-fold OOF evaluation
DO_FINAL_TRAIN = True   # whether to retrain on full data and save final models

# -- stability / performance settings (recommended for Windows CPU) --
USE_MKLDNN   = False
TORCH_THREADS = 4
TORCH_INTEROP = 2
BATCH_SIZE    = 64
HIDDEN        = 384      
PROJ_DIM      = 512
EPOCHS        = 50
WARMUP_EPOCHS = 3
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 8
GRAD_CLIP     = 1.0
THRESHOLD     = 1e-4
EPS           = 1e-8

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings
import numpy as np
import numpy.ma as ma
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

# -- configure PyTorch backend / threads --
try:
    torch.backends.mkldnn.enabled = bool(USE_MKLDNN)
except Exception:
    pass
try:
    torch.set_num_threads(int(TORCH_THREADS))
    torch.set_num_interop_threads(int(TORCH_INTEROP))
except Exception:
    pass

# ========= paths & shapes =========
# main training set (cropped to 250 frames)
NPZ_FILE_MAIN = "data_rebuild_merge_calib_crop.npz"

# extra _200 training set: fixed "per-class n[0,200] samples", total 30n (configurable)
NPZ_FILE_EXTRA_200     = "data_rebuild_merge_calib_crop_200.npz"
EXTRA_200_PER_CLASS    = 40
EXTRA_200_RANDOM_STATE = 42

SAVE_ENCODER  = "lstm_encoder.pt"
SAVE_SVM      = "lstm_svm_model.joblib"

N_SPLITS      = 5
RANDOM_STATE  = 42

# Note: T is 250 (post-crop length)
T, C, D = 250, 8, 3
F = C * D  # 24 features per time step

# ========= default classes & maps (used if NPZ doesn't contain 'classes') =========
CLASS_NAMES = [
    'bigberry_ds20', 'citrus_ds20', 'rough_ds20', 'smallberry_ds20', 'smooth_ds20', 'strawberry_ds20',
    'bigberry_ds30', 'citrus_ds30', 'rough_ds30', 'smallberry_ds30', 'smooth_ds30', 'strawberry_ds30',
    'bigberry_ef10', 'citrus_ef10', 'rough_ef10', 'smallberry_ef10', 'smooth_ef10', 'strawberry_ef10',
    'bigberry_ef30', 'citrus_ef30', 'rough_ef30', 'smallberry_ef30', 'smooth_ef30', 'strawberry_ef30',
    'bigberry_ef50', 'citrus_ef50', 'rough_ef50', 'smallberry_ef50', 'smooth_ef50', 'strawberry_ef50'
]
texture  = ['smooth', 'strawberry', 'bigberry', 'citrus', 'rough', 'smallberry']
softness = ['ds20', 'ds30', 'ef10', 'ef30', 'ef50']
flav_map = {i: texture.index(n.split('_')[0]) for i, n in enumerate(CLASS_NAMES)}
cond_map = {i: softness.index(n.split('_')[1]) for i, n in enumerate(CLASS_NAMES)}

warnings.filterwarnings("ignore",
    message="The number of unique classes is greater than 50% of the number of samples."
)

# ========= utilities =========
def seed_all(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def minmax_timewise(X):
    # min-max per sample along time axis
    Xmin = X.min(axis=1, keepdims=True)
    Xmax = X.max(axis=1, keepdims=True)
    return (X - Xmin) / (Xmax - Xmin + EPS)

def to_seq(X):
    # reshape to (N, T, F)
    return X.reshape(X.shape[0], T, F).astype(np.float32)

def time_warp_to_T(seq_tf, scale):
    # time-warp by scaling the time axis, then interpolate back to original T
    T0, F0 = seq_tf.shape
    t = np.arange(T0, dtype=np.float32)
    q = (t * scale).clip(0, T0 - 1)
    out = np.empty_like(seq_tf)
    for j in range(F0):
        out[:, j] = np.interp(q, t, seq_tf[:, j])
    return out.astype(np.float32)

def time_shift_roll(seq_tf, shift):
    if shift == 0: return seq_tf
    return np.roll(seq_tf, shift=shift, axis=0)

def time_random_crop_resize(seq_tf, keep_min=0.7):
    # random crop then resize back to original length by interpolation
    T0, F0 = seq_tf.shape
    L = int(np.random.uniform(keep_min, 1.0) * T0)
    if L >= T0:
        return seq_tf.astype(np.float32)
    start = int(np.random.randint(0, T0 - L + 1))
    crop = seq_tf[start:start+L]
    t_src = np.linspace(0, L-1, num=L, dtype=np.float32)
    t_dst = np.linspace(0, L-1, num=T0, dtype=np.float32)
    out = np.empty((T0, F0), dtype=np.float32)
    for j in range(F0):
        out[:, j] = np.interp(t_dst, t_src, crop[:, j])
    return out

# ========= Dataset classes =========
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

class AugSeqDataset(Dataset):
    """
    Training augmentations (applied in sequence):
      1) random time crop -> resize
      2) time-warp (scale 0.9~1.1)
      3) time-shift (±60)
      3.5) per-sensor amplitude jitter (each sensor's 3 axes multiplied by 0.8~1.2)
      4) random sensor drop (1~2 sensors zeroed across all timesteps)
      5) time mask (1~2 segments, length 10~40)
      6) small additive noise (sigma=0.01)
    """
    def __init__(self, X, y, rng=None):
        self.X = X; self.y = y.astype(np.int64)
        self.rng = np.random.default_rng(123) if rng is None else rng
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = self.X[i].copy()

        if self.rng.random() < 0.7:
            x = time_random_crop_resize(x, keep_min=0.7)

        if self.rng.random() < 0.8:
            s = self.rng.uniform(0.9, 1.1); x = time_warp_to_T(x, s)

        if self.rng.random() < 0.8:
            sh = int(self.rng.integers(-60, 61)); x = time_shift_roll(x, sh)

        # channel amplitude jitter (per sensor)
        if self.rng.random() < 0.7:
            xr = x.reshape(T, 8, 3)
            scales = self.rng.uniform(0.8, 1.2, size=(8,)).astype(np.float32)
            xr *= scales[None, :, None]
            x = xr.reshape(T, 24)

        # random sensor drop
        if self.rng.random() < 0.5:
            xr = x.reshape(T, 8, 3)
            k = int(self.rng.integers(1, 3))
            idx = self.rng.choice(8, size=k, replace=False)
            xr[:, idx, :] = 0.0
            x = xr.reshape(T, 24)

        # time masks
        if self.rng.random() < 0.7:
            m = int(self.rng.integers(1, 3))
            for _ in range(m):
                L = int(self.rng.integers(10, 41))
                s0 = int(self.rng.integers(0, T - L + 1))
                x[s0:s0+L] = 0.0

        # small additive noise
        if self.rng.random() < 0.7:
            x = x + self.rng.normal(0.0, 0.01, size=x.shape).astype(np.float32)
            x = np.clip(x, 0.0, 1.0)

        return torch.from_numpy(x), torch.tensor(self.y[i])

# ========= Model definitions =========
class AttentionPool(nn.Module):
    def __init__(self, dim, attn_dim=None):
        super().__init__()
        a = attn_dim or max(64, dim // 2)
        self.fc = nn.Sequential(nn.Linear(dim, a), nn.Tanh(), nn.Linear(a, 1))
    def forward(self, y):
        a = self.fc(y)               # (B,T,1)
        w = torch.softmax(a, dim=1)  # (B,T,1)
        attn = (w * y).sum(dim=1)    # (B,dim)
        return attn, w

class LSTMEncoder(nn.Module):
    def __init__(self, input_size=F, hidden=HIDDEN, num_layers=2, bidirectional=True,
                 dropout=0.5, proj_dim=PROJ_DIM, num_classes=30):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        out_dim = hidden * (2 if bidirectional else 1)
        self.attn = AttentionPool(out_dim)
        self.ln   = nn.LayerNorm(out_dim * 2)
        self.proj = nn.Linear(out_dim * 2, proj_dim)
        self.head = nn.Linear(proj_dim, num_classes)
    def forward(self, x):
        y, _ = self.lstm(x)
        attn_vec, _ = self.attn(y)
        y_max, _    = y.max(dim=1)
        feat = torch.cat([attn_vec, y_max], dim=1)
        feat = self.ln(feat)
        feat = torch.relu(self.proj(feat))
        logits = self.head(feat)
        return logits, feat

def make_criterion(class_weights=None):
    try:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    except TypeError:
        return nn.CrossEntropyLoss(weight=class_weights)

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups: pg["lr"] = lr

def train_encoder(model, train_loader, val_loader, device, class_weights=None):
    crit = make_criterion(class_weights)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cos  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    best_acc, best_state, no_imp = -1.0, None, 0
    for ep in range(1, EPOCHS+1):
        model.train(); run_loss = 0.0; n = 0
        if ep <= WARMUP_EPOCHS:
            set_lr(opt, LR * ep / max(1, WARMUP_EPOCHS))
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            if GRAD_CLIP is not None:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            run_loss += loss.item() * xb.size(0); n += xb.size(0)
        tr_loss = run_loss / max(1, n)

        model.eval(); corr=0; tot=0; val_loss=0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb=xb.to(device); yb=yb.to(device)
                logits, _ = model(xb)
                loss = crit(logits, yb)
                val_loss += loss.item() * xb.size(0)
                corr += (logits.argmax(1)==yb).sum().item()
                tot  += xb.size(0)
        val_acc = corr / max(1, tot)
        val_loss = val_loss / max(1, tot)
        cos.step()
        print(f"Ep{ep:02d}  train_loss={tr_loss:.4f}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print("  Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def encode_features(model, loader, device):
    model.eval(); feats=[]; labels=[]
    for xb, yb in loader:
        xb = xb.to(device)
        _, f = model(xb)
        feats.append(f.cpu().numpy()); labels.append(yb.numpy())
    return np.vstack(feats), np.concatenate(labels)

# ========= plotting =========
def plot_aggregated(y_true, y_pred, title_prefix="Cross-Validation Aggregated Results (Attn LSTM → SVM)"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    row_sum = cm.sum(axis=1, keepdims=True)
    ratio = np.divide(cm, row_sum, where=row_sum!=0)
    masked = ma.masked_less(ratio, THRESHOLD)
    cmap0 = plt.cm.Blues.copy(); cmap0.set_bad(color='white')

    correct = np.zeros((len(texture), len(softness)), dtype=int)
    total   = np.zeros_like(correct)
    for t, p in zip(y_true, y_pred):
        i, j = flav_map[t], cond_map[t]
        total[i,j]+=1
        if t==p: correct[i,j]+=1
    acc_fc = np.divide(correct, total, where=total!=0); acc_fc = np.nan_to_num(acc_fc)

    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(16,6),dpi=150)
    fig.suptitle(title_prefix, fontsize=20)
    im0=ax0.imshow(masked, cmap=cmap0, vmin=THRESHOLD, vmax=1.0, interpolation='nearest')
    n=len(CLASS_NAMES)
    ax0.set_title('Confusion Matrix (Proportion)', fontsize=16)
    ax0.set_xticks(np.arange(n)); ax0.set_yticks(np.arange(n))
    ax0.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=3)
    ax0.set_yticklabels(CLASS_NAMES, fontsize=3)
    ax0.set_xticks(np.arange(n+1)-0.5, minor=True)
    ax0.set_yticks(np.arange(n+1)-0.5, minor=True)
    ax0.grid(which='minor', color='white', linewidth=1)
    for i in range(n):
        for j in range(n):
            r=ratio[i,j]; txt='0' if r<THRESHOLD else f"{r:.2f}"
            clr='white' if r>0.5 else 'black'
            ax0.text(j,i,txt,ha='center',va='center',color=clr,fontsize=3)
    plt.colorbar(im0,ax=ax0,fraction=0.046,pad=0.4/10)

    im1=ax1.imshow(acc_fc, cmap='cividis', vmin=0.0, vmax=1.0, aspect='equal')
    ax1.set_title('Softness×Texture Accuracy', fontsize=16)
    ax1.set_xticks(np.arange(len(softness))); ax1.set_yticks(np.arange(len(texture)))
    ax1.set_xticklabels(softness, rotation=45, ha='right', fontsize=12)
    ax1.set_yticklabels(texture, fontsize=12)
    ax1.set_xticks(np.arange(len(softness)+1)-0.5, minor=True)
    ax1.set_yticks(np.arange(len(texture)+1)-0.5, minor=True)
    ax1.grid(which='minor', color='white', linewidth=1)
    for i in range(len(texture)):
        for j in range(len(softness)):
            v=acc_fc[i,j]; txt='0' if v==0 else f"{v:.2f}"
            clr='black' if v>0.5 else 'white'
            ax1.text(j,i,txt,ha='center',va='center',color=clr,fontsize=12)
    plt.colorbar(im1,ax=ax1,fraction=0.046,pad=0.4/10)
    plt.tight_layout(rect=[0,0,1,0.96]); plt.show()

    # Softness × Softness
    y_true_s=[cond_map[t] for t in y_true]; y_pred_s=[cond_map[p] for p in y_pred]
    cm_s=confusion_matrix(y_true_s,y_pred_s,labels=list(range(len(softness))))
    sums_s=cm_s.sum(axis=1,keepdims=True); ratio_s=np.divide(cm_s,sums_s,where=sums_s!=0)
    masked_s=ma.masked_less(ratio_s,THRESHOLD); cmap_s=plt.cm.Blues.copy(); cmap_s.set_bad(color='white')
    fig2,ax2=plt.subplots(figsize=(7,7),dpi=150)
    im2=ax2.imshow(masked_s,cmap=cmap_s,vmin=THRESHOLD,vmax=1.0,interpolation='nearest',zorder=0)
    ax2.set_title('Softness Confusion Matrix (Proportion)',fontsize=18,pad=14)
    ax2.set_xlabel('Predicted label',fontsize=12); ax2.set_ylabel('True label',fontsize=12)
    ax2.set_xticks(np.arange(len(softness))); ax2.set_yticks(np.arange(len(softness)))
    ax2.set_xticklabels(softness,rotation=0,ha='center',fontsize=12); ax2.set_yticklabels(softness,fontsize=12)
    ax2.set_xticks(np.arange(len(softness)+1)-0.5,minor=True); ax2.set_yticks(np.arange(len(softness)+1)-0.5,minor=True)
    ax2.grid(which='minor',color='black',linewidth=1.5,zorder=3)
    for i in range(len(softness)):
        for j in range(len(softness)):
            r=ratio_s[i,j]; txt='0' if r<THRESHOLD else f"{r:.2f}"
            clr='white' if r>0.5 else 'black'
            ax2.text(j,i,txt,ha='center',va='center',color=clr,fontsize=12)
    plt.colorbar(im2,ax=ax2,fraction=0.046,pad=0.04)
    fig2.tight_layout(rect=[0,0,1,0.95]); plt.show()

    # Texture × Texture
    y_true_t=[flav_map[t] for t in y_true]; y_pred_t=[flav_map[p] for p in y_pred]
    cm_t=confusion_matrix(y_true_t,y_pred_t,labels=list(range(len(texture))))
    sums_t=cm_t.sum(axis=1,keepdims=True); ratio_t=np.divide(cm_t,sums_t,where=sums_t!=0)
    masked_t=ma.masked_less(ratio_t,THRESHOLD); cmap_t=plt.cm.Blues.copy(); cmap_t.set_bad(color='white')
    fig3,ax3=plt.subplots(figsize=(8.5,8.5),dpi=150)
    im3=ax3.imshow(masked_t,cmap=cmap_t,vmin=THRESHOLD,vmax=1.0,interpolation='nearest',zorder=0)
    ax3.set_title('Texture Confusion Matrix (Proportion)',fontsize=18,pad=14)
    ax3.set_xlabel('Predicted label',fontsize=12); ax3.set_ylabel('True label',fontsize=12)
    ax3.set_xticks(np.arange(len(texture))); ax3.set_yticks(np.arange(len(texture)))
    ax3.set_xticklabels(texture,rotation=0,ha='center',fontsize=11); ax3.set_yticklabels(texture,fontsize=11)
    ax3.set_xticks(np.arange(len(texture)+1)-0.5,minor=True); ax3.set_yticks(np.arange(len(texture)+1)-0.5,minor=True)
    ax3.set_axisbelow(False); ax3.grid(which='minor',color='black',linewidth=1.5,zorder=3)
    for i in range(len(texture)):
        for j in range(len(texture)):
            r=ratio_t[i,j]; txt='0' if r<THRESHOLD else f"{r:.2f}"
            clr='white' if r>0.5 else 'black'
            ax3.text(j,i,txt,ha='center',va='center',color=clr,fontsize=11)
    plt.colorbar(im3,ax=ax3,fraction=0.046,pad=0.04)
    fig3.tight_layout(rect=[0,0,1,0.95]); plt.show()

# ========= Extra: load and align per-class samples from _200 dataset =========
def load_and_align_extra_200(npz_path, main_classes, per_class=20, seed=RANDOM_STATE):
    if not os.path.exists(npz_path):
        print(f"[INFO] Extra dataset not found: {npz_path}, skipping.")
        return None, None
    d = np.load(npz_path, allow_pickle=True)
    X2 = d["X"]; y2 = d["y"].astype(int)

    # If _200 contains 'classes' and it's different from main_classes, remap labels
    if 'classes' in d.files and main_classes is not None:
        extra_classes = [str(x) for x in d['classes'].tolist()]
        if extra_classes != main_classes:
            mapping = {i: main_classes.index(c) for i, c in enumerate(extra_classes)}
            y2 = np.array([mapping[i] for i in y2], dtype=int)
            print("[INFO] Remapped _200 labels to match main training classes order.")
    else:
        print("[WARN] _200 dataset does not include 'classes'; assuming its label ordering matches main training set.")

    # Sample per class (up to per_class), order does not matter
    rng = np.random.default_rng(seed)
    idx_sel = []
    n_classes = len(main_classes) if main_classes is not None else int(y2.max())+1
    for c in range(n_classes):
        idx_c = np.where(y2 == c)[0]
        if len(idx_c) == 0:
            print(f"[WARN] No samples for class {c} in extra data, skipping.")
            continue
        take = min(per_class, len(idx_c))
        sel = rng.choice(idx_c, size=take, replace=False)
        idx_sel.append(sel)
    if len(idx_sel) == 0:
        print("[WARN] Sampling from _200 returned empty selection.")
        return None, None
    idx_sel = np.concatenate(idx_sel)
    X2_sel = X2[idx_sel]
    y2_sel = y2[idx_sel]
    print(f"[INFO] Sampled {len(y2_sel)} examples from _200 dataset ({per_class} per class).")
    return X2_sel, y2_sel

# ========= main pipeline =========
def main():
    # -- override global class/mapping using classes in main NPZ if present --
    global CLASS_NAMES, texture, softness, flav_map, cond_map

    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) load main training set
    data_main = np.load(NPZ_FILE_MAIN, allow_pickle=True)
    X_main = data_main["X"]; y_main = data_main["y"].astype(int)

    # if main npz has 'classes', override CLASS_NAMES and rebuild mappings
    main_classes = None
    if 'classes' in data_main.files:
        main_classes = [str(x) for x in data_main['classes'].tolist()]
        CLASS_NAMES = main_classes
        tex_order  = ['smooth','strawberry','bigberry','citrus','rough','smallberry']
        soft_order = ['ds20','ds30','ef10','ef30','ef50']
        texture  = [t for t in tex_order  if any(name.startswith(t+"_") for name in CLASS_NAMES)]
        softness = [s for s in soft_order if any((name.split('_')[1] == s) for name in CLASS_NAMES)]
        flav_map = {i: texture.index(name.split('_')[0]) for i, name in enumerate(CLASS_NAMES)}
        cond_map = {i: softness.index(name.split('_')[1]) for i, name in enumerate(CLASS_NAMES)}
        print(f"[INFO] Main training classes count: {len(CLASS_NAMES)}")

    # 2) load and merge _200 extra data (fixed per-class sample count)
    X2, y2 = load_and_align_extra_200(
        NPZ_FILE_EXTRA_200, main_classes=CLASS_NAMES,
        per_class=EXTRA_200_PER_CLASS, seed=EXTRA_200_RANDOM_STATE
    )
    if X2 is not None and y2 is not None and len(y2) > 0:
        X_all = np.concatenate([X_main, X2], axis=0)
        y_all = np.concatenate([y_main, y2], axis=0)
    else:
        X_all, y_all = X_main, y_main
    print(f"[INFO] Total training samples: {len(y_all)} (main {len(y_main)} + extra {len(y_all)-len(y_main)})")

    # 3) shape and class checks
    assert X_all.ndim==4 and X_all.shape[1:]==(T,C,D), f"Expect X.shape[1:]==({T},{C},{D}), got {X_all.shape[1:]}"
    num_classes = len(np.unique(y_all))
    print(f"[INFO] Number of classes: {num_classes}")

    # 4) normalize and reshape to sequences
    X_all = to_seq(minmax_timewise(X_all)).astype(np.float32)

    # ===== OOF training & evaluation =====
    if DO_OOF:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        y_true_all, y_pred_all = [], []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all), start=1):
            print(f"\n===== Fold {fold}/{N_SPLITS} =====")
            X_tr_full, y_tr_full = X_all[tr_idx], y_all[tr_idx]
            X_te,      y_te      = X_all[te_idx], y_all[te_idx]

            # 9:1 early-stop validation (fix: correct indexing)
            skf_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
            in_tr, in_va = next(skf_inner.split(X_tr_full, y_tr_full))
            X_tr, y_tr = X_tr_full[in_tr], y_tr_full[in_tr]
            X_va, y_va = X_tr_full[in_va], y_tr_full[in_va]

            # class weights for loss
            classes_present = np.unique(y_tr)
            cw = compute_class_weight(class_weight='balanced', classes=classes_present, y=y_tr)
            class_weights = torch.tensor(cw, dtype=torch.float32).to(device)

            # sampling weights for class-balanced sampling
            cnt = np.bincount(y_tr, minlength=len(CLASS_NAMES)).astype(np.float32)
            cnt[cnt==0] = 1.0
            w = 1.0 / cnt
            sample_weights = w[y_tr].astype(np.float64)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

            train_loader = DataLoader(AugSeqDataset(X_tr, y_tr),
                                      batch_size=BATCH_SIZE, sampler=sampler, shuffle=False,
                                      num_workers=0, pin_memory=False)
            val_loader   = DataLoader(SeqDataset(X_va, y_va),     batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=0, pin_memory=False)
            test_loader  = DataLoader(SeqDataset(X_te, y_te),     batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=0, pin_memory=False)

            encoder = LSTMEncoder(input_size=F, hidden=HIDDEN, num_layers=2, bidirectional=True,
                                  dropout=0.5, proj_dim=PROJ_DIM, num_classes=num_classes).to(device)
            print("Training encoder (BiLSTM + Attention + Max) ...")
            encoder = train_encoder(encoder, train_loader, val_loader, device, class_weights)

            trval_loader = DataLoader(SeqDataset(X_tr_full, y_tr_full), batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=0, pin_memory=False)
            X_feat_trval, y_trval = encode_features(encoder, trval_loader, device)
            X_feat_test,  y_test  = encode_features(encoder, test_loader,  device)
            print(f"Features: trval={X_feat_trval.shape}  test={X_feat_test.shape}")

            pipe = Pipeline([('scaler', StandardScaler()),
                             ('svc',    SVC(probability=True, class_weight='balanced'))])
            param_grid = {
                'svc__kernel': ['rbf'],
                'svc__C':      [3, 10, 30, 100, 300, 1000],
                'svc__gamma':  [3e-4, 1e-4, 3e-5, 1e-5, 'scale', 'auto'],
            }
            grid = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=0)
            grid.fit(X_feat_trval, y_trval)
            best = grid.best_params_
            print(f"Best SVM: {best}, CV acc={grid.best_score_:.4f}")

            preds = grid.best_estimator_.predict(X_feat_test)

            acc  = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='macro', zero_division=0)
            rec  = recall_score(y_test, preds,    average='macro', zero_division=0)
            f1   = f1_score(y_test, preds,        average='macro', zero_division=0)
            print(f"Fold {fold} — acc={acc:.4f}  prec_m={prec:.4f}  rec_m={rec:.4f}  f1_m={f1:.4f}")

            y_true_all.extend(y_test); y_pred_all.extend(preds)

        y_true_all = np.array(y_true_all); y_pred_all = np.array(y_pred_all)
        overall_acc  = accuracy_score(y_true_all, y_pred_all)
        overall_prec = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        overall_rec  = recall_score(y_true_all, y_pred_all,    average='macro', zero_division=0)
        overall_f1   = f1_score(y_true_all, y_pred_all,        average='macro', zero_division=0)
        print("\nOverall (5-fold OOF): "
              f"acc={overall_acc:.4f}  prec_m={overall_prec:.4f}  rec_m={overall_rec:.4f}  f1_m={overall_f1:.4f}")
        plot_aggregated(y_true_all, y_pred_all)

    # ===== final full-data training and save =====
    if DO_FINAL_TRAIN:
        print("\n==== Train final encoder on full data (with 9:1 val split) ====")
        skf_full = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        tr_i, va_i = next(skf_full.split(X_all, y_all))

        # class weights + sampling for final training
        classes_full = np.unique(y_all[tr_i])
        cw_full = compute_class_weight(class_weight='balanced', classes=classes_full, y=y_all[tr_i])
        class_weights_full = torch.tensor(cw_full, dtype=torch.float32).to(device)

        cnt = np.bincount(y_all[tr_i], minlength=len(CLASS_NAMES)).astype(np.float32)
        cnt[cnt==0] = 1.0
        w = 1.0 / cnt
        sample_weights = w[y_all[tr_i]].astype(np.float64)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        tr_loader = DataLoader(AugSeqDataset(X_all[tr_i], y_all[tr_i]),
                               batch_size=BATCH_SIZE, sampler=sampler, shuffle=False,
                               num_workers=0, pin_memory=False)
        va_loader = DataLoader(SeqDataset(X_all[va_i], y_all[va_i]), batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=False)

        final_encoder = LSTMEncoder(input_size=F, hidden=HIDDEN, num_layers=2, bidirectional=True,
                                    dropout=0.5, proj_dim=PROJ_DIM, num_classes=num_classes).to(device)
        final_encoder = train_encoder(final_encoder, tr_loader, va_loader, device, class_weights_full)
        torch.save(final_encoder.state_dict(), SAVE_ENCODER)
        print(f"Saved encoder -> {SAVE_ENCODER}")

        all_loader = DataLoader(SeqDataset(X_all, y_all), batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=False)
        X_feat_all, y_all_feats = encode_features(final_encoder, all_loader, device)

        final_grid = GridSearchCV(
            estimator=Pipeline([('scaler', StandardScaler()),
                                ('svc',    SVC(probability=True, class_weight='balanced'))]),
            param_grid={'svc__kernel': ['rbf'],
                        'svc__C':[3,10,30,100,300,1000],
                        'svc__gamma':[3e-4,1e-4,3e-5,1e-5,'scale','auto']},
            scoring='accuracy', cv=5, n_jobs=-1, verbose=0
        )
        final_grid.fit(X_feat_all, y_all_feats)
        joblib.dump(final_grid.best_estimator_, SAVE_SVM)
        print(f"Saved SVM -> {SAVE_SVM}, params={final_grid.best_params_}, CV acc={final_grid.best_score_:.4f}")

if __name__ == "__main__":
    main()
