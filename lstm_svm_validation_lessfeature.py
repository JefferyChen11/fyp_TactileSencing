import warnings, os, numpy as np, numpy.ma as ma, joblib, matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore",
    message="The number of unique classes is greater than 50% of the number of samples."
)

# ===== Paths (kept consistent with training) =====
# Main training set (trimmed to 250 frames)
TRAIN_NPZ_FILE_MAIN  = "data_rebuild_merge_calib_crop.npz"
# Extra _200 training set (trimmed to 250 frames) — used to build training-domain statistics for CORAL
TRAIN_NPZ_FILE_EXTRA_200 = "data_rebuild_merge_calib_crop_200.npz"

# Whether to include _200 in the "training domain" statistics for CORAL
USE_EXTRA_200_IN_TRAIN_DOMAIN = True
# Same sampling used during training: take up to 40 per class from _200
EXTRA_200_PER_CLASS    = 40
EXTRA_200_RANDOM_STATE = 42

# Validation set
VAL_NPZ_FILE   = "data_rebuild_val_merge_calib_crop.npz"

# Encoder & SVM file candidates (prefer 16x4-new names, fallback to old names)
ENCODER_FILE_CANDIDATES   = ["lstm_encoder_16x4.pt", "lstm_encoder.pt"]
SVM_MODEL_FILE_CANDIDATES = ["lstm_svm_model_16x4.joblib", "lstm_svm_model.joblib"]

# ===== Keep subset of 16 classes: texture × material =====
KEEP_TEXTURES = ['smooth','strawberry','bigberry']  # will discard citrus, rough, etc.
KEEP_SOFTNESS = ['ds20','ef50','ef30']              # will discard ds30, ef10, etc.

# ===== TTA settings (adjustable) =====
SCALES = [0.95, 1.00, 1.05]
SHIFTS = [-60, -30, 0, 30, 60]
ALPHA_WEIGHT = 5.0     # temperature for probability weighting: weight = exp(alpha * margin)

# ===== Data dimensions =====
T, C, D = 250, 8, 3
F = C * D
BATCH = 64
THRESHOLD = 1e-4
EPS = 1e-8

# (CLASS_NAMES / texture / softness will be determined after loading & filtering)
CLASS_NAMES = None
texture  = None
softness = None

# ========== Preprocessing / utilities ==========
def minmax_timewise(X):
    Xmin = X.min(axis=1, keepdims=True); Xmax = X.max(axis=1, keepdims=True)
    return (X - Xmin) / (Xmax - Xmin + EPS)

def to_seq(X):  # (N,250,8,3)->(N,250,24)
    return X.reshape(X.shape[0], T, F).astype(np.float32)

def time_warp_to_T(seq_tf, scale):
    T0, F0 = seq_tf.shape
    t = np.arange(T0, dtype=np.float32)
    q = (t * scale).clip(0, T0-1)
    out = np.empty_like(seq_tf)
    for j in range(F0):
        out[:, j] = np.interp(q, t, seq_tf[:, j])
    return out.astype(np.float32)

def time_shift_roll(seq_tf, shift):
    if shift == 0: return seq_tf
    return np.roll(seq_tf, shift=shift, axis=0)

# ========== Encoder (adaptive loading: supports with/without attention) ==========
class AttentionPool(nn.Module):
    def __init__(self, dim, attn_dim=None):
        super().__init__()
        a = attn_dim or max(64, dim // 2)
        self.fc = nn.Sequential(nn.Linear(dim, a), nn.Tanh(), nn.Linear(a, 1))
    def forward(self, y):
        a = self.fc(y); w = torch.softmax(a, dim=1)
        return (w * y).sum(dim=1), w

class LSTMEncoderAttn(nn.Module):
    def __init__(self, input_size, hidden, num_layers, bidirectional, dropout, proj_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        out_dim = hidden * (2 if bidirectional else 1)
        self.attn = AttentionPool(out_dim)
        self.ln   = nn.LayerNorm(out_dim*2)
        self.proj = nn.Linear(out_dim*2, proj_dim)
        self.head = nn.Linear(proj_dim, num_classes)
    def forward(self, x):
        y, _ = self.lstm(x)
        attn_vec, _ = self.attn(y)
        y_max, _ = y.max(dim=1)
        feat = torch.cat([attn_vec, y_max], dim=1)
        feat = self.ln(feat)
        feat = torch.relu(self.proj(feat))
        logits = self.head(feat)
        return logits, feat

class LSTMEncoderNoAttn(nn.Module):
    def __init__(self, input_size, hidden, num_layers, bidirectional, dropout, proj_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        out_dim = hidden * (2 if bidirectional else 1)
        self.ln   = nn.LayerNorm(out_dim*2)
        self.proj = nn.Linear(out_dim*2, proj_dim)
        self.head = nn.Linear(proj_dim, num_classes)
    def forward(self, x):
        y, _ = self.lstm(x)
        y_mean = y.mean(dim=1); y_max, _ = y.max(dim=1)
        feat = torch.cat([y_mean, y_max], dim=1)
        feat = self.ln(feat)
        feat = torch.relu(self.proj(feat))
        logits = self.head(feat)
        return logits, feat

def build_encoder_from_ckpt(state_dict, input_size=F, device='cpu', num_classes=16):
    lstm_w = [k for k in state_dict if k.startswith("lstm.weight_ih_l")]
    assert lstm_w, "No LSTM weights found in state_dict"
    layers, bidir = set(), False
    for k in lstm_w:
        s = k.split("lstm.weight_ih_l")[-1]
        if s.endswith("_reverse"):
            layers.add(int(s[:-8])); bidir = True
        else:
            layers.add(int(s))
    num_layers = max(layers) + 1
    hidden = state_dict["lstm.weight_ih_l0"].shape[0] // 4
    proj_dim = state_dict["proj.weight"].shape[0] if "proj.weight" in state_dict \
               else state_dict["head.weight"].shape[1]
    has_attn = any(k.startswith("attn.") for k in state_dict)
    dropout = 0.5
    enc = (LSTMEncoderAttn if has_attn else LSTMEncoderNoAttn)(
        input_size, hidden, num_layers, bidir, dropout, proj_dim, num_classes
    ).to(device)
    try: enc.load_state_dict(state_dict, strict=True)
    except Exception: enc.load_state_dict(state_dict, strict=False)
    enc.eval()
    return enc

@torch.no_grad()
def encode_array(encoder, X, device, batch=BATCH):
    """X:(N,T,F) -> feats:(N,proj_dim)"""
    N = X.shape[0]; out = []
    for i in range(0, N, batch):
        xb = torch.from_numpy(X[i:i+batch]).to(device)
        _, f = encoder(xb)
        out.append(f.cpu().numpy())
    return np.vstack(out)

# ========== SVM scores & margin helpers ==========
def pipeline_scores_and_margin(pipe, X, num_classes):
    # Prefer decision_function if available
    try:
        s = pipe.decision_function(X)
        if s.ndim == 1: s = s[:, None]
        top2 = np.partition(s, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        e = np.exp(s - s.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return np.argmax(s, axis=1), margin, p
    except Exception:
        pass
    # Fall back to predict_proba
    try:
        p = pipe.predict_proba(X)
        top2 = np.partition(p, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        return np.argmax(p, axis=1), margin, p
    except Exception:
        pass
    # Last resort: predict only
    yhat = pipe.predict(X)
    return yhat, np.zeros_like(yhat, dtype=float), None

# ========== CORAL: align validation features to training domain ==========
def _eig_psd_sqrt_invsqrt(C, eps=1e-6):
    w, v = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    sqrtW = np.sqrt(w); invsqrtW = 1.0 / sqrtW
    C_sqrt = (v * sqrtW) @ v.T
    C_invsqrt = (v * invsqrtW) @ v.T
    return C_sqrt, C_invsqrt

def coral_fit(source_feats, target_feats):
    mu_s = source_feats.mean(axis=0)
    mu_t = target_feats.mean(axis=0)
    Cs = np.cov((source_feats - mu_s).T) + np.eye(source_feats.shape[1]) * 1e-6
    Ct = np.cov((target_feats - mu_t).T) + np.eye(target_feats.shape[1]) * 1e-6
    Cs_s, _ = _eig_psd_sqrt_invsqrt(Cs)
    _, Ct_is = _eig_psd_sqrt_invsqrt(Ct)
    M = Ct_is @ Cs_s
    return mu_s.astype(np.float64), mu_t.astype(np.float64), M.astype(np.float64)

def coral_apply(x, mu_s, mu_t, M):
    return ((x.astype(np.float64) - mu_t) @ M + mu_s).astype(np.float32)

# ========= Subset helpers: filter + remap ==========
def build_allowed_classes(classes, keep_textures, keep_softness):
    """Filter classes (list of names) to those matching both texture and softness; return allowed list (original order)"""
    allowed = [name for name in classes
               if (name.split('_')[0] in keep_textures) and (name.split('_')[1] in keep_softness)]
    if len(allowed) == 0:
        raise RuntimeError("Allowed set is empty: check KEEP_TEXTURES / KEEP_SOFTNESS and the dataset classes.")
    name_to_new = {name: i for i, name in enumerate(allowed)}
    return allowed, name_to_new

def filter_dataset_to_allowed_with_names(X, y, classes, allowed_names, name_to_new):
    """Filter + remap by class names (robust approach)"""
    cls_arr = np.array(classes, dtype=object)
    keep_mask = np.array([(cls_arr[yy] in allowed_names) for yy in y], dtype=bool)
    X_new = X[keep_mask]
    y_new = np.array([name_to_new[cls_arr[yy]] for yy in y[keep_mask]], dtype=int)
    return X_new, y_new, keep_mask

def filter_dataset_to_allowed_assume_same_order(X, y, train_allowed_names, train_all_classes):
    """
    When VAL NPZ lacks 'classes', assume labels follow the same order as train_all_classes:
    - find old indices in train_all_classes that correspond to train_allowed_names
    - keep samples with those old indices and map old_idx -> new [0..len-1]
    """
    old_idx_list = [train_all_classes.index(n) for n in train_allowed_names]
    old_to_new = {old: new for new, old in enumerate(old_idx_list)}
    keep_mask = np.isin(y, old_idx_list)
    X_new = X[keep_mask]
    y_new = np.array([old_to_new[int(yy)] for yy in y[keep_mask]], dtype=int)
    if not np.all(keep_mask):
        miss = np.unique(y[~keep_mask])
        print(f"[WARN] Validation set contains old label indices not in the 16-class subset: {miss.tolist()} (they were removed)."
              " Because 'classes' is missing, we assume label order matches training; if not, regenerate validation npz with 'classes'.")
    return X_new, y_new, keep_mask

def sample_per_class(X, y, per_class, seed=42, num_classes=None):
    """Subsample up to per_class examples per class; return subsampled X,y."""
    rng = np.random.default_rng(seed)
    if num_classes is None:
        num_classes = int(y.max()) + 1
    idx_all = []
    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        if len(idx_c) == 0:
            continue
        take = min(per_class, len(idx_c))
        sel = rng.choice(idx_c, size=take, replace=False)
        idx_all.append(sel)
    if len(idx_all) == 0:
        return X[:0], y[:0]
    idx_all = np.concatenate(idx_all)
    return X[idx_all], y[idx_all]

# ========== Plotting (keeps original visuals) ==========
def plot_aggregated(y_true, y_pred, CLASS_NAMES, texture, softness, title_prefix):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    row_sum = cm.sum(axis=1, keepdims=True)
    ratio   = np.divide(cm, row_sum, where=row_sum!=0)
    masked  = ma.masked_less(ratio, THRESHOLD)
    cmap0   = plt.cm.Blues.copy(); cmap0.set_bad(color='white')

    # class -> texture / softness index
    local_flav_map = {i: texture.index(n.split('_')[0]) for i, n in enumerate(CLASS_NAMES)}
    local_cond_map = {i: softness.index(n.split('_')[1]) for i, n in enumerate(CLASS_NAMES)}

    correct = np.zeros((len(texture), len(softness)), dtype=int)
    total   = np.zeros_like(correct)
    for t, p in zip(y_true, y_pred):
        i, j = local_flav_map[t], local_cond_map[t]
        total[i, j] += 1
        if t == p: correct[i, j] += 1
    acc_fc = np.divide(correct, total, where=total!=0); acc_fc = np.nan_to_num(acc_fc)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16,6), dpi=150)
    fig.suptitle(title_prefix, fontsize=20)
    im0 = ax0.imshow(masked, cmap=cmap0, vmin=THRESHOLD, vmax=1.0, interpolation='nearest')
    n = len(CLASS_NAMES)
    ax0.set_title('Confusion Matrix (Proportion)', fontsize=16)
    ax0.set_xticks(np.arange(n)); ax0.set_yticks(np.arange(n))
    ax0.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=6)
    ax0.set_yticklabels(CLASS_NAMES, fontsize=6)
    ax0.set_xticks(np.arange(n+1)-0.5, minor=True); ax0.set_yticks(np.arange(n+1)-0.5, minor=True)
    ax0.grid(which='minor', color='white', linewidth=1)
    for spine in ax0.spines.values(): spine.set_visible(False)
    for i in range(n):
        for j in range(n):
            r = ratio[i, j]; txt = '0' if r < THRESHOLD else f"{r:.2f}"
            clr = 'white' if r > 0.5 else 'black'
            ax0.text(j, i, txt, ha='center', va='center', color=clr, fontsize=6)
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.4/10)

    im1 = ax1.imshow(acc_fc, cmap='cividis', vmin=0.0, vmax=1.0, aspect='equal')
    ax1.set_title('Flavor×Condition Accuracy', fontsize=16)
    ax1.set_xticks(np.arange(len(softness))); ax1.set_yticks(np.arange(len(texture)))
    ax1.set_xticklabels(softness, rotation=45, ha='right', fontsize=12)
    ax1.set_yticklabels(texture, fontsize=12)
    ax1.set_xticks(np.arange(len(softness)+1)-0.5, minor=True)
    ax1.set_yticks(np.arange(len(texture)+1)-0.5, minor=True)
    ax1.grid(which='minor', color='white', linewidth=1)
    for i in range(len(texture)):
        for j in range(len(softness)):
            v = acc_fc[i, j]; txt = '0' if v == 0 else f"{v:.2f}"
            clr = 'black' if v > 0.5 else 'white'
            ax1.text(j, i, txt, ha='center', va='center', color=clr, fontsize=12)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.4/10)
    plt.tight_layout(rect=[0,0,1,0.96]); plt.show()

    # Softness × Softness
    y_true_s=[softness.index(CLASS_NAMES[t].split('_')[1]) for t in y_true]
    y_pred_s=[softness.index(CLASS_NAMES[p].split('_')[1]) for p in y_pred]
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
    ax2.set_axisbelow(False); ax2.grid(which='minor',color='black',linewidth=1.5,zorder=3)
    for i in range(len(softness)):
        for j in range(len(softness)):
            r=ratio_s[i,j]; txt='0' if r<THRESHOLD else f"{r:.2f}"
            clr='white' if r>0.5 else 'black'
            ax2.text(j,i,txt,ha='center',va='center',color=clr,fontsize=12)
    plt.colorbar(im2,ax=ax2,fraction=0.046,pad=0.04)
    fig2.tight_layout(rect=[0,0,1,0.95]); plt.show()

    # Texture × Texture
    y_true_t=[texture.index(CLASS_NAMES[t].split('_')[0]) for t in y_true]
    y_pred_t=[texture.index(CLASS_NAMES[p].split('_')[0]) for p in y_pred]
    cm_t=confusion_matrix(y_true_t,y_pred_t,labels=list(range(len(texture))))
    sums_t=cm_t.sum(axis=1,keepdims=True); ratio_t=np.divide(cm_t,sums_t,where=sums_t!=0)
    masked_t=ma.masked_less(ratio_t,THRESHOLD); cmap_t=plt.cm.Blues.copy(); cmap_t.set_bad(color='white')
    fig3,ax3=plt.subplots(figsize=(8.5,8.5),dpi=150)
    im3=ax3.imshow(masked_t,cmap=cmap_t,vmin=THRESHOLD,vmax=1.0,interpolation='nearest',zorder=0)
    ax3.set_title('Texture Confusion Matrix (Proportion)',fontsize=18,pad=14)
    ax3.set_xlabel('Predicted label',fontsize=12); ax3.set_ylabel('True label',fontsize=12)
    ax3.set_xticks(np.arange(len(texture))); ax3.set_yticks(np.arange(len(texture)))
    ax3.set_xticklabels(texture,rotation=0,ha='center',fontsize=12); ax3.set_yticklabels(texture,fontsize=12)
    ax3.set_xticks(np.arange(len(texture)+1)-0.5,minor=True); ax3.set_yticks(np.arange(len(texture)+1)-0.5,minor=True)
    ax3.set_axisbelow(False); ax3.grid(which='minor',color='black',linewidth=1.5,zorder=3)
    for i in range(len(texture)):
        for j in range(len(texture)):
            r=ratio_t[i,j]; txt='0' if r<THRESHOLD else f"{r:.2f}"
            clr='white' if r>0.5 else 'black'
            ax3.text(j,i,txt,ha='center',va='center',color=clr,fontsize=12)
    plt.colorbar(im3,ax=ax3,fraction=0.046,pad=0.04)
    fig3.tight_layout(rect=[0,0,1,0.95]); plt.show()

# ========== Main pipeline ==========
def main():
    # Make MKLDNN off for Windows CPU stability if applicable
    try: torch.backends.mkldnn.enabled = False
    except Exception: pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # === Load main training NPZ (required; must contain 'classes') ===
    tr_main = np.load(TRAIN_NPZ_FILE_MAIN, allow_pickle=True)
    Xtrm_raw = tr_main["X"]; ytrm_raw = tr_main["y"].astype(int)
    classes_main = [str(x) for x in tr_main["classes"].tolist()] if 'classes' in tr_main.files else None
    if classes_main is None:
        raise RuntimeError("Main training NPZ does not contain 'classes'. Please re-export the training NPZ with classes included.")

    # === Optionally load _200 (used for CORAL training-domain stats) ===
    Xtre_raw = None; ytre_raw = None; classes_extra = None
    if USE_EXTRA_200_IN_TRAIN_DOMAIN and os.path.exists(TRAIN_NPZ_FILE_EXTRA_200):
        tr_extra = np.load(TRAIN_NPZ_FILE_EXTRA_200, allow_pickle=True)
        Xtre_raw = tr_extra["X"]; ytre_raw = tr_extra["y"].astype(int)
        if 'classes' in tr_extra.files:
            classes_extra = [str(x) for x in tr_extra["classes"].tolist()]
        else:
            print("[WARN] _200 NPZ does not contain 'classes'; assuming its label order matches the main training NPZ. If not, regenerate _200 with 'classes'.")
    else:
        if USE_EXTRA_200_IN_TRAIN_DOMAIN:
            print(f"[WARN] _200 NPZ not found: {TRAIN_NPZ_FILE_EXTRA_200}. Using main training set only for CORAL domain stats.")

    # === Build allowed 16-class set ===
    allowed_names, name_to_new = build_allowed_classes(classes_main, KEEP_TEXTURES, KEEP_SOFTNESS)
    print(f"[INFO] Allowed classes ({len(allowed_names)}): {allowed_names}")

    # === Filter & remap main training set to allowed classes ===
    Xtrm_f, ytrm_f, _ = filter_dataset_to_allowed_with_names(
        Xtrm_raw, ytrm_raw, classes_main, allowed_names, name_to_new
    )
    if Xtrm_f.shape[0] == 0:
        raise RuntimeError("No samples remain in main training set after filtering. Check KEEP_* settings and training classes.")

    # === If _200 exists: filter, remap, and subsample per-class ===
    Xtre_sel = None; ytre_sel = None
    if Xtre_raw is not None:
        if classes_extra is not None:
            Xtre_f, ytre_f, _ = filter_dataset_to_allowed_with_names(
                Xtre_raw, ytre_raw, classes_extra, allowed_names, name_to_new
            )
        else:
            # No classes in _200: assume its label order matches main training classes
            Xtre_f, ytre_f, _ = filter_dataset_to_allowed_assume_same_order(
                Xtre_raw, ytre_raw, train_allowed_names=allowed_names, train_all_classes=classes_main
            )
        # Subsample up to EXTRA_200_PER_CLASS per class
        Xtre_sel, ytre_sel = sample_per_class(Xtre_f, ytre_f,
                                              per_class=EXTRA_200_PER_CLASS,
                                              seed=EXTRA_200_RANDOM_STATE,
                                              num_classes=len(allowed_names))

    # === Build training-domain for CORAL: main filtered (+ optional _200 subsample) ===
    if Xtre_sel is not None and Xtre_sel.shape[0] > 0:
        Xtr_all = np.concatenate([Xtrm_f, Xtre_sel], axis=0)
        ytr_all = np.concatenate([ytrm_f, ytre_sel], axis=0)
        print(f"[INFO] CORAL training domain: main {Xtrm_f.shape[0]} + _200 sampled {Xtre_sel.shape[0]} = {Xtr_all.shape[0]}")
    else:
        Xtr_all, ytr_all = Xtrm_f, ytrm_f
        print(f"[INFO] CORAL training domain: main only {Xtr_all.shape[0]} (no _200 included)")

    # === Validation set: load, filter & remap ===
    val = np.load(VAL_NPZ_FILE, allow_pickle=True)
    Xv_raw = val["X"]; yv_raw = val["y"].astype(int)
    classes_val = [str(x) for x in val["classes"].tolist()] if 'classes' in val.files else None
    if classes_val is not None:
        Xv_f, yv_f, _ = filter_dataset_to_allowed_with_names(
            Xv_raw, yv_raw, classes_val, allowed_names, name_to_new
        )
    else:
        print("[WARN] Validation NPZ does not contain 'classes'. Assuming validation label order matches training; filtering/remapping accordingly. If not correct, regenerate validation NPZ with 'classes'.")
        Xv_f, yv_f, _ = filter_dataset_to_allowed_assume_same_order(
            Xv_raw, yv_raw, train_allowed_names=allowed_names, train_all_classes=classes_main
        )
    if Xv_f.shape[0] == 0:
        raise RuntimeError("Validation set contains no samples from the 16-class subset. Check validation data or KEEP_* settings.")

    # === Final class names and visualization groups ===
    global CLASS_NAMES, texture, softness
    CLASS_NAMES = allowed_names
    texture  = KEEP_TEXTURES.copy()
    softness = KEEP_SOFTNESS.copy()
    NUM_CLASSES = len(CLASS_NAMES)

    # === Normalize + reshape to sequences ===
    Xtr = to_seq(minmax_timewise(Xtr_all))  # (N_tr,250,24)
    Xv  = to_seq(minmax_timewise(Xv_f))     # (N_v ,250,24)
    yv  = yv_f

    print(f"Train (filtered for CORAL): {Xtr.shape}  (classes={NUM_CLASSES})")
    print(f"Val (filtered):              {Xv.shape}")

    # === Load encoder (auto-adapt attention & dims) ===
    enc_path = next((p for p in ENCODER_FILE_CANDIDATES if os.path.exists(p)), None)
    if enc_path is None:
        raise FileNotFoundError(f"Encoder file not found: {ENCODER_FILE_CANDIDATES}")
    print(f"[INFO] Using encoder: {enc_path}")
    state = torch.load(enc_path, map_location=device)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    encoder = build_encoder_from_ckpt(state, input_size=F, device=device, num_classes=NUM_CLASSES)

    # === Compute training-domain features (one-time) and validation baseline features ===
    Feat_tr = encode_array(encoder, Xtr, device, batch=BATCH)  # (N_tr, Df)
    Feat_v_base = encode_array(encoder, Xv, device, batch=BATCH)
    mu_s, mu_t, M = coral_fit(Feat_tr, Feat_v_base)  # fit once

    # === Load SVM pipeline ===
    svm_path = next((p for p in SVM_MODEL_FILE_CANDIDATES if os.path.exists(p)), None)
    if svm_path is None:
        raise FileNotFoundError(f"SVM model file not found: {SVM_MODEL_FILE_CANDIDATES}")
    print(f"[INFO] Using SVM model: {svm_path}")
    pipe = joblib.load(svm_path)

    # === Selective TTA + probability-weighted fusion + CORAL alignment ===
    N = Xv.shape[0]
    y_pred = np.zeros(N, dtype=int)
    for i in range(N):
        base = Xv[i]  # (250,24)
        prob_sum = np.zeros((NUM_CLASSES,), dtype=np.float64)

        for s in SCALES:
            warped = time_warp_to_T(base, s)
            for sh in SHIFTS:
                seg = time_shift_roll(warped, sh)
                f   = encode_array(encoder, seg[None,...], device, batch=1)  # (1,Df)
                f   = coral_apply(f, mu_s, mu_t, M)                          # CORAL align
                yhat, margin, p = pipeline_scores_and_margin(pipe, f, num_classes=NUM_CLASSES)
                if p is None:
                    p = np.zeros((1, NUM_CLASSES), dtype=np.float64); p[0, yhat[0]] = 1.0
                    w = 1.0
                else:
                    w = float(np.exp(ALPHA_WEIGHT * margin[0]))
                prob_sum += w * p[0].astype(np.float64)

        y_pred[i] = int(np.argmax(prob_sum))
        if (i+1) % 50 == 0 or i+1 == N:
            print(f"\rTTA+CORAL infer {i+1}/{N}", end="")
    print()

    # === Metrics + plots ===
    acc  = accuracy_score(yv, y_pred)
    prec = precision_score(yv, y_pred, average='macro', zero_division=0)
    rec  = recall_score(yv, y_pred,    average='macro', zero_division=0)
    f1   = f1_score(yv, y_pred,        average='macro', zero_division=0)
    print("\nValidation on 16 classes (TTA prob-fusion + CORAL) Metrics:")
    print(f"accuracy: {acc:.4f}  precision_macro: {prec:.4f}  recall_macro: {rec:.4f}  f1_macro: {f1:.4f}")

    plot_aggregated(yv, y_pred, CLASS_NAMES, texture, softness,
                    title_prefix="Validation (16-class, LSTM→SVM, TTA prob-fusion + CORAL)")

if __name__ == "__main__":
    main()
