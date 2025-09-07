import warnings, numpy as np, numpy.ma as ma, joblib, matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore",
    message="The number of unique classes is greater than 50% of the number of samples."
)

# ===== Paths (kept consistent with training) =====
TRAIN_MAIN_NPZ_FILE  = "data_rebuild_merge_calib_crop.npz"          # main training set (3000)
TRAIN_EXTRA_NPZ_FILE = "data_rebuild_merge_calib_crop_200.npz"      # extra _200 (sample per-class from here)
VAL_NPZ_FILE         = "data_rebuild_val_merge_calib_crop.npz"      # validation set (1500)
ENCODER_FILE         = "lstm_encoder.pt"                           # trained encoder file
SVM_MODEL_FILE       = "lstm_svm_model.joblib"                     # trained SVM pipeline

# ===== _200 sampling configuration (kept same as training: per-class sampling) =====
EXTRA_200_PER_CLASS    = 40
EXTRA_200_RANDOM_STATE = 42

# ===== TTA settings (increase for potentially higher accuracy) =====
SCALES = [0.95, 1.00, 1.05]
SHIFTS = [-60, -30, 0, 30, 60]
ALPHA_WEIGHT = 5.0     # temperature for probability weighting: weight = exp(alpha * margin)

# ===== Data dimensions =====
T, C, D = 250, 8, 3
F = C * D
BATCH = 64
THRESHOLD = 1e-4
EPS = 1e-8

# ===== Class names (will be overridden if training NPZ contains 'classes') =====
CLASS_NAMES = [
    'bigberry_ds20','citrus_ds20','rough_ds20','smallberry_ds20','smooth_ds20','strawberry_ds20',
    'bigberry_ds30','citrus_ds30','rough_ds30','smallberry_ds30','smooth_ds30','strawberry_ds30',
    'bigberry_ef10','citrus_ef10','rough_ef10','smallberry_ef10','smooth_ef10','strawberry_ef10',
    'bigberry_ef30','citrus_ef30','rough_ef30','smallberry_ef30','smooth_ef30','strawberry_ef30',
    'bigberry_ef50','citrus_ef50','rough_ef50','smallberry_ef50','smooth_ef50','strawberry_ef50'
]
texture  = ['smooth','strawberry','bigberry','citrus','rough','smallberry']
softness = ['ds20','ds30','ef10','ef30','ef50']
flav_map = {i: texture.index(n.split('_')[0]) for i, n in enumerate(CLASS_NAMES)}
cond_map = {i: softness.index(n.split('_')[1]) for i, n in enumerate(CLASS_NAMES)}

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

def build_encoder_from_ckpt(state_dict, input_size=F, num_classes=None, device='cpu'):
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
    if num_classes is None:
        # try to infer from head.weight
        if "head.weight" in state_dict:
            num_classes = state_dict["head.weight"].shape[0]
        else:
            num_classes = 30
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
        # convert decision function to "pseudo-probabilities" via softmax for smoother fusion
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
    """C is symmetric PSD; return (C^1/2, C^-1/2)."""
    w, v = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    sqrtW = np.sqrt(w); invsqrtW = 1.0 / sqrtW
    C_sqrt = (v * sqrtW) @ v.T
    C_invsqrt = (v * invsqrtW) @ v.T
    return C_sqrt, C_invsqrt

def coral_fit(source_feats, target_feats):
    """Construct a one-shot linear alignment: X_t' = (X_t - mu_t) Ct^-1/2 Cs^1/2 + mu_s"""
    mu_s = source_feats.mean(axis=0)
    mu_t = target_feats.mean(axis=0)
    Cs = np.cov((source_feats - mu_s).T) + np.eye(source_feats.shape[1]) * 1e-6
    Ct = np.cov((target_feats - mu_t).T) + np.eye(target_feats.shape[1]) * 1e-6
    Cs_s, _ = _eig_psd_sqrt_invsqrt(Cs)
    _, Ct_is = _eig_psd_sqrt_invsqrt(Ct)
    M = Ct_is @ Cs_s   # right-multiply by sqrt(Cs), left-multiply by invsqrt(Ct)
    return mu_s.astype(np.float64), mu_t.astype(np.float64), M.astype(np.float64)

def coral_apply(x, mu_s, mu_t, M):
    # x:(*,D)  ->  (x - mu_t) @ M + mu_s
    return ((x.astype(np.float64) - mu_t) @ M + mu_s).astype(np.float32)

# ========== Plotting (matches original visual layout) ==========
def plot_aggregated(y_true, y_pred, title_prefix):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    row_sum = cm.sum(axis=1, keepdims=True)
    ratio   = np.divide(cm, row_sum, where=row_sum!=0)
    masked  = ma.masked_less(ratio, THRESHOLD)
    cmap0   = plt.cm.Blues.copy(); cmap0.set_bad(color='white')

    correct = np.zeros((len(texture), len(softness)), dtype=int)
    total   = np.zeros_like(correct)
    for t, p in zip(y_true, y_pred):
        i, j = flav_map[t], cond_map[t]
        total[i, j] += 1
        if t == p: correct[i, j] += 1
    acc_fc = np.divide(correct, total, where=total!=0); acc_fc = np.nan_to_num(acc_fc)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16,6), dpi=150)
    fig.suptitle(title_prefix, fontsize=20)
    im0 = ax0.imshow(masked, cmap=cmap0, vmin=THRESHOLD, vmax=1.0, interpolation='nearest')
    n = len(CLASS_NAMES)
    ax0.set_title('Confusion Matrix (Proportion)', fontsize=16)
    ax0.set_xticks(np.arange(n)); ax0.set_yticks(np.arange(n))
    ax0.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=3)
    ax0.set_yticklabels(CLASS_NAMES, fontsize=3)
    ax0.set_xticks(np.arange(n+1)-0.5, minor=True); ax0.set_yticks(np.arange(n+1)-0.5, minor=True)
    ax0.grid(which='minor', color='white', linewidth=1)
    for spine in ax0.spines.values(): spine.set_visible(False)
    for i in range(n):
        for j in range(n):
            r = ratio[i, j]; txt = '0' if r < THRESHOLD else f"{r:.2f}"
            clr = 'white' if r > 0.5 else 'black'
            ax0.text(j, i, txt, ha='center', va='center', color=clr, fontsize=3)
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
    ax2.set_axisbelow(False); ax2.grid(which='minor',color='black',linewidth=1.5,zorder=3)
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

# ========== Read training classes and align _200 labels ==========
def try_get_classes(npz_obj):
    if 'classes' in npz_obj.files:
        cls = [str(x) for x in npz_obj['classes'].tolist()]
        return cls
    return None

def remap_labels(y_old, from_classes, to_classes):
    """Map labels indexed by from_classes to indices of to_classes (by name match)."""
    mapping = {i: to_classes.index(c) for i, c in enumerate(from_classes)}
    return np.array([mapping[i] for i in y_old], dtype=int)

def load_and_align_extra_200_for_domain(npz_path, main_classes, per_class=20, seed=EXTRA_200_RANDOM_STATE):
    """Load _200 npz, remap labels if necessary, then sample per_class examples per class."""
    if npz_path is None or not npz_path or not os.path.exists(npz_path):
        print("[INFO] Extra _200 NPZ not found, skipping."); return None, None
    d = np.load(npz_path, allow_pickle=True)
    X2 = d["X"]; y2 = d["y"].astype(int)
    if 'classes' in d.files and main_classes is not None:
        extra_classes = [str(x) for x in d['classes'].tolist()]
        if extra_classes != main_classes:
            y2 = remap_labels(y2, extra_classes, main_classes)
            print("[INFO] Remapped _200 labels to match main training classes order.")
    # per-class sampling
    rng = np.random.default_rng(seed)
    idx_sel = []
    n_classes = len(main_classes)
    for c in range(n_classes):
        idx_c = np.where(y2 == c)[0]
        if len(idx_c) == 0:
            print(f"[WARN] _200 has no samples for class {c}, skipping.")
            continue
        take = min(per_class, len(idx_c))
        sel = rng.choice(idx_c, size=take, replace=False)
        idx_sel.append(sel)
    if len(idx_sel) == 0:
        print("[WARN] Sampling from _200 returned empty selection."); return None, None
    idx_sel = np.concatenate(idx_sel)
    return X2[idx_sel], y2[idx_sel]

# ========== Main pipeline ==========
import os

def main():
    global CLASS_NAMES, texture, softness, flav_map, cond_map

    # Make MKLDNN off for Windows CPU stability (if applicable)
    try: torch.backends.mkldnn.enabled = False
    except Exception: pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Load training-domain composition (main + sampled _200)
    tr_main = np.load(TRAIN_MAIN_NPZ_FILE, allow_pickle=True)
    Xtr_main = tr_main["X"]; ytr_main = tr_main["y"].astype(int)
    classes_main = try_get_classes(tr_main)
    if classes_main is not None:
        CLASS_NAMES = classes_main
        # rebuild flavor/softness lists and maps
        tex_order  = ['smooth','strawberry','bigberry','citrus','rough','smallberry']
        soft_order = ['ds20','ds30','ef10','ef30','ef50']
        texture  = [t for t in tex_order  if any(name.startswith(t+"_") for name in CLASS_NAMES)]
        softness = [s for s in soft_order if any((name.split('_')[1] == s) for name in CLASS_NAMES)]
        flav_map = {i: texture.index(name.split('_')[0]) for i, name in enumerate(CLASS_NAMES)}
        cond_map = {i: softness.index(name.split('_')[1]) for i, name in enumerate(CLASS_NAMES)}

    # sample EXTRA_200_PER_CLASS from _200
    Xtr_extra, ytr_extra = load_and_align_extra_200_for_domain(
        TRAIN_EXTRA_NPZ_FILE, main_classes=CLASS_NAMES,
        per_class=EXTRA_200_PER_CLASS, seed=EXTRA_200_RANDOM_STATE
    )
    if Xtr_extra is not None:
        Xtr_all = np.concatenate([Xtr_main, Xtr_extra], axis=0)
        ytr_all = np.concatenate([ytr_main, ytr_extra], axis=0)
    else:
        Xtr_all, ytr_all = Xtr_main, ytr_main

    print(f"Train domain: main={len(ytr_main)} + extra_subset={0 if ytr_extra is None else len(ytr_extra)} -> total={len(ytr_all)}")
    NUM_CLASSES = len(CLASS_NAMES)

    # unify normalization + reshape to sequences
    Xtr_all_tf = to_seq(minmax_timewise(Xtr_all))  # (Ntr, 250, 24)

    # 2) Load validation set
    val = np.load(VAL_NPZ_FILE)
    Xv  = to_seq(minmax_timewise(val["X"]))  # (Nva, 250, 24)
    yv  = val["y"].astype(int)
    print(f"Val: {Xv.shape[0]} samples")

    # 3) Load encoder
    state = torch.load(ENCODER_FILE, map_location=device)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    encoder = build_encoder_from_ckpt(state, input_size=F, num_classes=NUM_CLASSES, device=device)

    # 4) Compute training-domain features (one-time)
    Feat_tr = encode_array(encoder, Xtr_all_tf, device, batch=BATCH)  # (N_tr, Df)

    # 5) Encode validation base features (no TTA) to estimate target statistics for CORAL
    Feat_v_base = encode_array(encoder, Xv, device, batch=BATCH)
    mu_s, mu_t, M = coral_fit(Feat_tr, Feat_v_base)  # fit once (source=training domain, target=validation domain)

    # 6) Load SVM pipeline
    pipe = joblib.load(SVM_MODEL_FILE)

    # 7) Selective TTA + probability-weighted fusion + CORAL alignment
    N = Xv.shape[0]
    y_pred = np.zeros(N, dtype=int)
    for i in range(N):
        base = Xv[i]  # (250,24)
        prob_sum = np.zeros((NUM_CLASSES,), dtype=np.float64)

        for s in SCALES:
            warped = time_warp_to_T(base, s)
            for sh in SHIFTS:
                seg = time_shift_roll(warped, sh)              # (250,24)
                f   = encode_array(encoder, seg[None,...], device, batch=1)  # (1,Df)
                f   = coral_apply(f, mu_s, mu_t, M)            # CORAL align to training domain
                yhat, margin, p = pipeline_scores_and_margin(pipe, f, num_classes=NUM_CLASSES)
                if p is None:  # no probabilities available -> fallback to one-hot
                    p = np.zeros((1, NUM_CLASSES), dtype=np.float64)
                    p[0, yhat[0]] = 1.0
                    w = 1.0
                else:
                    w = float(np.exp(ALPHA_WEIGHT * margin[0]))  # confidence weighting
                prob_sum += w * p[0].astype(np.float64)

        y_pred[i] = int(np.argmax(prob_sum))
        if (i+1) % 50 == 0 or i+1 == N:
            print(f"\rTTA+CORAL infer {i+1}/{N}", end="")
    print()

    # 8) Metrics + visualization
    acc  = accuracy_score(yv, y_pred)
    prec = precision_score(yv, y_pred, average='macro', zero_division=0)
    rec  = recall_score(yv, y_pred,    average='macro', zero_division=0)
    f1   = f1_score(yv, y_pred,        average='macro', zero_division=0)
    print("\nValidation (TTA prob-fusion + CORAL) Metrics:")
    print(f"accuracy: {acc:.4f}  precision_macro: {prec:.4f}  recall_macro: {rec:.4f}  f1_macro: {f1:.4f}")

    plot_aggregated(yv, y_pred, title_prefix="Validation (LSTM→SVM, TTA prob-fusion + CORAL)")

if __name__ == "__main__":
    main()
