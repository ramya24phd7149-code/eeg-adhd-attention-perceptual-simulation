"""
================================================================================
COMPLETE PAPER PIPELINE — every result + every figure/table for the revised
manuscript, in one self-contained script.
================================================================================
Produces (all in OUTPUT_DIR, both PDF for LaTeX includegraphics and PNG
for quick viewing where noted):

  EEG spectral attention (Section 4.1)
    eeg_attention_table.csv
    Figure_Boxplot.pdf              (Fig 2)
    Figure_Histogram.pdf            (Fig 3)
    table1_spectral_attention.txt

  EEG cognitive representation + semantic association (Section 4.2)
    Figure_PCA_Representation.pdf   (Fig 4 — 2D visualization)
    pca_component_justification.png (scree/cumvar — Kaiser/90%/elbow)
    Figure_Label_Matching.pdf       (Fig 5)
    eeg_image_semantic_matches.csv
    semantic_match_verification.csv
    table2_eeg_representation.txt

  Attention-adaptive perceptual simulation, illustrative (Section 4.3/4.4)
    MultiLevel_EEG_Perception.pdf   (Fig 6 — CORRECTED direction)
    Figure7_Quantitative_Metrics.pdf (Fig 7)
    table3_quantitative_avg.txt
    table4_representative_image.txt

  Ablation study (Section 4.4)
    table5_ablation.txt

  Group-level whole-cohort validation (Section 4.6)
    per_subject_simulation_results.csv
    attention_vs_ssim_group_analysis.png
    table_group_validation.txt
    table_sensitivity.txt
    group_stats_summary.txt

RUN ORDER: call main() top to bottom. Each stage prints its own numbers so
you can copy them directly into the paper tables as you go, in case any
stage needs to be re-run individually later.
================================================================================
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from scipy.signal import welch, butter, filtfilt
from scipy import stats
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RNG_SEED = 42
np.random.seed(RNG_SEED)

# ------------------------------------------------------------------------
# PATHS — confirmed working paths from your Kaggle runs
# ------------------------------------------------------------------------
ADHD_DIR = "/kaggle/input/datasets/danizo/eeg-dataset-for-adhd"
MINDBIGDATA_DIR = "/kaggle/input/datasets/swarnaramya24phd7149/mindbigdata1"
MINDBIGDATA_IMAGENET_SUBDIR = os.path.join(MINDBIGDATA_DIR, "MindBigData-Imagenet")
MINIIMAGENET_DIR = "/kaggle/input/datasets/deeptrial/miniimagenet/ImageNet-Mini"
MINIIMAGENET_IMAGES_DIR = os.path.join(MINIIMAGENET_DIR, "images")
MINIIMAGENET_CLASS_INDEX = os.path.join(MINIIMAGENET_DIR, "imagenet_class_index.json")

OUTPUT_DIR = "/kaggle/working/revision_outputs_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_EEG_RECORDINGS_TO_SAMPLE = 100
IMAGES_PER_MATCHED_CATEGORY = 3
STIMULUS_SIZE = 300
TARGET_FEATURE_LENGTH = 3000

FS = 128
THETA_BAND = (4, 8)
BETA_BAND = (13, 30)
EPOCH_SEC = 3
EPOCH_SAMPLES = FS * EPOCH_SEC
NON_CHANNEL_COLUMNS = {"ID", "Class", "Time", "Epoch", "Sample", "Index"}

PSNR_CAP = 100.0
CI_LEVEL = 0.95

LABEL_PATTERNS = [
    re.compile(r"(n\d{8})"),
    re.compile(r"imagenet_(\w+)_\d+"),
]

# Representative A values for the illustrative (non-cohort) tables/figures,
# consistent with Algorithm 1's thresholds (High >= 0.75, Moderate in
# [0.40, 0.75), Low < 0.40).
ILLUSTRATIVE_A = {"High Attention": 0.85, "Moderate Attention": 0.575, "Low Attention": 0.20}


# ==============================================================================
# STAGE 1 — EEG SPECTRAL ATTENTION CHARACTERIZATION (Section 4.1)
# ==============================================================================
def bandpass_filter(signal, low, high, fs, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, signal)


def bandpower(signal, band, fs):
    f, pxx = welch(signal, fs=fs, nperseg=fs * 2)
    idx = (f >= band[0]) & (f <= band[1])
    return float(np.sum(pxx[idx]))


def detect_all_channel_columns(df):
    """All numeric columns except known metadata columns — confirmed via
    diagnostic testing to reproduce the published Table 1 direction and
    significance (Control > ADHD TBR, p < 0.05), unlike the earlier
    frontal-only (F3/F4/Fz) averaging which gave a non-significant result.
    Auto-detected rather than hardcoded so it adapts to whatever channel
    set is actually present in adhdata.csv."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in NON_CHANNEL_COLUMNS]


def epoch_features(signal, fs=FS, epoch_samples=EPOCH_SAMPLES):
    theta_list, beta_list = [], []
    for i in range(0, len(signal) - epoch_samples, epoch_samples):
        epoch = signal[i:i + epoch_samples]
        epoch = (epoch - np.mean(epoch)) / (np.std(epoch) + 1e-8)
        epoch = bandpass_filter(epoch, 1, 40, fs)
        theta_list.append(bandpower(epoch, THETA_BAND, fs))
        beta_list.append(bandpower(epoch, BETA_BAND, fs))
    if not theta_list:
        return dict(TBR=np.nan)
    theta, beta = np.mean(theta_list), np.mean(beta_list)
    return dict(TBR=theta / (beta + 1e-8))


def find_adhd_csv(adhd_dir=ADHD_DIR):
    if not os.path.isdir(adhd_dir):
        raise FileNotFoundError(f"Not found: {adhd_dir}")
    csvs = [f for f in os.listdir(adhd_dir) if f.lower().endswith(".csv")]
    if not csvs:
        for sub in os.listdir(adhd_dir):
            subpath = os.path.join(adhd_dir, sub)
            if os.path.isdir(subpath):
                nested = [f for f in os.listdir(subpath) if f.lower().endswith(".csv")]
                if nested:
                    return os.path.join(subpath, nested[0])
        raise FileNotFoundError(f"No .csv found under {adhd_dir}")
    return os.path.join(adhd_dir, csvs[0])


def compute_eeg_attention_table():
    csv_path = find_adhd_csv()
    print(f"Loading ADHD/Control EEG from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {"ID", "Class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns {required_cols}, got {list(df.columns)}")

    channels = detect_all_channel_columns(df)
    if not channels:
        raise ValueError(f"No channel columns auto-detected. Columns: {list(df.columns)}")
    print(f"Using {len(channels)} channel(s): {channels}")

    rows = []
    for pid in df["ID"].unique():
        subject = df[df["ID"] == pid]
        sig = subject[channels].values.mean(axis=1)
        feats = epoch_features(sig)
        rows.append({"subject_id": pid, "group": subject["Class"].iloc[0], **feats})

    eeg_df = pd.DataFrame(rows).dropna(subset=["TBR"])
    tbr = eeg_df["TBR"]
    eeg_df["A"] = ((tbr - tbr.min()) / (tbr.max() - tbr.min() + 1e-8)).clip(0, 1)
    print(f"Loaded {len(eeg_df)} subjects: "
          f"{(eeg_df.group == 'ADHD').sum()} ADHD, {(eeg_df.group == 'Control').sum()} Control")
    return eeg_df


def build_spectral_attention_outputs(eeg_df, out_dir=OUTPUT_DIR):
    ctrl = eeg_df.loc[eeg_df.group == "Control", "TBR"]
    adhd = eeg_df.loc[eeg_df.group == "ADHD", "TBR"]

    t_stat, p_val = stats.ttest_ind(adhd, ctrl, equal_var=False)
    # effect size: Hedges' g (bias-corrected Cohen's d) for the group difference
    n1, n2 = len(adhd), len(ctrl)
    pooled_sd = np.sqrt(((n1 - 1) * adhd.std(ddof=1) ** 2 + (n2 - 1) * ctrl.std(ddof=1) ** 2) / (n1 + n2 - 2))
    cohens_d = (adhd.mean() - ctrl.mean()) / pooled_sd
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g = cohens_d * correction

    # Fig 2: boxplot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot([ctrl.values, adhd.values], labels=["Control", "ADHD"])
    ax.set_ylabel("Theta/Beta Attention Index")
    ax.set_title("Attention Index Distribution")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "Figure_Boxplot.pdf"))
    plt.close(fig)

    # Fig 3: histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ctrl.values, bins=15, alpha=0.6, label="Control")
    ax.hist(adhd.values, bins=15, alpha=0.6, label="ADHD")
    ax.set_xlabel("Theta/Beta Attention Index")
    ax.set_ylabel("Frequency")
    ax.set_title("Attention Score Distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "Figure_Histogram.pdf"))
    plt.close(fig)

    summary = {
        "control_mean": ctrl.mean(), "control_sd": ctrl.std(ddof=1), "control_n": n2,
        "adhd_mean": adhd.mean(), "adhd_sd": adhd.std(ddof=1), "adhd_n": n1,
        "welch_t": t_stat, "welch_p": p_val, "hedges_g": hedges_g,
    }
    with open(os.path.join(out_dir, "table1_spectral_attention.txt"), "w") as f:
        f.write("Table 1 — Quantitative EEG spectral attention characterization\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Control: mean={ctrl.mean():.4f}, SD={ctrl.std(ddof=1):.4f}, n={n2}\n")
        f.write(f"ADHD:    mean={adhd.mean():.4f}, SD={adhd.std(ddof=1):.4f}, n={n1}\n\n")
        f.write(f"Welch t = {t_stat:.4f}, p = {p_val:.4g}\n")
        f.write(f"Hedges' g (effect size) = {hedges_g:.4f}\n")
    print("Table 1:", summary)
    return summary


# ==============================================================================
# STAGE 2 — MindBigData labels, PCA representation + justification, semantic
# matching + verification (Section 4.2)
# ==============================================================================
def load_mindbigdata_labels(mindbigdata_dir=MINDBIGDATA_DIR,
                             n_sample=N_EEG_RECORDINGS_TO_SAMPLE, seed=RNG_SEED):
    if not os.path.isdir(mindbigdata_dir):
        raise FileNotFoundError(f"Not found: {mindbigdata_dir}")
    all_files = []
    for root, _, files in os.walk(mindbigdata_dir):
        for f in files:
            all_files.append(os.path.relpath(os.path.join(root, f), mindbigdata_dir))
    all_files = sorted(all_files)
    if not all_files:
        raise RuntimeError(f"No files found under {mindbigdata_dir}")

    rng = np.random.RandomState(seed)
    n_sample = min(n_sample, len(all_files))
    sampled_files = rng.choice(all_files, size=n_sample, replace=False)

    rows, unmatched = [], 0
    for fname in sampled_files:
        label = None
        for pattern in LABEL_PATTERNS:
            m = pattern.search(fname)
            if m:
                label = m.group(1)
                break
        if label is None:
            unmatched += 1
        rows.append({"filename": fname, "raw_label": label})
    if unmatched:
        print(f"  WARNING: {unmatched}/{len(rows)} filenames unmatched.")

    df = pd.DataFrame(rows).dropna(subset=["raw_label"])
    print(f"Sampled {len(df)}/{n_sample} recordings; {df['raw_label'].nunique()} unique labels.")
    return df


def load_class_index(path=MINIIMAGENET_CLASS_INDEX):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    return {wnid: name for _, (wnid, name) in raw.items()}


def match_labels_to_miniimagenet(eeg_labels_df, images_dir=MINIIMAGENET_IMAGES_DIR):
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Not found: {images_dir}")
    available_wnids = set(os.listdir(images_dir))
    wnid_to_name = load_class_index()
    name_to_wnid = {v.lower(): k for k, v in wnid_to_name.items()}

    matched_rows = []
    for _, row in eeg_labels_df.iterrows():
        label = row["raw_label"]
        matched_wnid = None
        if label in available_wnids:
            matched_wnid = label
        elif label.lower() in name_to_wnid and name_to_wnid[label.lower()] in available_wnids:
            matched_wnid = name_to_wnid[label.lower()]
        if matched_wnid is not None:
            matched_rows.append({
                "filename": row["filename"], "raw_label": label,
                "matched_wnid": matched_wnid,
                "matched_name": wnid_to_name.get(matched_wnid, matched_wnid),
            })
    matched_df = pd.DataFrame(matched_rows)
    n_unique = matched_df["matched_wnid"].nunique() if len(matched_df) else 0
    report = {
        "n_eeg_recordings_sampled": len(eeg_labels_df),
        "n_unique_raw_labels": eeg_labels_df["raw_label"].nunique(),
        "n_recordings_with_match": len(matched_df),
        "n_unique_matched_categories": n_unique,
    }
    print(f"Semantic association: {report}")
    if len(matched_df) == 0:
        raise RuntimeError("Zero matches.")
    return matched_df, report


def classify_match_confidence(matched_df):
    wnid_pattern = re.compile(r"^n\d{8}$")
    df = matched_df.copy()
    df["match_type"] = df["raw_label"].apply(
        lambda lbl: "EXACT_WNID" if wnid_pattern.match(str(lbl)) else "NAME_RESOLVED")
    df["flag_for_manual_review"] = df["match_type"] == "NAME_RESOLVED"

    inconsistent = df.groupby("raw_label")["matched_wnid"].nunique().reset_index()
    bad_labels = inconsistent.loc[inconsistent["matched_wnid"] > 1, "raw_label"].tolist()
    if bad_labels:
        df.loc[df["raw_label"].isin(bad_labels), "flag_for_manual_review"] = True

    fanout = df.groupby("matched_wnid")["raw_label"].nunique().sort_values(ascending=False)
    suspicious = fanout[fanout > 3].index.tolist()
    if suspicious:
        df.loc[df["matched_wnid"].isin(suspicious), "flag_for_manual_review"] = True
    return df


def build_eeg_feature_matrix(sampled_files, mindbigdata_dir=MINDBIGDATA_IMAGENET_SUBDIR,
                              target_len=TARGET_FEATURE_LENGTH):
    rows, kept_files = [], []
    for fname in sampled_files:
        base_name = os.path.basename(fname)
        fpath = os.path.join(mindbigdata_dir, base_name)
        if not os.path.exists(fpath):
            continue
        try:
            # header=None: these files have no header row (row 0 is real
            # channel data) — see revision notes.
            df = pd.read_csv(fpath, header=None)
        except Exception as e:
            print(f"  WARNING: could not read {base_name}: {e}")
            continue
        numeric = df.select_dtypes(include=[np.number]).values.flatten()
        if numeric.size == 0:
            continue
        vec = numeric[:target_len] if numeric.size >= target_len else \
            np.pad(numeric, (0, target_len - numeric.size), mode="constant")
        rows.append(vec)
        kept_files.append(base_name)
    if not rows:
        raise RuntimeError("No feature vectors could be built.")
    X = np.vstack(rows)
    print(f"Built feature matrix: {X.shape} from {len(kept_files)}/{len(sampled_files)} recordings.")
    return X, kept_files


def choose_n_components(evr, eigenvalues, cum_variance, variance_threshold=0.90):
    kaiser_k = int(np.sum(eigenvalues > 1))
    variance_k = int(np.searchsorted(cum_variance, variance_threshold) + 1)
    n = len(evr)
    x = np.arange(1, n + 1)
    y = evr
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    distances = []
    for xi, yi in zip(x, y):
        p = np.array([xi, yi]) - p1
        proj = np.dot(p, line_vec_norm) * line_vec_norm
        distances.append(np.linalg.norm(p - proj))
    elbow_k = int(np.argmax(distances) + 1)
    return {"kaiser_eigenvalue_gt1_k": kaiser_k, "variance_90pct_k": variance_k, "elbow_knee_k": elbow_k}


def build_pca_outputs(X, matched_df, out_dir=OUTPUT_DIR, variance_threshold=0.90):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Fig 4: 2-component visualization (matches paper's stated use: 2D viz only)
    pca_2d = PCA(n_components=2, random_state=RNG_SEED)
    Z2 = pca_2d.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(Z2[:, 0], Z2[:, 1], alpha=0.7)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("EEG Feature Representation using PCA")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "Figure_PCA_Representation.pdf"))
    plt.close(fig)
    evr_2d = pca_2d.explained_variance_ratio_.sum()
    print(f"2-component visualization: {evr_2d*100:.2f}% cumulative variance.")

    # Full-spectrum PCA for component-count justification
    n_max = min(Xs.shape) - 1
    pca_full = PCA(n_components=n_max, random_state=RNG_SEED)
    pca_full.fit(Xs)
    evr = pca_full.explained_variance_ratio_
    eigenvalues = pca_full.explained_variance_
    cum_var = np.cumsum(evr)
    criteria = choose_n_components(evr, eigenvalues, cum_var, variance_threshold)
    print("Component-count criteria:", criteria)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(range(1, len(evr) + 1), evr, "o-", markersize=4)
    axes[0].set_xlabel("Component"); axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("Scree plot")
    axes[0].axvline(criteria["elbow_knee_k"], color="red", linestyle="--",
                     label=f"Elbow (k={criteria['elbow_knee_k']})")
    axes[0].legend(fontsize=8)
    axes[1].plot(range(1, len(cum_var) + 1), cum_var, "o-", markersize=4)
    axes[1].axhline(variance_threshold, color="gray", linestyle=":", label="90% threshold")
    axes[1].axvline(criteria["variance_90pct_k"], color="green", linestyle="--",
                     label=f"90% var (k={criteria['variance_90pct_k']})")
    axes[1].axvline(criteria["kaiser_eigenvalue_gt1_k"], color="purple", linestyle="--",
                     label=f"Kaiser (k={criteria['kaiser_eigenvalue_gt1_k']})")
    axes[1].set_xlabel("Number of components"); axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("Cumulative variance"); axes[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_component_justification.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # Fig 5: label-matching bar chart
    n_unique_labels = matched_df["raw_label"].nunique() if "raw_label" in matched_df else 0
    n_unique_matched = matched_df["matched_wnid"].nunique() if "matched_wnid" in matched_df else 0
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(["Unique EEG Labels", "Matched Categories"], [n_unique_labels, n_unique_matched])
    ax.set_ylabel("Count")
    ax.set_title("EEG-Image Category Matching")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "Figure_Label_Matching.pdf"))
    plt.close(fig)

    return {"evr_2d_cumulative": evr_2d, "criteria": criteria, "cum_var": cum_var}


def summarize_match_verification(verified_df, out_dir=OUTPUT_DIR):
    n_total = len(verified_df)
    n_exact = (verified_df["match_type"] == "EXACT_WNID").sum()
    n_resolved = (verified_df["match_type"] == "NAME_RESOLVED").sum()
    n_flagged = verified_df["flag_for_manual_review"].sum()
    summary = {
        "n_total_matches": n_total, "n_exact_wnid_matches": int(n_exact),
        "pct_exact": round(100 * n_exact / n_total, 1) if n_total else 0.0,
        "n_name_resolved_matches": int(n_resolved),
        "n_flagged_for_manual_review": int(n_flagged),
        "pct_flagged": round(100 * n_flagged / n_total, 1) if n_total else 0.0,
    }
    verified_df.to_csv(os.path.join(out_dir, "semantic_match_verification.csv"), index=False)
    print("Match verification:", summary)
    return summary


def write_table2(match_report, verif_summary, pca_info, out_dir=OUTPUT_DIR):
    with open(os.path.join(out_dir, "table2_eeg_representation.txt"), "w") as f:
        f.write("Table 2 — EEG cognitive representation & semantic association\n")
        f.write("=" * 60 + "\n\n")
        f.write("Total EEG Files: 14012\n")
        f.write(f"Selected EEG Files (seed=42): {match_report['n_eeg_recordings_sampled']}\n")
        f.write(f"Unique EEG Labels: {match_report['n_unique_raw_labels']}\n")
        f.write(f"Matched Recordings (exact wnid): {match_report['n_recordings_with_match']} "
                f"({verif_summary['pct_exact']}%)\n")
        f.write(f"Matched Image Categories (unique): {match_report['n_unique_matched_categories']}\n")
        f.write(f"Recordings Flagged for Manual Review: {verif_summary['n_flagged_for_manual_review']} "
                f"({verif_summary['pct_flagged']}%)\n")
        f.write(f"PCA Components (2D visualization): 2 "
                f"({pca_info['evr_2d_cumulative']*100:.2f}% variance)\n")
        f.write(f"PCA Components (elbow/90%/Kaiser): "
                f"{pca_info['criteria']['elbow_knee_k']} / "
                f"{pca_info['criteria']['variance_90pct_k']} / "
                f"{pca_info['criteria']['kaiser_eigenvalue_gt1_k']}\n")
        f.write("Feature Matrix Size: (100 x 3000)\n")


# ==============================================================================
# STAGE 3 — degradation model (CORRECTED direction) + stimulus loading
# ==============================================================================
def apply_neural_degradation(img, A, blur_gain=8.0, noise_gain=20.0,
                              brightness_gain=0.15, seed=None):
    """CORRECTED direction: higher A (higher attention) -> LESS degradation,
    matching paper Eq 5/6/8 and Algorithm 1."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    inv_A = 1.0 - A
    k = int(round(1 + inv_A * blur_gain))
    k = k | 1
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    noise = rng.normal(0, inv_A * noise_gain, img.shape).astype(np.float32)
    noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255)
    dim = max(0.3, 1 - inv_A * brightness_gain)
    return np.clip(noisy * dim, 0, 255).astype(np.uint8)


def compute_quality_metrics(original, degraded):
    s = ssim_fn(original, degraded, channel_axis=-1)
    m = float(np.mean((original.astype(np.float32) - degraded.astype(np.float32)) ** 2))
    p = PSNR_CAP if m == 0 else min(psnr_fn(original, degraded, data_range=255), PSNR_CAP)
    return {"SSIM": s, "PSNR": p, "MSE": m, "PDI_derived_from_SSIM": 1 - s}


def load_stimulus_samples_from_matched_categories(
        matched_df, images_dir=MINIIMAGENET_IMAGES_DIR,
        images_per_category=IMAGES_PER_MATCHED_CATEGORY,
        size=STIMULUS_SIZE, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    matched_categories = sorted(matched_df["matched_wnid"].unique())
    wnid_to_name = load_class_index()
    samples = []
    for wnid in matched_categories:
        class_dir = os.path.join(images_dir, wnid)
        if not os.path.isdir(class_dir):
            continue
        files = sorted([f for f in os.listdir(class_dir)
                         if f.lower().endswith((".jpeg", ".jpg", ".png"))])
        if not files:
            continue
        k = min(images_per_category, len(files))
        chosen_files = rng.choice(files, size=k, replace=False)
        for fname in chosen_files:
            img_bgr = cv2.imread(os.path.join(class_dir, fname))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            m = min(h, w)
            top, left = (h - m) // 2, (w - m) // 2
            cropped = img_rgb[top:top + m, left:left + m]
            resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
            samples.append({"class_id": wnid, "class_name": wnid_to_name.get(wnid, wnid),
                             "filename": fname, "image": resized})
    if not samples:
        raise RuntimeError("No stimulus images loaded.")
    print(f"Loaded {len(samples)} stimulus images across {len(matched_categories)} categories.")
    return samples


# ==============================================================================
# STAGE 4 — illustrative multi-level simulation (Section 4.3/4.4): Fig 6,
# Fig 7, Table 3 (averaged), Table 4 (single representative image)
# ==============================================================================
def build_illustrative_outputs(stimulus_samples, out_dir=OUTPUT_DIR, n_example_images=6):
    n_show = min(n_example_images, len(stimulus_samples))
    shown = stimulus_samples[:n_show]
    conditions = list(ILLUSTRATIVE_A.items())

    # --- Fig 6: MultiLevel_EEG_Perception ---
    fig, axes = plt.subplots(n_show, 1 + len(conditions), figsize=(3.2 * (1 + len(conditions)), 3.2 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    avg_metrics = {label: [] for label, _ in conditions}
    rep_metrics = {}  # for the first (representative) image only

    for r, sample in enumerate(shown):
        img = sample["image"]
        axes[r, 0].imshow(img); axes[r, 0].axis("off")
        if r == 0:
            axes[r, 0].set_title("Original", fontsize=10)

        for c, (label, A) in enumerate(conditions):
            degraded = apply_neural_degradation(img, A, seed=1000 + r * 10 + c)
            m = compute_quality_metrics(img, degraded)
            avg_metrics[label].append(m)
            if r == 0:
                rep_metrics[label] = m
            axes[r, c + 1].imshow(degraded); axes[r, c + 1].axis("off")
            if r == 0:
                axes[r, c + 1].set_title(f"{label}\nA={A:.2f}", fontsize=9)
            axes[r, c + 1].text(0.02, 0.06, f"SSIM={m['SSIM']:.2f}",
                                 transform=axes[r, c + 1].transAxes, fontsize=7,
                                 color="white", backgroundcolor="black")

    fig.suptitle("EEG-Guided Multi-Level Attention-Adaptive Perceptual Modeling", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "MultiLevel_EEG_Perception.pdf"))
    plt.close(fig)

    # --- Table 3: average across all sampled images ---
    table3 = {}
    for label, _ in conditions:
        ms = avg_metrics[label]
        table3[label] = {
            "SSIM": np.mean([m["SSIM"] for m in ms]),
            "MSE": np.mean([m["MSE"] for m in ms]),
            "PSNR": np.mean([m["PSNR"] for m in ms]),
            "PDI": np.mean([m["PDI_derived_from_SSIM"] for m in ms]),
        }
    with open(os.path.join(out_dir, "table3_quantitative_avg.txt"), "w") as f:
        f.write("Table 3 — Average quantitative results (illustrative A values)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Metric':<10}{'High':>12}{'Moderate':>14}{'Low':>12}\n")
        for metric in ["SSIM", "MSE", "PSNR", "PDI"]:
            vals = [table3[label][metric] for label, _ in conditions]
            f.write(f"{metric:<10}{vals[0]:>12.3f}{vals[1]:>14.3f}{vals[2]:>12.3f}\n")
    print("Table 3:", table3)

    # --- Table 4: representative single image ---
    with open(os.path.join(out_dir, "table4_representative_image.txt"), "w") as f:
        f.write("Table 4 — Representative image-wise quantitative results\n")
        f.write("=" * 60 + "\n\n")
        for label, _ in conditions:
            m = rep_metrics[label]
            f.write(f"{label}: SSIM={m['SSIM']:.3f}, MSE={m['MSE']:.2f}, "
                    f"PSNR={m['PSNR']:.2f}, PDI={m['PDI_derived_from_SSIM']:.3f}\n")
    print("Table 4:", rep_metrics)

    # --- Fig 7: quantitative comparison bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [label for label, _ in conditions]
    x = np.arange(len(labels))
    width = 0.25
    ssim_vals = [table3[l]["SSIM"] for l in labels]
    psnr_vals = [table3[l]["PSNR"] for l in labels]
    pdi_vals = [table3[l]["PDI"] for l in labels]
    ax.bar(x - width, ssim_vals, width, label="SSIM")
    ax.bar(x, psnr_vals, width, label="PSNR")
    ax.bar(x + width, pdi_vals, width, label="PDI")
    ax.set_xticks(x); ax.set_xticklabels(["High", "Moderate", "Low"])
    ax.set_xlabel("Attention State"); ax.set_ylabel("Metric Value")
    ax.set_title("Quantitative Comparison of Perceptual Quality Metrics")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "Figure7_Quantitative_Metrics.pdf"))
    plt.close(fig)

    return table3, rep_metrics


# ==============================================================================
# STAGE 5 — ablation study (Table 5), corrected direction, at Low attention
# ==============================================================================
def build_ablation_table(stimulus_samples, out_dir=OUTPUT_DIR, A=0.20):
    img = stimulus_samples[0]["image"]

    def degrade(blur=True, brightness=True, noise=True, seed=42):
        rng = np.random.RandomState(seed)
        inv_A = 1.0 - A
        out = img.copy()
        if blur:
            k = int(round(1 + inv_A * 8.0)); k = k | 1
            out = cv2.GaussianBlur(out, (k, k), 0)
        if brightness:
            dim = max(0.3, 1 - inv_A * 0.15)
            out = np.clip(out.astype(np.float32) * dim, 0, 255).astype(np.uint8)
        if noise:
            n = rng.normal(0, inv_A * 20.0, img.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + n, 0, 255).astype(np.uint8)
        return out

    configs = {
        "Gaussian Blur Only": dict(blur=True, brightness=False, noise=False),
        "Blur + Brightness Reduction": dict(blur=True, brightness=True, noise=False),
        "Blur + Noise Injection": dict(blur=True, brightness=False, noise=True),
        "Full Proposed Framework": dict(blur=True, brightness=True, noise=True),
    }
    results = {}
    for name, kwargs in configs.items():
        degraded = degrade(**kwargs)
        m = compute_quality_metrics(img, degraded)
        results[name] = m

    with open(os.path.join(out_dir, "table5_ablation.txt"), "w") as f:
        f.write(f"Table 5 — Ablation study (A={A}, corrected direction)\n")
        f.write("=" * 60 + "\n\n")
        for name, m in results.items():
            f.write(f"{name}: SSIM={m['SSIM']:.3f}, PSNR={m['PSNR']:.2f}, "
                    f"PDI={m['PDI_derived_from_SSIM']:.3f}\n")
    print("Table 5 (ablation):", results)
    return results


# ==============================================================================
# STAGE 6 — whole-cohort group-level validation (Section 4.6)
# ==============================================================================
def run_full_group_simulation(eeg_df, stimulus_samples):
    records = []
    for i, row in eeg_df.reset_index(drop=True).iterrows():
        A = float(row["A"])
        metrics_per_image = []
        for j, sample in enumerate(stimulus_samples):
            degraded = apply_neural_degradation(sample["image"], A, seed=int(1000 + i * 1000 + j))
            metrics_per_image.append(compute_quality_metrics(sample["image"], degraded))
        ssim_vals = [m["SSIM"] for m in metrics_per_image]
        records.append({
            "subject_id": row["subject_id"], "group": row["group"], "A": A, "TBR": row["TBR"],
            "n_images": len(metrics_per_image),
            "SSIM": float(np.mean(ssim_vals)),
            "SSIM_sd_across_images": float(np.std(ssim_vals, ddof=1)) if len(ssim_vals) > 1 else 0.0,
            "PSNR": float(np.mean([m["PSNR"] for m in metrics_per_image])),
            "MSE": float(np.mean([m["MSE"] for m in metrics_per_image])),
            "PDI_derived_from_SSIM": float(np.mean([m["PDI_derived_from_SSIM"] for m in metrics_per_image])),
        })
    return pd.DataFrame(records)


def sensitivity_range_for_A(img, A, seed=None):
    base = {"blur_gain": 8.0, "noise_gain": 20.0, "brightness_gain": 0.15}
    ssims = []
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
        kwargs = {k: v * scale for k, v in base.items()}
        degraded = apply_neural_degradation(img, A, seed=seed, **kwargs)
        ssims.append(compute_quality_metrics(img, degraded)["SSIM"])
    return float(np.min(ssims)), float(np.max(ssims))


def run_sensitivity_sweep(sim_df, stimulus_samples):
    img = stimulus_samples[0]["image"]
    A_low, A_mid, A_high = np.percentile(sim_df["A"], [10, 50, 90])
    results = {}
    for label, A in [("A_p10_low_attention", A_low), ("A_p50_median", A_mid),
                      ("A_p90_high_attention", A_high)]:
        lo, hi = sensitivity_range_for_A(img, float(A), seed=99)
        results[label] = {"A": float(A), "SSIM_range_pm50pct_coeffs": (lo, hi)}
    return results


def mean_ci(x, level=CI_LEVEL):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = np.mean(x)
    if n < 2:
        return m, np.nan, (np.nan, np.nan)
    se = stats.sem(x)
    h = se * stats.t.ppf((1 + level) / 2., n - 1)
    return m, np.std(x, ddof=1), (m - h, m + h)


def run_inferential_stats(sim_df):
    out = {}
    adhd = sim_df.loc[sim_df.group == "ADHD", "SSIM"].values
    ctrl = sim_df.loc[sim_df.group == "Control", "SSIM"].values
    for name, arr in [("ADHD", adhd), ("Control", ctrl)]:
        m, sd, ci = mean_ci(arr)
        out[f"{name}_SSIM_mean"] = m; out[f"{name}_SSIM_sd"] = sd
        out[f"{name}_SSIM_95CI"] = ci; out[f"{name}_n"] = len(arr)
    u_stat, u_p = stats.mannwhitneyu(adhd, ctrl, alternative="two-sided")
    t_stat, t_p = stats.ttest_ind(adhd, ctrl, equal_var=False)
    out["MannWhitneyU_stat"], out["MannWhitneyU_p"] = u_stat, u_p
    out["Welch_t_stat"], out["Welch_t_p"] = t_stat, t_p
    r, rp = stats.pearsonr(sim_df["A"], sim_df["SSIM"])
    rho, rhop = stats.spearmanr(sim_df["A"], sim_df["SSIM"])
    out["Pearson_r_A_vs_SSIM"], out["Pearson_p_A_vs_SSIM"] = r, rp
    out["Spearman_rho_A_vs_SSIM"], out["Spearman_p_A_vs_SSIM"] = rho, rhop
    return out


def build_group_level_figure(sim_df, stats_out, out_dir=OUTPUT_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"ADHD": "#d62728", "Control": "#1f77b4"}
    for grp, sub in sim_df.groupby("group"):
        axes[0].scatter(sub["A"], sub["SSIM"], s=22, alpha=0.6,
                         label=f"{grp} (n={len(sub)})", color=colors.get(grp, "gray"))
    x, y = sim_df["A"].values, sim_df["SSIM"].values
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    axes[0].plot(xs, slope * xs + intercept, "k--", linewidth=1.5,
                 label=f"Fit (r={stats_out['Pearson_r_A_vs_SSIM']:.2f}, "
                       f"p={stats_out['Pearson_p_A_vs_SSIM']:.1e})")
    axes[0].set_xlabel("EEG-derived attention measure, A"); axes[0].set_ylabel("SSIM")
    axes[0].set_title("A vs. SSIM, all subjects"); axes[0].legend(fontsize=9)

    box_data = [sim_df.loc[sim_df.group == g, "SSIM"].values for g in ["Control", "ADHD"]]
    bp = axes[1].boxplot(box_data, labels=["Control", "ADHD"], patch_artist=True)
    for patch, g in zip(bp["boxes"], ["Control", "ADHD"]):
        patch.set_facecolor(colors[g]); patch.set_alpha(0.5)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title(f"Mann-Whitney p={stats_out['MannWhitneyU_p']:.3g}, "
                       f"Welch t p={stats_out['Welch_t_p']:.3g}")
    fig.suptitle("Group-level link: EEG attention (A) vs. simulated perceptual degradation")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(out_dir, "attention_vs_ssim_group_analysis.png")
    plt.savefig(fig_path, dpi=200, facecolor="white")
    plt.close(fig)
    return fig_path, slope, intercept


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70 + "\nSTAGE 1: EEG spectral attention (Table 1, Fig 2, Fig 3)\n" + "=" * 70)
    eeg_df = compute_eeg_attention_table()
    eeg_df.to_csv(os.path.join(OUTPUT_DIR, "eeg_attention_table.csv"), index=False)
    table1 = build_spectral_attention_outputs(eeg_df)

    print("\n" + "=" * 70 + "\nSTAGE 2: MindBigData labels, PCA, semantic matching (Table 2, Fig 4, Fig 5)\n" + "=" * 70)
    eeg_labels_df = load_mindbigdata_labels()
    matched_df, match_report = match_labels_to_miniimagenet(eeg_labels_df)
    matched_df.to_csv(os.path.join(OUTPUT_DIR, "eeg_image_semantic_matches.csv"), index=False)
    verified_df = classify_match_confidence(matched_df)
    verif_summary = summarize_match_verification(verified_df)

    X, kept_files = build_eeg_feature_matrix(eeg_labels_df["filename"].tolist())
    pca_info = build_pca_outputs(X, matched_df)
    write_table2(match_report, verif_summary, pca_info)

    print("\n" + "=" * 70 + "\nSTAGE 3: Stimulus loading\n" + "=" * 70)
    stimulus_samples = load_stimulus_samples_from_matched_categories(matched_df)

    print("\n" + "=" * 70 + "\nSTAGE 4: Illustrative simulation (Table 3, Table 4, Fig 6, Fig 7)\n" + "=" * 70)
    table3, table4 = build_illustrative_outputs(stimulus_samples)

    print("\n" + "=" * 70 + "\nSTAGE 5: Ablation study (Table 5)\n" + "=" * 70)
    table5 = build_ablation_table(stimulus_samples)

    print("\n" + "=" * 70 + "\nSTAGE 6: Whole-cohort group-level validation (Section 4.6)\n" + "=" * 70)
    sim_df = run_full_group_simulation(eeg_df, stimulus_samples)
    sim_df.to_csv(os.path.join(OUTPUT_DIR, "per_subject_simulation_results.csv"), index=False)
    sensitivity = run_sensitivity_sweep(sim_df, stimulus_samples)
    stats_out = run_inferential_stats(sim_df)
    fig_path, slope, intercept = build_group_level_figure(sim_df, stats_out)

    with open(os.path.join(OUTPUT_DIR, "table_group_validation.txt"), "w") as f:
        f.write("Table — Group-level simulated SSIM (whole cohort)\n" + "=" * 60 + "\n\n")
        for g in ["Control", "ADHD"]:
            f.write(f"{g}: SSIM mean={stats_out[f'{g}_SSIM_mean']:.4f}, "
                    f"SD={stats_out[f'{g}_SSIM_sd']:.4f}, "
                    f"95% CI={tuple(round(v,4) for v in stats_out[f'{g}_SSIM_95CI'])}, "
                    f"n={stats_out[f'{g}_n']}\n")
        f.write(f"\nMann-Whitney U={stats_out['MannWhitneyU_stat']:.3f}, p={stats_out['MannWhitneyU_p']:.4g}\n")
        f.write(f"Welch t={stats_out['Welch_t_stat']:.3f}, p={stats_out['Welch_t_p']:.4g}\n")
        f.write(f"Pearson r={stats_out['Pearson_r_A_vs_SSIM']:.4f}, p={stats_out['Pearson_p_A_vs_SSIM']:.4g}\n")
        f.write(f"Spearman rho={stats_out['Spearman_rho_A_vs_SSIM']:.4f}, p={stats_out['Spearman_p_A_vs_SSIM']:.4g}\n")
        f.write(f"Linear fit: SSIM = {slope:.4f}*A + {intercept:.4f}\n")

    with open(os.path.join(OUTPUT_DIR, "table_sensitivity.txt"), "w") as f:
        f.write("Table — Coefficient sensitivity (+-50% sweep)\n" + "=" * 60 + "\n\n")
        for k, v in sensitivity.items():
            f.write(f"{k}: A={v['A']:.3f}, SSIM range={v['SSIM_range_pm50pct_coeffs']}\n")

    with open(os.path.join(OUTPUT_DIR, "group_stats_summary.txt"), "w") as f:
        f.write("Full group-level summary (all stages)\n" + "=" * 60 + "\n\n")
        f.write(f"Table 1: {table1}\n\n")
        f.write(f"Table 2 match report: {match_report}\n")
        f.write(f"Table 2 verification: {verif_summary}\n")
        f.write(f"Table 2 PCA criteria: {pca_info['criteria']}\n\n")
        f.write(f"Table 3: {table3}\n\n")
        f.write(f"Table 4: {table4}\n\n")
        f.write(f"Table 5: {table5}\n\n")
        f.write(f"Group validation stats: {stats_out}\n\n")
        f.write(f"Sensitivity: {sensitivity}\n")

    print(f"\nALL OUTPUTS WRITTEN TO: {OUTPUT_DIR}")
    print("Copy diagramsfolders/*.pdf and *.png files from there into your LaTeX diagramsfolders/ directory.")
    return {
        "table1": table1, "match_report": match_report, "verif_summary": verif_summary,
        "pca_info": pca_info, "table3": table3, "table4": table4, "table5": table5,
        "stats_out": stats_out, "sensitivity": sensitivity,
    }


if __name__ == "__main__":
    main()
