"""
full_pipeline.py  --  EEG-parameterized image degradation: complete analysis pipeline
==============================================================================
Per-child theta/beta ratio (TBR) from the ADHD/Control EEG, label-based stimulus
selection, image degradation controlled by A, distortion measures, cohort-level
and sensitivity analyses.

Run on Kaggle with the three datasets attached (ADHD/Control EEG .mat files,
MindBigData, ImageNet-Mini). Everything is written to OUTPUT_DIR.

PRE-SPECIFIED CHOICES (written down before the run; edit only with a reason)
  EEG    : continuous 1-40 Hz zero-phase Butterworth (order 4) per channel ->
           non-overlapping 3-s epochs -> epoch rejection by amplitude rule ->
           Welch PSD per channel (256-sample Hann segments, 50% overlap) ->
           TBR_c = mean_epochs(theta_c) / mean_epochs(beta_c) per channel ->
           global TBR = mean over 19 channels.  No per-epoch z-scoring.
  Montages (equal prominence, NOT pre-specified as a single primary):
           Global (19 ch), Frontal (F3,F4,Fz), Fronto-central (Fz,Cz,F3,F4,C3,C4), Cz.
  TBR stats: Mann-Whitney U (rank-based), probability of superiority,
           Cliff's delta (= rank-biserial for two groups), stratified
           percentile bootstrap (B=10000, seed 42); Welch + Hedges' g on log TBR.
           Effect direction is always Control minus ADHD.
  A      : percentile rank of global TBR over all children (primary);
           sensitivity: log-TBR winsorised min-max, original raw min-max.
  Simulation: same operators/order/coefficients as the main pipeline
           (blur -> noise -> brightness, gains 8 / 20 / 0.15, factor (1-A)).
  Simulated metrics: DESCRIPTIVE ONLY (no p-values, no CIs, no fit line).
==============================================================================
"""
import os, re, sys, json, glob, time, shutil, platform, warnings
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import scipy
import skimage
from scipy import stats
from scipy.signal import welch, butter, filtfilt
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------ config --
ENV = os.environ.get
RNG_SEED = 42
ADHD_DIR = ENV("REV_ADHD_DIR", "/kaggle/input/datasets/swarnaramya24phd7149/adhd-dataset19")
MINDBIGDATA_DIR = ENV("REV_MBD_DIR", "/kaggle/input/datasets/swarnaramya24phd7149/mindbigdata1")
MINDBIGDATA_SUBDIR = os.path.join(MINDBIGDATA_DIR, "MindBigData-Imagenet")
MINIIMAGENET_DIR = ENV("REV_IMNET_DIR", "/kaggle/input/datasets/deeptrial/miniimagenet/ImageNet-Mini")
IMAGES_DIR = os.path.join(MINIIMAGENET_DIR, "images")
CLASS_INDEX = os.path.join(MINIIMAGENET_DIR, "imagenet_class_index.json")
OUTPUT_DIR = ENV("REV_OUT", "/kaggle/working/revision_outputs_v2")

N_EEG_RECORDINGS = 100
IMAGES_PER_CATEGORY = 3
STIMULUS_SIZE = 300
FS = 128
EPOCH = 3 * FS                      # 384 samples
BAND_FILTER = (1, 40)
THETA, BETA = (4, 8), (13, 30)
WELCH_NPERSEG, WELCH_NOVERLAP = 256, 128
EXPECTED_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']   # Channel_Labels.docx
MONTAGES = {"Global (19 ch)": None, "Frontal (F3,F4,Fz)": ["F3", "F4", "Fz"],
            "Fronto-central (Fz,Cz,F3,F4,C3,C4)": ["Fz", "Cz", "F3", "F4", "C3", "C4"], "Cz": ["Cz"]}
MONTAGE_COL = {"Global (19 ch)": "TBR_global", "Frontal (F3,F4,Fz)": "TBR_frontal",
               "Fronto-central (Fz,Cz,F3,F4,C3,C4)": "TBR_frontocentral", "Cz": "TBR_Cz"}
FENCE_PRIMARY, FENCE_STRICT = 3.0, 1.5              # Tukey multipliers: far-out fence (primary), inner fence (sensitivity)
EEG_ONLY = ENV("REV_EEG_ONLY", "0") == "1"          # run only the EEG block (fast; no images, no simulation)
N_BOOT = int(ENV("REV_NBOOT", "10000"))
PSNR_CAP = 100.0
BASE_GAINS = (8.0, 20.0, 0.15)      # blur, noise, brightness
SSIM_KW = dict(channel_axis=-1, data_range=255, win_size=7, K1=0.01, K2=0.03,
               gaussian_weights=False, use_sample_covariance=True)
SWEEP_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]
GROUP_SWEEP_FACTORS = [0.5, 1.5]
N_JOBS = int(ENV("REV_JOBS", str(os.cpu_count() or 1)))
TEST_MAX_SUBJECTS = int(ENV("REV_TEST_MAX_SUBJECTS", "0"))   # 0 = all (testing aid only)
SWEEP_IMAGE_LIMIT = int(ENV("REV_SWEEP_IMAGES", "0"))        # 0 = full pool (reviewer asked for full pool)
LABEL_PATTERNS = [re.compile(r"(n\d{8})"), re.compile(r"imagenet_(\w+)_\d+")]

os.makedirs(OUTPUT_DIR, exist_ok=True)
_LOG = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); _LOG.append(s)
def header(t):
    log("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
def save_fig(fig, name):
    try: fig.tight_layout()
    except Exception: pass
    fig.savefig(os.path.join(OUTPUT_DIR, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, name + ".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ================================================================ statistics ==
def hedges_g(a, b):
    """Hedges' g, a minus b."""
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (n1 + n2) - 9))

def prob_superiority(x, y):
    """P(x > y) + 0.5 P(x == y)."""
    ys = np.sort(y)
    less = np.searchsorted(ys, x, "left"); le = np.searchsorted(ys, x, "right")
    return (less.sum() + 0.5 * (le - less).sum()) / (len(x) * len(y))

def describe(a):
    a = np.asarray(a, float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return dict(n=len(a), mean=a.mean(), sd=a.std(ddof=1) if len(a) > 1 else np.nan,
                median=med, q1=q1, q3=q3, min=a.min(), max=a.max())

def tbr_group_stats(x, y, B=N_BOOT, seed=RNG_SEED):
    """x = Control, y = ADHD. Rank-based primary + parametric on log scale."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    u = stats.mannwhitneyu(x, y, alternative="two-sided")
    ps = prob_superiority(x, y)
    welch_raw = stats.ttest_ind(x, y, equal_var=False)
    welch_log = stats.ttest_ind(np.log(x), np.log(y), equal_var=False)
    rng = np.random.RandomState(seed)
    n1, n2 = len(x), len(y)
    lx, ly = np.log(x), np.log(y)
    bps, bmd, bg = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        xi = rng.randint(0, n1, n1); yi = rng.randint(0, n2, n2)
        bps[b] = prob_superiority(x[xi], y[yi])
        bmd[b] = np.median(x[xi]) - np.median(y[yi])
        bg[b] = hedges_g(lx[xi], ly[yi])
    ci = lambda v: tuple(np.percentile(v, [2.5, 97.5]))
    return dict(
        n_control=n1, n_adhd=n2, U=u.statistic, p_mwu=u.pvalue,
        prob_superiority=ps, ps_ci=ci(bps),
        cliffs_delta=2 * ps - 1, cliffs_delta_ci=tuple(2 * np.array(ci(bps)) - 1),
        median_diff=np.median(x) - np.median(y), median_diff_ci=ci(bmd),
        welch_raw_t=welch_raw.statistic, welch_raw_p=welch_raw.pvalue,
        welch_log_t=welch_log.statistic, welch_log_p=welch_log.pvalue,
        hedges_g_log=hedges_g(lx, ly), hedges_g_log_ci=ci(bg),
        hedges_g_raw=hedges_g(x, y))

def cliffs_delta(x, y):
    return 2 * prob_superiority(np.asarray(x, float), np.asarray(y, float)) - 1


# ============================================================ EEG pipeline ===
def find_adhd_source():
    """Finds the ADHD/Control data under ADHD_DIR: the original IEEE .mat files (preferred), or a CSV with columns ID and Class."""
    mats = sorted(glob.glob(os.path.join(ADHD_DIR, "**", "*.mat"), recursive=True))
    if mats: return "mat", mats
    for c in sorted(glob.glob(os.path.join(ADHD_DIR, "**", "*.csv"), recursive=True)):
        try:
            if {"ID", "Class"} <= set(pd.read_csv(c, nrows=1).columns): return "csv", c
        except Exception:
            pass
    raise FileNotFoundError(f"No ADHD data found under {ADHD_DIR}: need the original .mat files or adhdata.csv (columns ID, Class)")

_B, _A = butter(4, [BAND_FILTER[0] / (FS / 2), BAND_FILTER[1] / (FS / 2)], btype="band")

def load_adhd():
    kind, src = find_adhd_source()
    ch = EXPECTED_CHANNELS          # column order of the CSV and of the .mat files = order of the dataset's Channel_Labels file
    subj = {}
    if kind == "csv":
        df = pd.read_csv(src)
        got = [c for c in df.columns if c not in ("ID", "Class")]
        if got != EXPECTED_CHANNELS: log(f"WARNING: channel order differs from Channel_Labels.docx order!\n  found {got}")
        ch = got
        for pid in list(pd.unique(df["ID"])):
            sdf = df[df.ID == pid]; subj[pid] = (sdf["Class"].iloc[0], sdf[ch].values.astype(np.float64))
        log(f"ADHD/Control CSV file: {src}")
    else:
        from scipy.io import loadmat
        for path in src:
            pid = os.path.splitext(os.path.basename(path))[0]
            group = "ADHD" if "ADHD_part" in path else ("Control" if "Control_part" in path else None)
            if group is None: continue
            m = loadmat(path); arr = m[pid] if pid in m else [v for k, v in m.items() if not k.startswith("__")][0]
            subj[pid] = (group, np.asarray(arr, dtype=np.float64))
        log(f"ADHD/Control .mat files: {len(subj)} files, e.g. {src[0]}")
    if TEST_MAX_SUBJECTS:
        keep = []
        for g in ["ADHD", "Control"]:
            keep += [i for i in subj if subj[i][0] == g][:TEST_MAX_SUBJECTS // 2]
        subj = {i: subj[i] for i in keep}
    log(f"  subjects={len(subj)}  (ADHD={sum(v[0]=='ADHD' for v in subj.values())}, "
        f"Control={sum(v[0]=='Control' for v in subj.values())});  channels={len(ch)}")
    return subj, ch

def filtered_epochs(sig):
    """Continuous filtering (all channels) then non-overlapping 3-s epochs (full epochs only)."""
    filt = np.column_stack([filtfilt(_B, _A, sig[:, c]) for c in range(sig.shape[1])]).astype(np.float32)
    n_ep = sig.shape[0] // EPOCH
    ep = filt[:n_ep * EPOCH].reshape(n_ep, EPOCH, sig.shape[1])
    return ep

def band_idx(f, band):
    return (f >= band[0]) & (f <= band[1])

def prepare_epochs(subj):
    """Continuous filtering -> 3-s epochs; returns epochs, epoch max-over-channel peak-to-peak, flat flags."""
    eps, flat_flags, p2p_max = {}, {}, {}
    for pid, (g, sig) in subj.items():
        ep = filtered_epochs(sig)
        eps[pid] = ep
        p2p = ep.max(axis=1) - ep.min(axis=1)
        p2p_max[pid] = p2p.max(axis=1)
        flat_flags[pid] = (ep.std(axis=1) < 1e-8).any(axis=1)
    return eps, p2p_max, flat_flags

def compute_tbr(subj, ch, eps, p2p_max, flat_flags, mult, keep_psd=False):
    pooled = np.concatenate(list(p2p_max.values()))
    q1, q3 = np.percentile(pooled, [25, 75])
    thr = q3 + mult * (q3 - q1)
    rows, tbr_c, psd_store = [], {}, {}
    idx = {m: [ch.index(c) for c in cs] for m, cs in MONTAGES.items() if cs is not None}
    for pid, (g, sig) in subj.items():
        ep = eps[pid]
        keep = (p2p_max[pid] <= thr) & (~flat_flags[pid])
        n_tot, n_keep = len(keep), int(keep.sum())
        row = dict(subject_id=pid, group=g, n_samples=sig.shape[0], duration_s=sig.shape[0] / FS,
                   n_epochs_total=n_tot, n_epochs_retained=n_keep)
        if n_keep == 0:
            for col in MONTAGE_COL.values(): row[col] = np.nan
            row["TBR_global_pooled"] = np.nan; rows.append(row); continue
        f, P = welch(ep[keep].astype(np.float64), fs=FS, nperseg=WELCH_NPERSEG,
                     noverlap=WELCH_NOVERLAP, axis=1)          # (n_keep, n_freq, n_ch)
        th = P[:, band_idx(f, THETA), :].sum(axis=1).mean(axis=0)     # (n_ch,)
        be = P[:, band_idx(f, BETA), :].sum(axis=1).mean(axis=0)
        tbr = th / be
        tbr_c[pid] = tbr
        if keep_psd: psd_store[pid] = (f, P.mean(axis=0))
        row["TBR_global"] = tbr.mean(); row["TBR_global_pooled"] = th.mean() / be.mean()
        for m, ii in idx.items(): row[MONTAGE_COL[m]] = tbr[ii].mean()
        rows.append(row)
    df = pd.DataFrame(rows)
    df["seed_idx"] = np.arange(len(df))                 # position in the original subject order (keeps noise seeds stable)
    tbr_ch_df = pd.DataFrame({pid: v for pid, v in tbr_c.items()}, index=ch).T; tbr_ch_df.index.name = "subject_id"
    return df, tbr_ch_df, psd_store, thr

def eeg_analysis(subj, ch):
    header("EEG: per-channel PSD, continuous filtering, epoch rejection")
    eps, p2p_max, flat_flags = prepare_epochs(subj)
    tbr_df, tbr_ch_df, psd_store, thr = compute_tbr(subj, ch, eps, p2p_max, flat_flags, FENCE_PRIMARY, keep_psd=True)
    log(f"Epoch rejection rule (primary): reject an epoch if max-over-channels peak-to-peak amplitude > "
        f"{thr:.2f} (Tukey far-out fence Q3+{FENCE_PRIMARY}*IQR of all {sum(len(v) for v in p2p_max.values())} epochs; "
        f"data units as recorded), or if any channel is flat. No EOG channel exists, so no ocular-artifact regression/ICA was applied.")
    zero = tbr_df.loc[tbr_df.TBR_global.isna(), "subject_id"].tolist()
    log(f"Children with ZERO retained epochs (no TBR; excluded from every TBR analysis): {zero if zero else 'none'}")
    tbr_df = tbr_df.dropna(subset=["TBR_global"]).reset_index(drop=True)
    log(f"Epochs: total={tbr_df.n_epochs_total.sum()}, retained={tbr_df.n_epochs_retained.sum()} "
        f"({100 * tbr_df.n_epochs_retained.sum() / tbr_df.n_epochs_total.sum():.1f}%)")
    for g, s_ in tbr_df.groupby("group"):
        log(f"  {g}: duration per child mean={s_.duration_s.mean():.1f}s (SD {s_.duration_s.std():.1f}, range {s_.duration_s.min():.1f}-{s_.duration_s.max():.1f}); "
            f"retained epochs per child mean={s_.n_epochs_retained.mean():.1f} (min {s_.n_epochs_retained.min()}, max {s_.n_epochs_retained.max()}); "
            f"children with <10 retained epochs: {(s_.n_epochs_retained < 10).sum()}")
    tbr_df["fraction_retained"] = tbr_df.n_epochs_retained / tbr_df.n_epochs_total
    rec = tbr_df.groupby("group").agg(n=("subject_id", "count"), duration_mean=("duration_s", "mean"), duration_sd=("duration_s", "std"),
                                       duration_min=("duration_s", "min"), duration_max=("duration_s", "max"),
                                       epochs_total_mean=("n_epochs_total", "mean"), epochs_retained_mean=("n_epochs_retained", "mean"),
                                       epochs_retained_min=("n_epochs_retained", "min"), epochs_retained_max=("n_epochs_retained", "max"))
    rec.to_csv(os.path.join(OUTPUT_DIR, "table_recording_summary_by_group.csv"))
    return tbr_df, tbr_ch_df, psd_store, thr, (eps, p2p_max, flat_flags)

def rejection_sensitivity(subj, ch, prep):
    header("Sensitivity: stricter epoch rejection (Tukey inner fence Q3+1.5*IQR)")
    eps, p2p_max, flat_flags = prep
    df, _, _, thr = compute_tbr(subj, ch, eps, p2p_max, flat_flags, FENCE_STRICT)
    zero = df.loc[df.TBR_global.isna(), "subject_id"].tolist(); df = df.dropna(subset=["TBR_global"])
    log(f"Stricter threshold = {thr:.2f}; epochs retained = {df.n_epochs_retained.sum()} of {df.n_epochs_total.sum()} "
        f"({100 * df.n_epochs_retained.sum() / df.n_epochs_total.sum():.1f}%); children with zero retained epochs: {zero if zero else 'none'}")
    rows = []
    for name, col in MONTAGE_COL.items():
        c = df.loc[df.group == "Control", col].values; a = df.loc[df.group == "ADHD", col].values
        s_ = tbr_group_stats(c, a, B=N_BOOT); s_["montage"] = name; rows.append(s_)
        log(f"  [{name}] n={len(c)}/{len(a)}  MWU p={s_['p_mwu']:.4f}; P(Control>ADHD)={s_['prob_superiority']:.3f} "
            f"[{s_['ps_ci'][0]:.3f}, {s_['ps_ci'][1]:.3f}]; Cliff's delta={s_['cliffs_delta']:.3f}; median diff={s_['median_diff']:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "table1_sensitivity_stricter_rejection.csv"), index=False)

def eeg_tables(tbr_df):
    header("TBR by montage: Table 1 + group statistics")
    cols = dict(MONTAGE_COL)
    t1, st = [], []
    for name, col in cols.items():
        ctrl = tbr_df.loc[tbr_df.group == "Control", col].values
        adhd = tbr_df.loc[tbr_df.group == "ADHD", col].values
        for g, a in (("Control", ctrl), ("ADHD", adhd)):
            d = describe(a); d.update(montage=name, group=g)
            sub = tbr_df[tbr_df.group == g]
            d["retained_epochs_mean"] = sub.n_epochs_retained.mean()
            d["retained_epochs_min"] = sub.n_epochs_retained.min()
            d["retained_epochs_max"] = sub.n_epochs_retained.max()
            t1.append(d)
        s = tbr_group_stats(ctrl, adhd); s["montage"] = name; st.append(s)
        log(f"\n[{name}]  Control n={len(ctrl)} median={np.median(ctrl):.3f} | ADHD n={len(adhd)} median={np.median(adhd):.3f}")
        log(f"  Mann-Whitney U={s['U']:.1f}, p={s['p_mwu']:.4f}")
        log(f"  P(Control>ADHD)={s['prob_superiority']:.3f}  95% CI [{s['ps_ci'][0]:.3f}, {s['ps_ci'][1]:.3f}];  "
            f"Cliff's delta={s['cliffs_delta']:.3f} [{s['cliffs_delta_ci'][0]:.3f}, {s['cliffs_delta_ci'][1]:.3f}]")
        log(f"  Median difference (Control-ADHD)={s['median_diff']:.3f}  95% CI [{s['median_diff_ci'][0]:.3f}, {s['median_diff_ci'][1]:.3f}]")
        log(f"  log-scale Welch t={s['welch_log_t']:.3f}, p={s['welch_log_p']:.4f};  Hedges' g (log)={s['hedges_g_log']:.3f} "
            f"[{s['hedges_g_log_ci'][0]:.3f}, {s['hedges_g_log_ci'][1]:.3f}]")
        log(f"  raw-scale Welch t={s['welch_raw_t']:.3f}, p={s['welch_raw_p']:.4f};  Hedges' g (raw)={s['hedges_g_raw']:.3f}")
    log("\nNOTE: four electrode selections are reported with equal prominence; none was pre-specified as primary and no multiplicity correction is applied.")
    pd.DataFrame(t1).to_csv(os.path.join(OUTPUT_DIR, "table1_tbr_by_montage.csv"), index=False)
    pd.DataFrame(st).to_csv(os.path.join(OUTPUT_DIR, "table1_tbr_group_statistics.csv"), index=False)
    return pd.DataFrame(t1), pd.DataFrame(st)

def extreme_recordings(tbr_df, tbr_ch_df, psd_store, ch):
    header("Extreme recordings")
    q1, q3 = np.percentile(tbr_df.TBR_global, [25, 75]); fence = q3 + 3 * (q3 - q1)
    flagged = tbr_df.loc[tbr_df.TBR_global > fence, "subject_id"].tolist()
    log(f"Far-out rule on global TBR (Q3+3*IQR = {fence:.3f}): flagged = {flagged}")
    top = tbr_df.sort_values("TBR_global", ascending=False).head(6)
    log(top[["subject_id", "group", "TBR_global", "TBR_frontal", "TBR_Cz", "n_epochs_retained", "n_epochs_total"]].round(3).to_string(index=False))
    for pid in top.subject_id.head(3):
        r = tbr_ch_df.loc[pid]
        log(f"  {pid}: highest per-channel TBR -> " + ", ".join(f"{c}={r[c]:.1f}" for c in r.sort_values(ascending=False).index[:4]))
    tbr_ch_df.round(4).to_csv(os.path.join(OUTPUT_DIR, "tbr_per_channel.csv"))
    # PSD figure: top-3 children vs cohort median PSD
    f0 = next(iter(psd_store.values()))[0]
    med = np.median(np.stack([v[1] for v in psd_store.values()]), axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, pid in zip(axes, top.subject_id.head(3)):
        f, P = psd_store[pid]
        for c in range(P.shape[1]): ax.semilogy(f, P[:, c], lw=0.8, alpha=0.7)
        ax.semilogy(f0, med.mean(axis=1), "k--", lw=2, label="cohort median (mean over channels)")
        ax.axvspan(*THETA, color="tab:orange", alpha=0.15); ax.axvspan(*BETA, color="tab:green", alpha=0.15)
        ax.set_xlim(0, 40); ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"{pid} ({tbr_df.loc[tbr_df.subject_id == pid, 'group'].iloc[0]}), global TBR="
                     f"{tbr_df.loc[tbr_df.subject_id == pid, 'TBR_global'].iloc[0]:.1f}")
    axes[0].set_ylabel("PSD (data units$^2$/Hz), one line per channel"); axes[0].legend(fontsize=7)
    save_fig(fig, "fig_extreme_recordings_psd")
    return flagged


# ============================================================ A and simulation
def scale_A(tbr, kind):
    tbr = np.asarray(tbr, float)
    if kind == "percentile":
        return (stats.rankdata(tbr, method="average") - 1) / (len(tbr) - 1)
    if kind == "minmax_raw":
        return (tbr - tbr.min()) / (tbr.max() - tbr.min())
    if kind == "minmax_log_winsor":
        l = np.log(tbr); lo, hi = np.percentile(l, [1, 99]); l = np.clip(l, lo, hi)
        return (l - lo) / (hi - lo)
    raise ValueError(kind)

def degrade(img, A, blur=True, noise=True, bright=True, gains=BASE_GAINS, seed=None):
    """Main pipeline operator. Order: blur -> noise(+clip) -> brightness. Degradation ~ (1-A)."""
    bg, ng, brg = gains
    rng = np.random.RandomState(seed) if seed is not None else np.random
    inv = 1.0 - A
    out = img
    if blur:
        k = int(round(1 + inv * bg)); k = k | 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    x = out.astype(np.float32)
    if noise:
        x = np.clip(x + rng.normal(0, inv * ng, img.shape).astype(np.float32), 0, 255)
    if bright:
        x = x * max(0.3, 1 - inv * brg)
    return np.clip(x, 0, 255).astype(np.uint8)

def metrics(orig, deg):
    s = ssim_fn(orig, deg, **SSIM_KW)
    mse = float(np.mean((orig.astype(np.float32) - deg.astype(np.float32)) ** 2))
    p = PSNR_CAP if mse == 0 else min(psnr_fn(orig, deg, data_range=255), PSNR_CAP)
    return s, mse, p

_STIM = []
def _sim_subject(args):
    i, A, gains, img_idx = args
    S, M, P = [], [], []
    for j in img_idx:
        img = _STIM[j]
        s, m, p = metrics(img, degrade(img, A, gains=gains, seed=1000 + i * 1000 + j))
        S.append(s); M.append(m); P.append(p)
    return np.array(S), np.array(M), np.array(P)

def simulate_cohort(A_vec, orig_idx, gains=BASE_GAINS, img_idx=None, label=""):
    """orig_idx = subject index used in the noise-seed scheme (1000 + i*1000 + j)."""
    img_idx = list(range(len(_STIM))) if img_idx is None else list(img_idx)
    t0 = time.time()
    jobs = [(int(i), float(a), gains, img_idx) for i, a in zip(orig_idx, A_vec)]
    if N_JOBS > 1:
        with mp.get_context("fork").Pool(N_JOBS) as pool:
            res = pool.map(_sim_subject, jobs, chunksize=2)
    else:
        res = [_sim_subject(j) for j in jobs]
    S = np.stack([r[0] for r in res]); M = np.stack([r[1] for r in res]); P = np.stack([r[2] for r in res])
    log(f"  simulated {len(jobs)} children x {len(img_idx)} images [{label}] in {time.time() - t0:.0f}s")
    return S, M, P

def cohort_summary(sub_df, S, M, P, A, tag):
    """Descriptive only. sub_df has 'group'."""
    sc = S.mean(axis=1); pc = P.mean(axis=1); mc = M.mean(axis=1)
    out = dict(tag=tag)
    for g in ("Control", "ADHD"):
        d = describe(sc[sub_df.group.values == g])
        for k, v in d.items(): out[f"SSIM_{g}_{k}"] = v
    c, a = sc[sub_df.group.values == "Control"], sc[sub_df.group.values == "ADHD"]
    out["SSIM_mean_diff_Control_minus_ADHD"] = c.mean() - a.mean()
    out["SSIM_median_diff_Control_minus_ADHD"] = np.median(c) - np.median(a)
    out["SSIM_cliffs_delta"] = cliffs_delta(c, a)
    out["A_cliffs_delta"] = cliffs_delta(A[sub_df.group.values == "Control"], A[sub_df.group.values == "ADHD"])
    out["A_p10"], out["A_p50"], out["A_p90"] = np.percentile(A, [10, 50, 90])
    out["n_A_gt_0.2"] = int((A > 0.2).sum()); out["n_A_gt_0.5"] = int((A > 0.5).sum())
    out["SSIM_p10_p90_across_children"] = tuple(np.percentile(sc, [10, 90]))
    out["pearson_r_A_SSIM"] = stats.pearsonr(A, sc)[0]; out["spearman_rho_A_SSIM"] = stats.spearmanr(A, sc)[0]
    out["n_PSNR_capped_child_image_pairs"] = int((P >= PSNR_CAP).sum())
    out["PSNR_mean_of_per_image"] = pc.mean(); out["PSNR_of_mean_MSE"] = 10 * np.log10(255 ** 2 / mc.mean())
    return out

def fmt_cohort(o):
    log(f"  [{o['tag']}] SSIM Control mean={o['SSIM_Control_mean']:.4f} (SD {o['SSIM_Control_sd']:.4f}, median {o['SSIM_Control_median']:.4f}) | "
        f"ADHD mean={o['SSIM_ADHD_mean']:.4f} (SD {o['SSIM_ADHD_sd']:.4f}, median {o['SSIM_ADHD_median']:.4f})")
    log(f"      diff of means (Control-ADHD)={o['SSIM_mean_diff_Control_minus_ADHD']:.4f}, diff of medians={o['SSIM_median_diff_Control_minus_ADHD']:.4f}, "
        f"Cliff's delta SSIM={o['SSIM_cliffs_delta']:.3f} (A: {o['A_cliffs_delta']:.3f}); A p10/p50/p90="
        f"{o['A_p10']:.3f}/{o['A_p50']:.3f}/{o['A_p90']:.3f}; n(A>0.2)={o['n_A_gt_0.2']}, n(A>0.5)={o['n_A_gt_0.5']}")
    log(f"      r(A,SSIM)={o['pearson_r_A_SSIM']:.4f}, rho={o['spearman_rho_A_SSIM']:.4f} [internal consistency check, no p-values]; "
        f"PSNR-capped pairs={o['n_PSNR_capped_child_image_pairs']}; PSNR mean-of-images={o['PSNR_mean_of_per_image']:.2f} dB vs PSNR-of-mean-MSE={o['PSNR_of_mean_MSE']:.2f} dB")


# =================================================== stimuli (unchanged logic) =
def load_class_index():
    if not os.path.exists(CLASS_INDEX): return {}
    with open(CLASS_INDEX) as f: raw = json.load(f)
    return {wnid: name for _, (wnid, name) in raw.items()}

def mbd_labels(n_sample, seed=RNG_SEED):
    all_files = []
    for root, _, files in os.walk(MINDBIGDATA_DIR):
        for f in files: all_files.append(os.path.relpath(os.path.join(root, f), MINDBIGDATA_DIR))
    all_files = sorted(all_files)
    rng = np.random.RandomState(seed)
    sampled = rng.choice(all_files, size=min(n_sample, len(all_files)), replace=False)
    rows = []
    for fname in sampled:
        label = None
        for pat in LABEL_PATTERNS:
            m = pat.search(fname)
            if m: label = m.group(1); break
        rows.append({"filename": fname, "raw_label": label})
    return pd.DataFrame(rows).dropna(subset=["raw_label"]), all_files

def match_labels(eeg_labels_df):
    avail = set(os.listdir(IMAGES_DIR)); w2n = load_class_index()
    n2w = {v.lower(): k for k, v in w2n.items()}
    rows = []
    for _, r in eeg_labels_df.iterrows():
        lab = r["raw_label"]; m = None
        if lab in avail: m = lab
        elif lab.lower() in n2w and n2w[lab.lower()] in avail: m = n2w[lab.lower()]
        if m is not None:
            rows.append({"filename": r["filename"], "raw_label": lab, "matched_wnid": m, "matched_name": w2n.get(m, m)})
    return pd.DataFrame(rows)

def stimulus_selection():
    header("MindBigData file count, recording format, label-based stimulus selection")
    eeg_labels_df, all_files = mbd_labels(N_EEG_RECORDINGS)
    pat = re.compile(r"MindBigData_Imagenet_Insight_n\d{8}_\d+_\d+_\d+\.csv$")
    n_csv = sum(bool(pat.search(os.path.basename(f))) for f in all_files)
    log(f"Files under MindBigData dir: {len(all_files)}; CSV files matching the recording-name pattern: {n_csv}")
    # recording structure of the sampled files
    nsamp, chans = [], set()
    for rel in eeg_labels_df.filename:
        p = os.path.join(MINDBIGDATA_DIR, rel)
        if not os.path.exists(p): p = os.path.join(MINDBIGDATA_SUBDIR, os.path.basename(rel))
        try:
            with open(p, errors="ignore") as fh:
                lines = [l.strip().split(",") for l in fh if l.strip()]
            chans.add(tuple(l[0] for l in lines)); nsamp.append(min(len(l) - 1 for l in lines))
        except Exception as e:
            log(f"  could not read {p}: {e}")
    if nsamp:
        log(f"Sampled recordings: channel sets={sorted(chans)}; samples per channel min/median/max = "
            f"{min(nsamp)}/{int(np.median(nsamp))}/{max(nsamp)}  (128 Hz => ~{np.median(nsamp)/128:.2f} s)")
    matched = match_labels(eeg_labels_df)
    if len(matched) == 0: raise RuntimeError("Zero label matches.")
    ex = (matched.raw_label.str.match(r"^n\d{8}$")).sum()
    sel = dict(total_files=len(all_files), csv_files=n_csv, sampled=len(eeg_labels_df),
               unique_labels=int(eeg_labels_df.raw_label.nunique()), matched_recordings=len(matched),
               unique_matched_categories=int(matched.matched_wnid.nunique()), exact_wnid=int(ex),
               n_imagenet_class_folders=len(os.listdir(IMAGES_DIR)),
               n_imagenet_images=sum(len(os.listdir(os.path.join(IMAGES_DIR, d))) for d in os.listdir(IMAGES_DIR)
                                      if os.path.isdir(os.path.join(IMAGES_DIR, d))))
    log("Selection summary:", sel)
    pd.Series(sel).to_csv(os.path.join(OUTPUT_DIR, "table2_selection_summary.csv"))
    matched.to_csv(os.path.join(OUTPUT_DIR, "label_matches.csv"), index=False)
    # label-matching bar chart (92 labels vs 86 matched categories)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Unique EEG labels\n(100 recordings)", "Matched image\ncategories"],
           [sel["unique_labels"], sel["unique_matched_categories"]], color=["tab:blue", "tab:orange"])
    ax.set_ylabel("Count"); ax.set_title("Label-based category matching")
    save_fig(fig, "fig_label_matching")
    # sample-size sweep (matching only; PCA part removed)
    sweep = []
    for n in (50, 100, 200, 500):
        e, _ = mbd_labels(n); m = match_labels(e)
        sweep.append(dict(N=n, unique_labels=e.raw_label.nunique(), matched_recordings=len(m),
                          unique_matched_categories=m.matched_wnid.nunique() if len(m) else 0,
                          match_pct=round(100 * len(m) / len(e), 1)))
    sweep = pd.DataFrame(sweep); log("Sample-size sweep (label matching only):\n" + sweep.to_string(index=False))
    sweep.to_csv(os.path.join(OUTPUT_DIR, "table_sample_size_sweep.csv"), index=False)
    return matched, sel

def load_stimuli(matched):
    rng = np.random.RandomState(RNG_SEED); w2n = load_class_index()
    imgs, man = [], []
    for wnid in sorted(matched.matched_wnid.unique()):
        d = os.path.join(IMAGES_DIR, wnid)
        if not os.path.isdir(d): continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpeg", ".jpg", ".png")))
        if not files: continue
        for fname in rng.choice(files, size=min(IMAGES_PER_CATEGORY, len(files)), replace=False):
            bgr = cv2.imread(os.path.join(d, fname))
            if bgr is None: continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]; m = min(h, w)
            crop = rgb[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]
            imgs.append(cv2.resize(crop, (STIMULUS_SIZE, STIMULUS_SIZE), interpolation=cv2.INTER_AREA))
            man.append(dict(wnid=wnid, name=w2n.get(wnid, wnid), filename=fname))
    pd.DataFrame(man).to_csv(os.path.join(OUTPUT_DIR, "stimulus_manifest.csv"), index=False)
    log(f"Stimulus pool: {len(imgs)} images from {len(set(m['wnid'] for m in man))} categories "
        f"(max {IMAGES_PER_CATEGORY} per category; unreadable images skipped)")
    return imgs


# ================================================================ analyses ===
def level_table(A_levels, img_idx=None, gains=BASE_GAINS, seed_fn=None, configs=None):
    """Illustrative transfer-function runs over the image pool at given A levels."""
    img_idx = list(range(len(_STIM))) if img_idx is None else img_idx
    res = {}
    for c, A in enumerate(A_levels):
        S, M, P = [], [], []
        for j in img_idx:
            s, m, p = metrics(_STIM[j], degrade(_STIM[j], A, gains=gains, seed=1000 + 10 * j + c))
            S.append(s); M.append(m); P.append(p)
        res[A] = (np.array(S), np.array(M), np.array(P))
    return res

def transfer_function_tables(A_levels):
    header("Illustrative transfer function over the full image pool")
    res = level_table(A_levels)
    rows = []
    for A, (S, M, P) in res.items():
        for name, v in (("SSIM", S), ("PSNR", P), ("MSE", M)):
            d = describe(v); d.update(A=round(float(A), 4), metric=name); rows.append(d)
        log(f"  A={A:.3f}: SSIM {S.mean():.3f} +/- {S.std(ddof=1):.3f} (range {S.min():.3f}-{S.max():.3f}); "
            f"PSNR {P.mean():.2f} +/- {P.std(ddof=1):.2f} dB; MSE {M.mean():.1f} +/- {M.std(ddof=1):.1f}; "
            f"PSNR of mean MSE={10*np.log10(255**2/M.mean()):.2f} dB")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUTPUT_DIR, "table3_transfer_function_full_pool.csv"), index=False)
    first = [dict(A=round(float(A), 4), SSIM=v[0][0], MSE=v[1][0], PSNR=v[2][0]) for A, v in res.items()]
    pd.DataFrame(first).to_csv(os.path.join(OUTPUT_DIR, "table4_first_image_only.csv"), index=False)
    # separate-panel figure (no PDI)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (k, name, unit) in zip(axes, [(0, "SSIM", ""), (2, "PSNR", " (dB)"), (1, "MSE", "")]):
        m = [res[A][k].mean() for A in A_levels]; sd = [res[A][k].std(ddof=1) for A in A_levels]
        ax.bar([f"{A:.2f}" for A in A_levels], m, yerr=sd, capsize=4, color="tab:blue", alpha=0.8)
        ax.set_xlabel("A (cohort 10th / 50th / 90th percentile)"); ax.set_ylabel(name + unit)
        ax.set_title(f"{name}, mean $\\pm$ SD over {len(_STIM)} images", fontsize=9)
    save_fig(fig, "fig_metrics_panels")
    return res

def ablation_full_pool(A_levels, main_res):
    header("Ablation rerun with the main pipeline over the full image pool")
    cfgs = {"Blur only": (True, False, False), "Blur + brightness": (True, False, True),
            "Blur + noise": (True, True, False), "All operations (main pipeline)": (True, True, True)}
    rows = []
    for c, A in enumerate(A_levels):
        for name, (b, n, br) in cfgs.items():
            S, P = [], []
            for j in range(len(_STIM)):
                s, m, p = metrics(_STIM[j], degrade(_STIM[j], A, blur=b, noise=n, bright=br, seed=1000 + 10 * j + c))
                S.append(s); P.append(p)
            S, P = np.array(S), np.array(P)
            rows.append(dict(A=round(float(A), 4), config=name, SSIM_mean=S.mean(), SSIM_sd=S.std(ddof=1),
                             PSNR_mean=P.mean(), PSNR_sd=P.std(ddof=1)))
            if name.startswith("All"):
                assert np.allclose(S, main_res[A][0]), "ablation 'All' must equal the main pipeline"
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUTPUT_DIR, "table5_ablation_full_pool.csv"), index=False)
    log(df.round(3).to_string(index=False))

def coefficient_sweeps(A_levels, tbr_df, A_primary):
    header("Coefficient sensitivity, each gain separately, full pool")
    log("Method: (a) transfer-function sweep at A = cohort 10th/50th/90th percentile; (b) whole-cohort group comparison for "
        "each coefficient scaled alone (x0.5, x1.5) and all three scaled jointly.")
    modes = {"blur": lambda f: (BASE_GAINS[0] * f, BASE_GAINS[1], BASE_GAINS[2]),
             "noise": lambda f: (BASE_GAINS[0], BASE_GAINS[1] * f, BASE_GAINS[2]),
             "brightness": lambda f: (BASE_GAINS[0], BASE_GAINS[1], BASE_GAINS[2] * f),
             "joint (all three)": lambda f: tuple(g * f for g in BASE_GAINS)}
    rows = []
    for mode, fn in modes.items():
        for f in SWEEP_FACTORS:
            for A in A_levels:
                S = np.array([metrics(_STIM[j], degrade(_STIM[j], A, gains=fn(f), seed=99 + j))[0] for j in range(len(_STIM))])
                rows.append(dict(mode=mode, factor=f, A=round(float(A), 4), SSIM_mean=S.mean(), SSIM_sd=S.std(ddof=1),
                                 SSIM_min=S.min(), SSIM_max=S.max()))
    a = pd.DataFrame(rows); a.to_csv(os.path.join(OUTPUT_DIR, "table7a_sweep_transfer_function.csv"), index=False)
    log("(a) SSIM range (min over factors of mean SSIM .. max) per mode and A:")
    log(a.groupby(["mode", "A"]).SSIM_mean.agg(["min", "max"]).round(3).to_string())
    # (b) group-level
    img_idx = list(range(len(_STIM)))
    if SWEEP_IMAGE_LIMIT: img_idx = img_idx[:SWEEP_IMAGE_LIMIT]; log(f"  NOTE: group sweep uses only first {len(img_idx)} images (REV_SWEEP_IMAGES set)")
    g = tbr_df.reset_index(drop=True); idx = g.seed_idx.values
    out = []
    for mode, fn in modes.items():
        for f in GROUP_SWEEP_FACTORS:
            S, M, P = simulate_cohort(A_primary, idx, gains=fn(f), img_idx=img_idx, label=f"{mode} x{f}")
            o = cohort_summary(g, S, M, P, A_primary, f"{mode} x{f}"); o.update(mode=mode, factor=f); out.append(o); fmt_cohort(o)
    return a, pd.DataFrame(out)


# ================================================================== figures ===
def tbr_figures(tbr_df):
    cols = dict(MONTAGE_COL)
    rng = np.random.RandomState(RNG_SEED)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    for ax, (name, col) in zip(axes, cols.items()):
        data = [tbr_df.loc[tbr_df.group == g, col].values for g in ("Control", "ADHD")]
        ax.boxplot(data, showfliers=False, widths=0.5)
        for k, (d, c) in enumerate(zip(data, ("tab:blue", "tab:red"))):
            ax.scatter(k + 1 + rng.uniform(-0.12, 0.12, len(d)), d, s=12, alpha=0.65, color=c, zorder=3)
        ax.set_yscale("log"); ax.set_xticks([1, 2]); ax.set_xticklabels(["Control", "ADHD"])
        ax.set_ylabel("TBR (log scale)"); ax.set_title(name)
    save_fig(fig, "fig_tbr_boxplot_points")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, (name, col) in zip(axes, cols.items()):
        allv = np.log10(tbr_df[col].values); edges = np.histogram_bin_edges(allv, bins=20)
        for g, c in (("Control", "tab:blue"), ("ADHD", "tab:red")):
            ax.hist(np.log10(tbr_df.loc[tbr_df.group == g, col].values), bins=edges, alpha=0.55, color=c, label=g)
        ax.set_xlabel("log$_{10}$ TBR"); ax.set_ylabel("Number of children"); ax.set_title(name); ax.legend()
    save_fig(fig, "fig_tbr_histogram_common_bins")

def cohort_figure(sub_df, A, ssim):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5)); colors = {"ADHD": "tab:red", "Control": "tab:blue"}
    for g in ("Control", "ADHD"):
        m = sub_df.group.values == g
        axes[0].scatter(A[m], ssim[m], s=18, alpha=0.65, color=colors[g], label=f"{g} (n={m.sum()})")
    axes[0].set_xlabel("A (percentile rank of global TBR)"); axes[0].set_ylabel("Simulated SSIM (mean over image pool)")
    axes[0].legend(); axes[0].set_title("Simulated SSIM vs A (descriptive)")
    rng = np.random.RandomState(RNG_SEED); data = [ssim[sub_df.group.values == g] for g in ("Control", "ADHD")]
    axes[1].boxplot(data, showfliers=False, widths=0.5)
    for k, (d, c) in enumerate(zip(data, ("tab:blue", "tab:red"))): axes[1].scatter(k + 1 + rng.uniform(-0.12, 0.12, len(d)), d, s=12, alpha=0.65, color=c, zorder=3)
    axes[1].set_xticks([1, 2]); axes[1].set_xticklabels(["Control", "ADHD"]); axes[1].set_ylabel("Simulated SSIM")
    axes[1].set_title("Simulated SSIM by group (descriptive)")
    save_fig(fig, "fig_cohort_A_vs_ssim")


# ==================================================================== main ===
def main():
    global _STIM
    t_start = time.time()
    header("Environment / settings")
    C1, C2 = (SSIM_KW["K1"] * 255) ** 2, (SSIM_KW["K2"] * 255) ** 2
    manifest = dict(python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
                    skimage=skimage.__version__, opencv=cv2.__version__, pandas=pd.__version__,
                    ssim_settings={k: (v if not isinstance(v, bool) else bool(v)) for k, v in SSIM_KW.items()},
                    SSIM_C1=C1, SSIM_C2=C2, welch=dict(nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP, window="hann(default)"),
                    bootstrap_B=N_BOOT, seed=RNG_SEED, base_gains=BASE_GAINS, n_jobs=N_JOBS)
    log(json.dumps(manifest, indent=1))

    # ---- EEG
    subj, ch = load_adhd()
    tbr_df, tbr_ch_df, psd_store, thr, prep = eeg_analysis(subj, ch); manifest["epoch_rejection_threshold"] = float(thr)
    t1, st = eeg_tables(tbr_df)
    flagged = extreme_recordings(tbr_df, tbr_ch_df, psd_store, ch)
    tbr_figures(tbr_df)
    rejection_sensitivity(subj, ch, prep)
    if EEG_ONLY:
        tbr_df.to_csv(os.path.join(OUTPUT_DIR, "tbr_per_subject.csv"), index=False)
        manifest["runtime_min"] = round((time.time() - t_start) / 60, 1); manifest["mode"] = "EEG_ONLY"
        with open(os.path.join(OUTPUT_DIR, "run_manifest.json"), "w") as f: json.dump(manifest, f, indent=1, default=str)
        with open(os.path.join(OUTPUT_DIR, "SUMMARY_PASTE_BACK_EEG_ONLY.txt"), "w") as f: f.write("\n".join(_LOG))
        shutil.make_archive(OUTPUT_DIR.rstrip("/") + "_zip", "zip", OUTPUT_DIR)
        log(f"\nDONE (EEG only) in {manifest['runtime_min']} min."); return

    # ---- A scalings
    header("Construction of A")
    tbr_df["flagged_extreme"] = tbr_df.subject_id.isin(flagged)
    scalings = ["percentile", "minmax_log_winsor", "minmax_raw"]
    for k in scalings: tbr_df[f"A_{k}"] = scale_A(tbr_df.TBR_global.values, k)
    A_primary = tbr_df["A_percentile"].values
    # blur saturation statement
    k_vals = np.array([(int(round(1 + (1 - a) * BASE_GAINS[0])) | 1) for a in A_primary])
    noise_sd = (1 - A_primary) * BASE_GAINS[1]; dim = np.maximum(0.3, 1 - (1 - A_primary) * BASE_GAINS[2])
    kv, kc = np.unique(k_vals, return_counts=True)
    log("Primary A = percentile rank. Blur kernel sizes in cohort (px: n children): " + ", ".join(f"{int(a)}: {int(b)}" for a, b in zip(kv, kc)))
    log(f"  noise SD range {noise_sd.min():.2f}-{noise_sd.max():.2f}; brightness factor range {dim.min():.3f}-{dim.max():.3f}; "
        f"floor 0.3 reached: {(dim <= 0.3).sum()} children; blur fixed at 9 px for A<=0.1875: {(A_primary <= 0.1875).sum()} children")
    A_raw = tbr_df["A_minmax_raw"].values
    log(f"Legacy min-max A: p10/p50/p90 = {np.percentile(A_raw, [10, 50, 90]).round(3)}, n(A>0.2)={(A_raw > 0.2).sum()}, "
        f"states with old thresholds 0.40/0.75: High={(A_raw >= .75).sum()}, Moderate={((A_raw >= .4) & (A_raw < .75)).sum()}, Low={(A_raw < .4).sum()}")
    tbr_df.to_csv(os.path.join(OUTPUT_DIR, "tbr_per_subject.csv"), index=False)

    # ---- stimuli
    matched, sel = stimulus_selection()
    _STIM = load_stimuli(matched); manifest["n_stimulus_images"] = len(_STIM)
    manifest["selection"] = sel

    # ---- illustrative transfer function (cohort percentiles of primary A)
    A_levels = [float(x) for x in np.percentile(A_primary, [10, 50, 90])]
    main_res = transfer_function_tables(A_levels)
    ablation_full_pool(A_levels, main_res)

    # ---- cohort simulation for the three scalings (descriptive only)
    header("Cohort-level application (descriptive; no p-values, CIs or fit line)")
    idx = tbr_df.seed_idx.values; coh, store = [], {}
    for k in scalings:
        A = tbr_df[f"A_{k}"].values
        S, M, P = simulate_cohort(A, idx, label=f"A={k}")
        o = cohort_summary(tbr_df, S, M, P, A, f"A={k}"); coh.append(o); fmt_cohort(o); store[k] = (S, M, P)
        if k == "percentile":
            tbr_df["SSIM_percentile"] = S.mean(axis=1); tbr_df["PSNR_percentile"] = P.mean(axis=1); tbr_df["MSE_percentile"] = M.mean(axis=1)
            tbr_df["SSIM_sd_across_images"] = S.std(axis=1, ddof=1)
        if k == "minmax_raw": tbr_df["SSIM_minmax_raw"] = S.mean(axis=1)
    pd.DataFrame(coh).to_csv(os.path.join(OUTPUT_DIR, "table6_cohort_descriptives_by_scaling.csv"), index=False)
    tbr_df.to_csv(os.path.join(OUTPUT_DIR, "per_subject_results.csv"), index=False)
    cohort_figure(tbr_df, A_primary, tbr_df["SSIM_percentile"].values)

    # ---- exclusion analysis (renormalise + rerun)
    header("Exclusion of flagged extreme recordings: renormalise A and rerun")
    if flagged:
        keep = ~tbr_df.subject_id.isin(flagged).values; sub = tbr_df[keep].reset_index(drop=True); ex_rows = []
        s_ex = tbr_group_stats(sub.loc[sub.group == "Control", "TBR_global"].values, sub.loc[sub.group == "ADHD", "TBR_global"].values)
        log(f"  TBR (global) without {flagged}: MWU p={s_ex['p_mwu']:.4f}, P(Control>ADHD)={s_ex['prob_superiority']:.3f}, "
            f"Cliff's delta={s_ex['cliffs_delta']:.3f}, median diff={s_ex['median_diff']:.3f}")
        for k in ("percentile", "minmax_raw"):
            A = scale_A(sub.TBR_global.values, k)
            S, M, P = simulate_cohort(A, sub.seed_idx.values, label=f"excl., A={k}")
            o = cohort_summary(sub, S, M, P, A, f"excluded {len(flagged)}; A={k}"); ex_rows.append(o); fmt_cohort(o)
        pd.DataFrame(ex_rows).to_csv(os.path.join(OUTPUT_DIR, "table_exclusion_rerun.csv"), index=False)
    else:
        log("  No recording exceeded the far-out fence; exclusion analysis not applicable.")

    # ---- coefficient sweeps
    sweep_a, sweep_b = coefficient_sweeps(A_levels, tbr_df, A_primary)
    sweep_b.to_csv(os.path.join(OUTPUT_DIR, "table7b_sweep_group_comparison.csv"), index=False)

    manifest["runtime_min"] = round((time.time() - t_start) / 60, 1)
    with open(os.path.join(OUTPUT_DIR, "run_manifest.json"), "w") as f: json.dump(manifest, f, indent=1, default=str)
    with open(os.path.join(OUTPUT_DIR, "SUMMARY_PASTE_BACK.txt"), "w") as f: f.write("\n".join(_LOG))
    shutil.make_archive(OUTPUT_DIR.rstrip("/") + "_zip", "zip", OUTPUT_DIR)
    log(f"\nDONE in {manifest['runtime_min']} min. Outputs: {OUTPUT_DIR} (zip: {OUTPUT_DIR.rstrip('/')}_zip.zip)")

if __name__ == "__main__":
    main()
