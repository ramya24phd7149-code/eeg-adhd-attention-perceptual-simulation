# EEG-Guided Attention-Adaptive Perceptual Simulation (PONE-D-26-36508)

Author-generated code for the manuscript:

**"An EEG-Guided Computational Framework for Attention-Adaptive Perceptual
Simulation to Explore Attention Variability in ADHD"**
Swarna Ramya, Bommareddy Lokesh — VIT-AP University
PLOS ONE, Manuscript ID: PONE-D-26-36508

This repository contains all author-generated code used to compute the
results reported in the manuscript: EEG spectral feature extraction,
EEG–image semantic matching and verification, the attention-adaptive
perceptual degradation model, and the whole-cohort statistical validation.
It is provided without restriction, as required by PLOS ONE's code-sharing
policy.

## Contents

| File | Purpose |
|---|---|
| `complete_paper_pipeline.py` | Main end-to-end pipeline. Produces every table (1–7) and every author-generated figure (2–5, 7, group-validation, PCA-justification) reported in the manuscript. |
| `requirements.txt` | Python package dependencies. |

## Datasets used (not included in this repository — publicly available elsewhere)

1. **ADHD/Control EEG Dataset** — Nasrabadi et al., IEEE DataPort.
   DOI: [10.21227/rzfh-zn36](https://doi.org/10.21227/rzfh-zn36)
   Kaggle mirror used in this pipeline: `danizo/eeg-dataset-for-adhd`
   (`adhdata.csv`; columns: `ID`, `Class`, plus 19 EEG channel columns).

2. **MindBigData EEG-ImageNet Dataset** — David Vivancos.
   [mindbigdata.com/opendb](https://www.mindbigdata.com/opendb/index.html)
   Kaggle mirror used: `swarnaramya24phd7149/mindbigdata1`
   (subfolder `MindBigData-Imagenet/`; per-file CSVs, no header row, 5
   channels: AF3, AF4, T7, T8, Pz).

3. **Mini-ImageNet Dataset** — introduced by Vinyals et al. (2016),
   *Matching Networks for One Shot Learning*, NeurIPS 29, pp. 3630–3638.
   Kaggle mirror used: `deeptrial/miniimagenet`
   (`ImageNet-Mini/images/<wnid>/*.jpeg`, `imagenet_class_index.json`).

## How to reproduce

The script was written and run inside a **Kaggle Notebook** with all three
datasets attached as Kaggle Datasets (so they resolve under
`/kaggle/input/datasets/...`). To reproduce:

1. Create a new Kaggle Notebook.
2. Add the three datasets above under "Add Input" (their Kaggle slugs are
   given in the table above).
3. Upload/paste `complete_paper_pipeline.py` into the notebook and run it,
   or run it as a script:

   ```bash
   pip install -r requirements.txt
   python complete_paper_pipeline.py
   ```

   If running outside Kaggle, edit the path constants at the top of the
   script (`ADHD_DIR`, `MINDBIGDATA_DIR`, `MINIIMAGENET_DIR`, `OUTPUT_DIR`)
   to point at your local copies of the three datasets.

4. All outputs are written to `OUTPUT_DIR` (`/kaggle/working/revision_outputs_full`
   by default).

## Table / figure → script output mapping

| Manuscript item | Output file |
|---|---|
| Table 1 (spectral attention) | `table1_spectral_attention.txt` |
| Fig 2 (boxplot) | `Figure_Boxplot.pdf` |
| Fig 3 (histogram) | `Figure_Histogram.pdf` |
| Table 2 (EEG representation & matching) | `table2_eeg_representation.txt` |
| Fig 4 (PCA 2D scatter) | `Figure_PCA_Representation.pdf` |
| PCA justification (scree/cumvar) | `pca_component_justification.png` |
| Fig 5 (label matching) | `Figure_Label_Matching.pdf` |
| Table 3 (average quantitative results) | `table3_quantitative_avg.txt` |
| Table 4 (representative image results) | `table4_representative_image.txt` |
| Fig 7 (quantitative metrics bar chart) | `Figure7_Quantitative_Metrics.pdf` |
| Table 5 (ablation study) | `table5_ablation.txt` |
| Table 6 (group-level validation) | `table_group_validation.txt` |
| Fig (group validation scatter + boxplot) | `attention_vs_ssim_group_analysis.png` |
| Table 7 (coefficient sensitivity) | `table_sensitivity.txt` |
| Per-subject simulation results (raw) | `per_subject_simulation_results.csv` |
| Full run summary (all stages) | `group_stats_summary.txt` |

Note: the manuscript's Fig 1 (architecture diagram) and Fig 6 (schematic
degradation illustration) are generated directly in the LaTeX source as
inline TikZ graphics and are not produced by this script, since they are
fully synthetic, author-generated, non-photographic figures.

## Key implementation notes (corrections made during revision)

- **Degradation direction**: `apply_neural_degradation()` degrades
  proportional to `(1 - A)`, i.e. higher attention (`A`) → *less*
  degradation, matching Eqs. 5/6/8 and Algorithm 1 of the manuscript. An
  earlier version of this code had this inverted; the numbers in the
  current manuscript reflect the corrected direction.
- **Channel averaging**: `detect_all_channel_columns()` auto-detects and
  averages *all* numeric EEG channel columns (19 channels), not a frontal
  subset, reproducing the published Table 1 direction and significance.
- **Reproducibility**: the 100-recording MindBigData sample is drawn with
  a fixed random seed (`RNG_SEED = 42`).
- **PDI**: reported everywhere as `PDI = 1 - SSIM` (i.e., explicitly
  derived from SSIM, not an independent metric).

## License

Code is released without restriction for reproducibility purposes, per
PLOS ONE's code-sharing policy. See the manuscript's Data Availability
statement for dataset licensing terms (each dataset carries its own
license from its original repository).

## Citation

If you use this code, please cite the manuscript once published (citation
details to be added upon acceptance) and the manuscript ID
**PONE-D-26-36508** in the interim.
