# EEG-parameterized image degradation: a software demonstration

Code for the article "EEG-parameterized image degradation: a software demonstration using the theta/beta ratio of children with and without ADHD" (PLOS ONE).
The code is a software demonstration. It makes no claim about attention, perception or diagnosis (see the article).

## Data (not included; download from the sources below)
| Data | Source | Used for |
|---|---|---|
| ADHD/Control EEG, 61 + 60 children, 19 channels, 128 Hz | IEEE DataPort, doi 10.21227/rzfh-zn36 (original `.mat` files) | TBR |
| MindBigData EEG-ImageNet (14,012 recordings; 14,013 files including `WordReport-v1.04.txt`) | https://www.mindbigdata.com/opendb/imagenet.html | file names only (ImageNet category labels); no EEG signal is used |
| ImageNet-Mini (1,000 class folders, 3,923 ILSVRC2012 validation images) | Kaggle `deeptrial/miniimagenet` | stimulus images (not redistributed) |

## Files
| File | Purpose |
|---|---|
| `full_pipeline.py` | complete pipeline: EEG processing and TBR (four electrode selections, stricter-rejection sensitivity), stimulus selection, image degradation, distortion measures, cohort-level and sensitivity analyses |
| `requirements.txt` | Python packages (versions as in the article; Python 3.12.12) |

## Run order
1. In a Kaggle notebook (CPU only) attach the three datasets and run `full_pipeline.py`. The folders of the datasets and of the output are set at the top of the script and can be changed with the environment variables `REV_ADHD_DIR`, `REV_MBD_DIR`, `REV_IMNET_DIR` and `REV_OUT`. Output: CSV tables (including the four electrode selections and the stricter-rejection analysis), figures, `run_manifest.json`, `SUMMARY_PASTE_BACK.txt`.

## Settings and seeds
- EEG: 1-40 Hz zero-phase Butterworth (order 4) on the continuous recording, 3-s epochs, epoch rejection by the far-out fence Q3 + 3 IQR of the epoch peak-to-peak amplitude (sensitivity: Q3 + 1.5 IQR), Welch PSD (256-sample Hann segments, 50% overlap), TBR = theta (4-8 Hz) / beta (13-30 Hz) per channel.
- Random seeds: 42 (sampling of the 100 recordings, selection of up to 3 images per category, bootstrap with 10,000 stratified resamples); noise seeds 1000 + 1000 s + j (cohort), 1000 + 10 j + c (illustrative values of A), 99 + j (coefficient sweeps).
- SSIM: scikit-image, 7 x 7 uniform window, data range 255, K1 = 0.01, K2 = 0.03, sample covariance, per RGB channel then averaged.
- Software: Python 3.12.12, NumPy 2.0.2, SciPy 1.16.3, scikit-image 0.25.2, OpenCV 4.13.0, pandas 2.3.3.

## License
MIT (see `LICENSE`).
