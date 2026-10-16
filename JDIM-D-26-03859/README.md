# Multimodal learning for ankylosing spondylitis classification

Research code accompanying the study **A Multimodal Deep Learning Model Integrating Sacroiliac Joint Radiographs and Clinical Variables for Assisted Diagnosis of Ankylosing Spondylitis: Development and External Validation**.

The study compares image-only ResNet50 and ViT-B/16 models with a multimodal ViT-B/16 model integrating sacroiliac joint radiographs and five clinical variables: age, sex, ESR, hs-CRP, and HLA-B27 status.


## Study overview

| Component | Description |
| --- | --- |
| Development cohort | 1,178 participants: 691 with AS and 487 controls |
| External cohort | 255 participants: 66 with AS and 189 controls |
| Internal evaluation | Ten-fold cross-validation |
| Selected configuration | Multimodal ViT-B/16 with histogram equalization |
| External checkpoint | Fold 7, epoch 156; validation accuracy 93.22% |
| External classification threshold | 0.43, selected by maximizing the Youden index on the fold 7 validation subset and then fixed for external testing |

The preprocessing configuration was selected using mean ten-fold validation accuracy. The final single checkpoint was selected using the highest fold-specific validation accuracy within that configuration. Internal validation folds also informed checkpoint selection; internal estimates should be interpreted with that limitation.

## Main files

| File or directory | Purpose |
| --- | --- |
| `dataloader.py` | Image-only dataset loading and transforms |
| `dataloader_multimodal.py` | Image/clinical dataset loading and fold-specific standardization |
| `multimodal_vit.py` | ViT image encoder, clinical encoder, and fusion classifier |
| `AS/HE.py`, `AS/CLAHE.py` | Histogram equalization and CLAHE preprocessing |
| `AS/Gamma_Correction.py`, `AS/Gaussian_Noise_Augmentation.py` | Additional preprocessing experiments |
| `AS/train-resnet50-10fold.py` | ResNet50 training |
| `AS/train-vit-10fold.py` | Image-only ViT training |
| `AS/train_multimodal-vit.py` | Multimodal ViT training |
| `AS/resnet50_val_summary.py`, `AS/vit_b16_val_summary.py`, `AS/val_summary_multimodal_vit.py` | Fold evaluation and pooled ROC plots |
| `AS/infer_external.py` | External inference and image-to-participant aggregation; see release-status notes |
| `Manuscript-Revision/summarize_cross_validation.py` | Internal performance summaries using mean and sample SD |
| `Manuscript-Revision/external_calibration.py` | External calibration plot and Brier score analysis |
| `Manuscript-Revision/clinical_only_external.py` | Development-trained clinical-only baseline and paired external comparison |

## Environment

The exported environment uses Python 3.11.7, PyTorch 2.2.2, torchvision 0.17.2, and Pillow 10.2.0. Package versions are recorded in `requirements.txt`. The reported GPU environment used CUDA 12.1 and an NVIDIA GeForce RTX 4060. These describe the exported environment, rather than independently verified versions for every historical experiment.

Install dependencies in a dedicated Python environment:

```bash
python -m pip install -r requirements.txt
```

Select a PyTorch installation appropriate for the target CPU/GPU environment if necessary. A clean installation and end-to-end execution have not been independently validated for this release.

## Input data and preprocessing

Patient data are not distributed. Configure local input/output paths at the top of each script before execution. Make the repository root available on `PYTHONPATH` so scripts in subdirectories can import the root-level modules.

### Image-only models

`dataloader.py` expects a **headerless** CSV with two columns, in this order: image path and binary label. Labels are `1` for AS and `0` for controls. Image paths must resolve in the local environment.

### Multimodal model

`dataloader_multimodal.py` expects a CSV with headers:

```text
image_path,label,年龄,esr,crp,性别,b27
```

The model input order is **age, ESR, hs-CRP, sex, HLA-B27**. Age, ESR, and hs-CRP are standardized using means and sample standard deviations from the corresponding training subset. Sex and HLA-B27 are supplied as numeric categorical values without standardization; their encoding must match the training data exactly. The same training statistics and feature order must be used for validation and external inference.

Images are converted to RGB, resized to 256 × 256, center-cropped to 224 × 224, and normalized using ImageNet channel means and standard deviations. Training additionally uses random horizontal flipping and rotation up to 10 degrees. Offline image preprocessing must match the checkpoint's training condition. The supplied gamma script uses gamma = 0.5 with reciprocal exponent 1/gamma = 2.

### External analysis files

External calibration requires one row per participant with `patient_id`, `label`, and `prob_AS`. The clinical-only comparison additionally requires the clinical variables and development data. Participant identifiers must be unique and consistently matched; any public example identifiers must be synthetic.

## Analysis workflow

1. Prepare the local radiographs, clinical variables, and fold-specific CSV files.
2. Configure preprocessing and train/evaluate each model in a separate experiment directory. Several scripts use relative checkpoint directories, so sharing a working directory across experiments can overwrite outputs.
3. Select exactly one intended checkpoint per fold for evaluation. Scripts using the first wildcard match require particular care when multiple checkpoint files are present.
4. Summarize the ten fold-level metrics using `summarize_cross_validation.py`. It reports the arithmetic mean and sample SD (`ddof=1`), not a patient-level confidence interval.
5. Apply the verified external inference implementation using the selected checkpoint and its training-subset statistics. Apply the fixed threshold of 0.43 only when calculating threshold-dependent external metrics.
6. Run external calibration and the clinical-only comparison using locally held participant-level predictions. Paired comparisons use the same resampled participants for both models, without refitting models during the 2,000 bootstrap iterations.

The summary script expects these six filenames in its input directory:

```text
resnet50_original.csv
resnet50_gamma.csv
vit_original.csv
vit_clahe.csv
multimodal_vit_original.csv
multimodal_vit_he.csv
```

Each file must contain ten folds and the metric columns expected by the script. Rename copies of the supplied aggregate files to these names, then run:

```bash
python Manuscript-Revision/summarize_cross_validation.py --input-dir PATH_TO_FOLD_METRICS --output-dir PATH_TO_SUMMARY
```

AUC calculated from pooled validation predictions is distinct from the arithmetic mean of ten fold-specific AUC values; the two need not coincide.

## Model weights

The selected checkpoint is `best_model_epoch156_acc93.22.pth` from the histogram-equalized multimodal fold 7 experiment. Release it together with the exact fold 7 training-subset means, standard deviations, feature order, and categorical encoding information. A complete trained state dictionary contains the trained backbone as well as the clinical branch and classifier. The current inference script must be adjusted to avoid trying to load an empty local initialization-weight path.

Weights and normalization statistics should be linked here when the final release is available. Until then, this repository alone does not support reproducing external inference.

## Data availability and intended use

Patient-level clinical data, radiographs, and individual predictions are not publicly available because permission for external sharing has not been granted. Only appropriately reviewed aggregate results should be published. Generated participant-level prediction files must remain private.

This code is intended for research. The study evaluates retrospective classification; it does not establish clinical utility or equivalence to clinician assessment. External probability calibration and broader validation require further evaluation.


