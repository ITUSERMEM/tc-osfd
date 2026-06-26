# TC-OSFD: Target-Calibrated Open-Set Fault Discovery

Reference implementation of the TC-OSFD pipeline for emerging-fault discovery from a fragmented historical fault library under operating-condition shift.

## Method

TC-OSFD is a three-stage pipeline:

1. **Class-Conditional MMD fusion** (`pilot_class_conditional_mmd.py`): aligns two source domains class by class, skipping classes missing from one source, to consolidate a fragmented historical fault library into a unified known-fault feature space.
2. **Target calibration** (`pilot_target_finetune.py`): fine-tunes the fused model on a small set of labeled target known-class samples using cross-entropy plus entropy minimization.
3. **Per-class Mahalanobis thresholding** (`pilot_per_class_threshold.py`, `pilot_idea_inference_variants.py`): models each known class as a Gaussian over source features plus high-confidence target pseudo-labels, and sets a per-class distance threshold to reject emerging faults.

Backbone: 1-D ResNet-18 on 1600-point FFT magnitude spectra, 512-dim features.

## Repository layout

```
tc-osfd-release/
├── src/
│   ├── pilot_target_finetune.py      # entry point: CC-MMD + target FT + per-class threshold (main results)
│   ├── pilot_class_conditional_mmd.py # CC-MMD source training + task enumeration
│   ├── pilot_idea_inference_variants.py # eval_per_class_threshold (main-table H-score)
│   ├── pilot_per_class_threshold.py  # per-class threshold evaluation (TAM/TAM+CCAT/TC-OSFD)
│   ├── pilot_tam_extensions.py       # config CFG, data loaders, Mahalanobis helpers, TAM+CCAT eval
│   ├── resnet1d.py                   # 1-D ResNet-18 backbone
│   └── utils.py                      # MMD loss functions
├── incomplete_health_mmd/
│   ├── __init__.py
│   └── data_loader.py                # load_source_domain: read .mat, FFT, normalize
├── README.md
├── requirements.txt
└── .gitignore
```

## Requirements

Python 3.12+, PyTorch with CUDA.

```
pip install -r requirements.txt
```

Key dependencies: torch, numpy, scipy, scikit-learn.

## Data

The pipeline expects the PU bearing dataset as a single MATLAB file `C-PUdata12.mat` (12 fault classes, loads 6 to 9) placed at the repository root. Classes 0 to 8 are treated as known historical faults and classes 9 to 11 as emerging faults.

Download the PU dataset from the KAt-DataCenter (Lessmeier et al., 2016). The file is about 1.2 GB and is not included in this repository.

## Usage

### Train the CC-MMD fused source model

```bash
cd src
python pilot_class_conditional_mmd.py
```

Trained checkpoints are saved to `../models/notebook_classcond/` as `pu_task<idx>_seed<seed>_<variant>.pth` (variants: `erm`, `global`, `classcond`).

### Run TC-OSFD (main results)

```bash
cd src
python pilot_target_finetune.py
```

By default this runs the PU far-target tasks with target load 9 over seeds 8 to 10. To evaluate all 12 leave-one-load-out tasks, edit `target_tasks` in `main()`:

```python
target_tasks = tasks  # all 12 tasks instead of only target load 9
```

### Evaluate threshold strategies

```bash
python pilot_per_class_threshold.py   # TAM / TAM+CCAT / TC-OSFD comparison
```

## Configuration

All dataset and training constants are defined in `pilot_tam_extensions.py` under `CFG`:

- `loads`: [6, 7, 8, 9] (leave-one-load-out gives 12 tasks)
- `num_known`: 9, `num_unknown`: 3
- `src_type_a` / `src_type_b`: fragmented source label partitions
- `samples_per_class`: 800

GPU index is set by `GPU = 0` at the top of `pilot_tam_extensions.py`; change it or set `CUDA_VISIBLE_DEVICES`.

## Protocol notes

Reported H-scores use an oracle threshold protocol in which the Mahalanobis percentile and CCAT scaling factor are grid-selected on the target test set. A deployable fixed configuration (percentile 95, no CCAT scaling) reaches 94.00 percent average H on PU versus 95.34 percent under the oracle protocol, a 1.34-point gap that confirms the method is largely insensitive to threshold tuning. See the manuscript for details.

## Reference

If you use this code, please cite the accompanying manuscript.
