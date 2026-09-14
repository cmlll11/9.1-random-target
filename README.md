# Hard-sample GAP backdoor probe

This repository is the minimal implementation of the current CIFAR-10
verification experiment. It trains paired clean and BackdoorBench classifiers,
fits the official image-dependent `x + f(x)` GAP generator, selects hard
samples, and compares targeted attacks on clean and backdoored models.

The current protocol uses disjoint data partitions:

- 3,000 training images are reserved only for hard-sample selection;
- 47,000 training images are used to train every classifier and GAP generator;
- candidate hard samples are selected from the held-out 3,000-image partition.

The experiment is launched on the GPU server through:

```bash
env PYTHON_BIN=/path/to/python GPU_ID=0 DATA_ROOT=/path/to/cifar10 \
  bash bash/run_hard_sample_gap.sh
```

The launcher trains 10 clean classifiers, 20 official BackdoorBench
classifiers (BadNet, LF, Blended, and WaNet), and one x+f GAP generator for
each model. It then selects 100 samples that all five selection-clean
generators fail to move to target class 0 and evaluates them on the remaining
clean and qualified backdoor models.

The main outputs are written locally and ignored by Git:

- `artifacts/models/hard_sample_gap/`
- `artifacts/mappings/hard_sample_gap/`
- `artifacts/hard_samples/epsilon4/`
- `reports/hard_sample_gap_model_gates.json`
- `reports/hard_sample_gap_summary.json`
- `reports/hard_sample_gap_per_model.csv`

The official dependencies are kept as Git submodules under
`third_party/BackdoorBench` and `third_party/GAP`. Clone with
`--recurse-submodules`.

## Stage 1A/1B PGD-Probe pilot

The new pilot reuses existing checkpoints and does not train GAP generators.
Stage 1A uses Clean seeds 0 and 1 with Blended and WaNet seeds 0 and 1. It
searches CIFAR-100 with a coarse targeted-PGD epsilon grid, refines the Clean
Top-40 using 100-step PGD with three restarts, and compares the resulting
Top-30 samples on paired Clean/Backdoor models. Stage 1B performs within-model
and leave-one-Clean-model-out Ridge Probe evaluation on Clean seeds 0--2.

Run Stage 1A on the GPU server:

```bash
cd /path/to/9.1
PYTHON_BIN=/path/to/venv/bin/python \
DATA_ROOT=/path/to/data \
MODEL_ROOT=/path/to/9.1/artifacts/models/hard_sample_gap \
GPU_ID=0 \
bash bash/run_stage1a_bridge.sh
```

After Stage 1A, set `STAGE1A_RUN_DIR` to its result directory and run Stage 1B:

```bash
cd /path/to/9.1
STAGE1A_RUN_DIR=/path/to/9.1/results/stage1a_bridge/stage1a_YYYYMMDDTHHMMSSZ \
PYTHON_BIN=/path/to/venv/bin/python \
DATA_ROOT=/path/to/data \
MODEL_ROOT=/path/to/9.1/artifacts/models/hard_sample_gap \
GPU_ID=0 \
bash bash/run_stage1b_probe.sh
```

Stage 1B requires the existing partition metadata at
`$DATA_ROOT/hard_sample_gap/shared/partition.json` for Stage 1B, because the Probe
labels intentionally come from each model's known training samples. Results
are written under `results/stage1a_bridge/` and `results/stage1b_probe/`; no
existing run directory is overwritten.

## CIFAR-100 Probe transfer experiment

The transfer experiment trains a Ridge Probe from 1,000 CIFAR-100 train images
using Clean seeds 3 and 4, then applies it to a disjoint 1,000-image CIFAR-100
test pool on target Clean seeds 0--2. It compares Probe Top-100,
target-margin Top-100, and a refined-PGD reference obtained from coarse Top-300
selection. The primary Clean-only selector comparison is followed by same-seed
Clean versus BadNet, LF, Blended, and WaNet evaluation.

Run it on the GPU server with:

```bash
cd /path/to/9.1
PYTHON_BIN=/path/to/venv/bin/python \
DATA_ROOT=/path/to/data \
MODEL_ROOT=/path/to/9.1/artifacts/models/hard_sample_gap \
BATCH_SIZE=64 \
GPU_ID=0 \
bash bash/run_probe_cifar100_transfer.sh
```

The fine grid is `0.5, 1, 1.5, 2, 3, 4, 8, 16, 32 / 255`. Results are written
under `results/stage1b_cifar100_transfer/`.

## Stage 1C target-free untargeted transfer experiment

Stage 1C trains a target-free Ridge Probe on CIFAR-100 train images with
Clean seeds 3 and 4.  It evaluates Probe, top-1/top-2 margin, and a refined
PGD reference on Clean seeds 0--2, then performs deployment-style independent
selection on Clean, BadNet, LF, Blended, and WaNet models.  Part B also uses a
fixed Random-100 set shared within each seed as the no-selection baseline.

Untargeted success means that the model prediction changes from its original
prediction; no CIFAR-10 target class is supplied.  The grid is
`0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8, 16, 32 / 255`.  The strong pass on
`CoarseTop300` is cached and reused for the refined reference and the other
Part A selectors.  Results are written under
`results/stage1c_cifar100_untargeted/`.

Run it on the GPU server with:

```bash
cd /path/to/9.1
PYTHON_BIN=/home/cml/.conda/envs/mdl-uap/bin/python \
DATA_ROOT=/home/cml/8.11/data \
MODEL_ROOT=/home/cml/8.11/artifacts/models/hard_sample_gap \
BACKDOORBENCH_ROOT=/path/to/9.1/third_party/BackdoorBench \
BATCH_SIZE=64 \
GPU_ID=0 \
bash bash/run_probe_cifar100_untargeted.sh
```

An existing model-gate report can optionally be supplied through
`QUALITY_REPORT=/path/to/hard_sample_gap_model_gates.json`; its native trigger
ASR values are copied into `model_quality.csv` without adding trigger logic to
the untargeted experiment.

## Stage WT wrong-target targeted-PGD pilot

Stage WT tests whether a target-specific targeted attack still separates Clean
and BadNet models when the real BadNet target is unknown.  The default targets
are `0,1,3,7`: target 0 is the known-target positive control and 1, 3, and 7
are fixed wrong-target pilot cases.  Clean seeds 3 and 4 fit one
target-specific Ridge Probe per target on CIFAR-100 train images.  Clean and
BadNet seeds 0--2 independently select Probe Top-100, target-margin Top-100,
and Random-100 samples from their own eligible CIFAR-100 test pools, where
the original prediction is not already the selected target.

The targeted-PGD evaluation uses 100 steps, three random restarts, and the
`0.5, 1, 1.5, 2, 3, 4 / 255` grid.  Results include deployment-style records,
the Clean-Probe paired diagnostic, small-budget ASR gaps, and the fixed
random baseline under `results/stage_wt_wrong_target/`.

Run it on the GPU server with:

```bash
cd /path/to/9.1-random-target
PYTHON_BIN=/home/cml/.conda/envs/mdl-uap/bin/python \
DATA_ROOT=/home/cml/8.11/data \
MODEL_ROOT=/home/cml/8.11/artifacts/models/hard_sample_gap \
BACKDOORBENCH_ROOT=/path/to/9.1-random-target/third_party/BackdoorBench \
QUALITY_REPORT=/home/cml/8.11/reports/hard_sample_gap_model_gates.json \
BATCH_SIZE=64 \
GPU_ID=0 \
bash bash/run_probe_cifar100_wrong_target.sh
```

Set `TARGETS=0,1,3,7` for the pilot.  After the pilot, set
`TARGETS=0,1,2,3,4,5,6,7,8,9` to estimate the fraction of effective wrong
targets on CIFAR-10.

## Stage 1D-WT target-0 Model Zoo trigger-direction mechanism

The mechanism experiment does not load raw checkpoints.  It loads the
registered aliases `clean0`--`clean3`, `badnet0`, `blended0`, `wanet0`,
`inputaware0`, `ssba0`, and `adaptive_blend01` from the shared Model Zoo.
Configure the Model Zoo package and root first:

```bash
pip install -e /home/cml/backdoor-model-zoo
export MODEL_ZOO_ROOT=/home/cml/model_zoo
```

Run the paired target-0 mechanism experiment on the server:

```bash
cd /home/cml/9.1-random-target
tmux new -s stage1d-analysis
PYTHON_BIN=/home/cml/.conda/envs/mdl-uap/bin/python \
MODEL_ZOO_ROOT=/home/cml/model_zoo \
DATA_ROOT=/home/cml/8.11/data \
TRIGGER_ARTIFACT_ROOT=/home/cml/8.11/artifacts/models/stage1d_wt_official \
BACKDOORBENCH_ROOT=/home/cml/9.1-random-target/third_party/BackdoorBench \
SSBA_DECODER_PATH=/home/cml/8.11/data/stage1d_ssba_encoder/checkpoints/stage1d_cifar10_ssba_decoder.pth \
SSBA_REFERENCE_TEST_ARRAY=/home/cml/8.11/data/stage1d_ssba_poisoned/cifar10_ssba_test_b1.npy \
GPU_ID=0 bash bash/run_stage1d_wt_alignment.sh
```

The experiment trains one target-0 Probe using Clean seeds 1--3.  Clean0
selects one shared Top-100.  Every model keeps that Top-100, but samples whose
original prediction already equals target 0 are marked
`ineligible_original_target` and are never counted as PGD successes.  For each
backdoor type, the paired control cohort is selected only from eligible samples
whose official trigger reaches class 0; Clean uses the same images and the same
trigger.

Only 1/255 and 1.5/255 are used for PGD.  Shared-trigger attacks use trigger
prototype/concentration metrics; SSBA and Input-Aware additionally use
same-vs-shuffle metrics.  SSBA PGD continues even when its exact encoder
provenance check reports a small documented uint8 reproduction difference: the
official encoder is still used, and the provenance report is retained.  To
rerun only SSBA after the other attack families are complete, set
`BACKDOOR_GROUPS=ssba`; the SSBA alignment uses same-vs-shuffle metrics.
Results are written under `results/stage1d_target0_trigger_alignment/`.
### Successful-PGD direction concentration

`bash/run_stage1d_concentration.sh` evaluates the fixed Clean0 Probe
Top-100 at `0.75/255`, `1/255`, and `1.25/255` by default.  It uses the
registered Model Zoo aliases and computes `C_adv_success` only among samples
that are eligible and successfully targeted at the same epsilon.  It does not
condition the cohort on trigger activation; this keeps the concentration
analysis separate from the trigger-alignment control cohort.

Set `SELECTION_FILE` to the `selected_targeted_robust_samples.csv` from the
completed target-0 Probe run before launching it.

### CIFAR-10 pixel-space trigger mechanism

`bash/run_stage1d_pixel_trigger_mechanism.sh` is an independent mechanism
experiment.  It does not use Probe scores or a robust-sample selector.  Each
of BadNet, Blended, WaNet, Input-Aware, Adaptive-Blend, and SSBA gets its own
randomly ordered cohort of 100 CIFAR-10 test images.  A selected image must
have a non-zero true label, succeed under target-0 PGD on both Clean0 and the
corresponding backdoor model, and be classified as target 0 by that model
after its own official trigger is applied.

The runner tries `1/255` first and independently switches a backdoor family
to `1.5/255` if that family cannot form a 100-image cohort.  Pixel residuals
are saved per sample, together with Clean/backdoor PGD concentration,
residual-to-trigger cosine similarity, trigger concentration, actual L-infinity
and L2 norms, and the complete Model Zoo provenance.  Results are written to
`results/stage1d_pixel_trigger_mechanism/`.

Example server invocation:

```bash
cd /home/cml/9.1-random-target
PYTHON_BIN=/home/cml/.conda/envs/mdl-uap/bin/python \
MODEL_ZOO_ROOT=/home/cml/model_zoo \
DATA_ROOT=/home/cml/8.11/data \
TRIGGER_ARTIFACT_ROOT=/home/cml/8.11/artifacts/models/stage1d_wt_official \
BACKDOORBENCH_ROOT=/home/cml/9.1-random-target/third_party/BackdoorBench \
GPU_ID=1 BATCH_SIZE=64 \
bash bash/run_stage1d_pixel_trigger_mechanism.sh
```

### CIFAR-10 Probe Top-500 targeted-PGD ASR

`bash/run_probe_cifar10_top500_asr.sh` trains one target-0 Ridge Probe from
Clean1, Clean2, and Clean3 CIFAR-10 train samples. Clean0 scores the complete
CIFAR-10 test split and selects one shared Probe Top-500; the same images are
then evaluated on Clean0, BadNet0, Blended0, WaNet0, Input-Aware0, SSBA0, and
Adaptive-Blend01. No backdoor model participates in Probe fitting or sample
selection. Targeted-PGD ASR is reported at `1/255` and `1.5/255`, with
`eligible` excluding images already predicted as target 0. Results are written
under `results/stage1e_cifar10_probe_top500_asr/`.

Example server invocation:

```bash
cd /home/cml/9.1-random-target
PYTHON_BIN=/home/cml/.conda/envs/mdl-uap/bin/python \
MODEL_ZOO_ROOT=/home/cml/model_zoo \
DATA_ROOT=/home/cml/8.11/data \
GPU_ID=1 BATCH_SIZE=64 TRAIN_COUNT=1000 TOP_K=500 \
bash bash/run_probe_cifar10_top500_asr.sh
```

### Clean0/BadNet0 Probe Top-100 layerwise trigger-path mechanism

`bash/run_stage1d_layerwise_trigger_mechanism.sh` trains the target-0 Ridge
Probe on Clean1, Clean2, and Clean3 CIFAR-10 train samples, scores the
complete CIFAR-10 test split with Clean0, and selects exactly one shared
Probe Top-100. The previous fixed joint-success cohort is not used by this
protocol. Clean0 and BadNet0 run the same target-0 PGD on these 100 selected
images before the layerwise analysis.

The analysis compares feature residuals `h(x_adv)-h(x)` and
`h(T(x))-h(x)` at `pixel`, `conv1`, `layer1`, `layer2`, `layer3`, `layer4`,
and `avgpool`. Convolutional maps are flattened without early pooling. It
writes per-sample alignment, residual norms, pairwise PGD/trigger
concentration, Model Zoo provenance, and three depth curves under
`results/stage1d_layerwise_trigger_mechanism/`.

Example server invocation:

```bash
cd /home/cml/9.1-random-target
PYTHON_BIN=/home/cml/.conda/envs/mdl-uap/bin/python \
MODEL_ZOO_ROOT=/home/cml/model_zoo \
MODEL_ZOO_SOURCE_ROOT=/home/cml/backdoor-model-zoo \
DATA_ROOT=/home/cml/8.11/data \
BACKDOORBENCH_ROOT=/home/cml/9.1-random-target/third_party/BackdoorBench \
GPU_ID=1 BATCH_SIZE=100 \
bash bash/run_stage1d_layerwise_trigger_mechanism.sh
```

The result saves Probe training records and parameters, Clean0 selection
scores, PGD endpoint records, per-sample layerwise alignment and residual
norms, pairwise PGD/trigger concentration, Model Zoo provenance, and the
three depth curves under `results/stage1d_layerwise_trigger_mechanism/`.
The result describes representation-direction alignment and concentration;
it does not establish a literal causal path.
