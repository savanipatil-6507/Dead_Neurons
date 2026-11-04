```markdown
# Dead Neuron Recovery and Conservative Pruning

A PyTorch research project that detects inactive ReLU neurons, attempts to revive them through synaptic stripping, and gradually prunes selected neurons from a CNN trained on CIFAR-100.

The pipeline compares three models:

1. Baseline — trained CNN before intervention.
2. Recovered — baseline after synaptic stripping and fine-tuning.
3. Pruned — recovered model after importance testing, structural pruning, and final fine-tuning.

## Overview

A ReLU neuron may produce zero output across observed inputs, leaving part of a network inactive. This project investigates whether such neurons can be recovered before deciding which ones to remove.

The workflow is:

```text
CIFAR-100
    ↓
Train baseline CNN
    ↓
Detect dead neurons
    ↓
Attempt recovery with synaptic stripping
    ↓
Test neuron importance through ablation
    ↓
Gradually prune selected neurons
    ↓
Fine-tune and evaluate
```

A neuron is considered dead when its post-ReLU output is exactly zero for every image in a fixed calibration set. For convolutional layers, this condition also applies across all spatial positions. This definition is specific to the calibration set and does not prove inactivity on every possible input.

## Methodology

### 1. Baseline training

A small VGG-style CNN without Batch Normalization is trained on CIFAR-100. It contains three convolutional layers and one fully connected hidden layer.

```text
Conv1 → ReLU → Pool
Conv2 → ReLU → Pool
Conv3 → ReLU → Pool
Flatten → FC1 → ReLU → Dropout → FC2
```

The baseline hidden-layer widths are `(64, 128, 256, 512)`.

### 2. Dead-neuron detection

Forward hooks record post-ReLU activations on a fixed set of 5,000 training images. Neurons are tracked by their original layer and index throughout the pipeline.

### 3. Synaptic-stripping recovery

For each initially dead neuron, incoming weights are ranked by their estimated average contribution:

```text
contribution = weight × mean input
```

Weights with the most negative estimated contributions are progressively set to zero until the neuron fires or the configured limit is reached. Stripped weights remain masked during fine-tuning.

An optional reinitialization fallback is available and reported separately. The pipeline also includes a control model fine-tuned without stripping.

### 4. Importance testing

Neurons are temporarily disabled without modifying their weights:

- Remaining dead neurons are tested for validation accuracy and loss changes.
- Revived neurons are kept by default and must satisfy stricter tests to become pruning candidates.
- Group ablations examine collective effects.

### 5. Gradual structural pruning

Eligible neurons are removed in small batches. Each batch is checked jointly, and the smaller model is compared numerically against the corresponding ablated model. Stages that violate validation safety thresholds are rolled back.

### 6. Final evaluation

The pruned model is fine-tuned and compared with the baseline and recovered models. The test set is reserved for the final comparison and is not used to make pruning decisions.

## Dataset

The project uses CIFAR-100 through Torchvision.

| Split | Images | Purpose |
|---|---:|---|
| Training | 45,000 | Model training |
| Validation | 5,000 | Model selection and pruning decisions |
| Calibration | 5,000 | Activity detection and contribution estimation |
| Test | 10,000 | Final evaluation |

The calibration set is a fixed subset of the training split. Normalization statistics are computed from training images only. Training augmentation includes random crops and horizontal flips.

## Results

The following results are from the documented **single-seed run**.

| Model | Parameters | Validation Accuracy | Test Accuracy | Test Loss |
|---|---:|---:|---:|---:|
| Baseline | 2,519,780 | 52.86% | 52.89% | 1.7436 |
| Recovered | 2,519,780 | 53.98% | 54.46% | 1.7021 |
| Pruned | 1,253,257 | 55.14% | 55.74% | 1.6543 |

### Neuron statistics

| Measure | Count |
|---|---:|
| Baseline hidden neurons | 960 |
| Initially dead | 435 |
| Revived after recovery fine-tuning | 263 |
| Still dead after recovery | 172 |
| Newly dead after recovery | 1 |
| Remaining dead pruned | 173 |
| Revived neurons pruned | 77 |
| Total neurons pruned | 250 |
| Hidden neurons remaining | 710 |

The 710 remaining neurons are surviving neurons, not necessarily all active after final fine-tuning.

### Key observations

- The final model reduced parameter count by **50.26%**, corresponding to approximately **2.01× compression**.
- Test accuracy increased by **2.85 percentage points** over the baseline.
- A plain fine-tuning control revived **2** initially dead neurons and achieved **53.10%** validation accuracy, compared with **263** revived neurons and **53.98%** validation accuracy for the recovered model.
- The pruned model received additional fine-tuning, so its accuracy gain cannot be attributed to pruning alone.
- Parameter reduction does not by itself demonstrate an inference-speed or energy improvement.

## Requirements

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- tqdm
- Google Colab or another Python environment

A CUDA-capable GPU is recommended. The documented experiment used a **Tesla T4**.

## Getting Started

### Google Colab

1. Open the project notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**, if available.
3. Run the notebook cells in order.
4. CIFAR-100 will be downloaded through Torchvision.
5. Review the generated tables, graphs, and checkpoints.

### Local Environment

Install the required packages:

```bash
pip install torch torchvision numpy pandas matplotlib tqdm jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Open the project notebook and run its cells in order. Update the configured data and checkpoint paths if needed. CPU execution is possible but may be significantly slower.

## Configuration

The notebook exposes settings through configuration objects, including:

- Random seed and dataset paths.
- Model widths and dropout.
- Training and fine-tuning epochs.
- Calibration-set size.
- Synaptic-stripping fractions.
- Optional reinitialization fallback.
- Ablation and pruning thresholds.
- Pruning batch size and minimum layer width.
- Checkpoint resumption.
- Synthetic-data smoke testing.

The synthetic smoke test uses random data to check the pipeline only; its accuracy numbers are not meaningful.

## Outputs

The notebook saves checkpoints and result artifacts in the configured checkpoint directory.

| Artifact | Description |
|---|---|
| Baseline checkpoint | Selected baseline model |
| Recovered checkpoint | Recovered model after fine-tuning |
| Pruned checkpoint | Final structurally pruned model |
| `summary.csv` | Model-level comparison |
| `layerwise.csv` | Layer-wise neuron statistics |
| `results.png` | General result visualizations |
| `results_pipeline.png` | Recovery and pruning visualizations |
| `final_results.json` | Configuration, metrics, identities, and decisions |

## Reproducibility and Limitations

- Results are based on **one seed, one architecture, and one dataset**.
- Dead neurons are identified on a finite calibration set.
- Repeated use of the validation set may bias model and pruning decisions.
- GPU operations may not be bitwise deterministic.
- No latency, energy, or deployment benchmark is included.
- Results are not directly comparable with published methods evaluated under different experimental conditions.

## Research Background

This project is motivated by:

- **A Demon at Work: Leveraging Neuron Death for Efficient Neural Network Pruning** — uses neuron inactivity to guide pruning.
- **Synaptic Stripping: How Pruning Can Bring Dead Neurons Back to Life** — investigates selectively removing connections to reactivate dead neurons.

This implementation adapts these ideas into a recovery-first, ablation-guided structural pruning pipeline.

## Future Work

- Repeat experiments across multiple seeds.
- Evaluate additional architectures and datasets.
- Add matched-training controls to isolate pruning effects.
- Measure inference latency, memory usage, and energy consumption.
- Compare pruning-only and recovery-only approaches.
- Test different calibration sets and recovery thresholds.
