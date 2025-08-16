# StressDetector-EEG

A comprehensive EEG-based stress detection system using deep learning models. This project implements multiple state-of-the-art neural network architectures for stress classification from EEG signals with advanced hyperparameter optimization, configuration-based training, and comprehensive model benchmarking.

## Features

- **Multiple Models**: EEGNet, FBCNet, TSCeption, and EEGNeX
- **Dataset Support**: DEAP and SAM40 datasets
- **Configuration-Based Training**: JSON configuration files for reproducible experiments
- **GPU Acceleration**: Full CUDA support for faster training
- **Hyperparameter Tuning**: Automated tuning with Optuna
- **Best Model Saving**: Automatic checkpointing of best validation performance
- **Early Stopping**: Configurable early stopping to prevent overfitting
- **Experiment Tracking**: Comprehensive result logging and visualization
- **Model Benchmarking**: Comprehensive performance analysis with speed, memory, and accuracy metrics
- **Modular Design**: Easy to extend with new models and datasets

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Windows/Linux/macOS

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/StressDetector-EEG.git
   cd StressDetector-EEG
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**:
   - Windows:
     ```bash
     .\venv\Scripts\Activate.ps1
     ```
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

4. **Install PyTorch with CUDA support**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Install torch-scatter (Windows users)**:
   ```bash
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
   ```

6. **Install remaining dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Windows-Specific Notes

- The project includes multiprocessing fixes for Windows compatibility
- torch-scatter must be installed separately with the correct CUDA version
- DataLoader workers are set to 0 to avoid multiprocessing issues

## Quick Start

### 1. Training with Pre-Optimized Configurations

The project includes pre-optimized configuration files with best hyperparameters:

```bash
# Train MMTSCeption (98.67% validation accuracy)
python train.py --config configs/train/mmtsception_complete_config.json

# Train MMEEGNet (79.77% validation accuracy)
python train.py --config configs/train/mmeegnet_complete_config.json

# Train MMFBCNet (77.58% validation accuracy)
python train.py --config configs/train/mmfbnet_complete_config.json

# Train MMEEGNeX (81.54% validation accuracy)
python train.py --config configs/train/mmeegnex_complete_config.json
```

### 2. Hyperparameter Tuning

Run automated hyperparameter tuning:

```bash
python tune.py --model MMTSCeption --dataset DEAP --trials 50 --epochs 30
```

### 3. Monitor with Dashboard

Launch Optuna Dashboard to monitor optimization progress:

```bash
optuna-dashboard sqlite:///.exp/optuna_storage/MMTSCeption_DEAP_optimization.db
```

## Usage

### Available Models

- `MMEEGNet`: Standard EEGNet architecture
- `MMFBCNet`: Filter Bank Convolutional Network
- `MMTSCeption`: Temporal-Spatial Convolutional Network
- `MMEEGNeX`: Enhanced EEGNet with residual connections

### Available Datasets

- `DEAP`: Database for Emotion Analysis using Physiological signals (repurposed for stress detection through label mapping)
- `SAM40`: Stress Analysis in Motion dataset

### Configuration-Based Training

The training system uses JSON configuration files that specify all training and model parameters:

#### Configuration Structure
```json
{
  "model_type": "MMTSCeption",
  "dataset_type": "DEAP",
  "batch_size": 128,
  "num_epochs": 200,
  "learning_rate": 0.003795344412190762,
  "work_dir": "./.exp/mmtsception_experiments",
  "experiment_name": "mmtsception_best_params",
  "resume": false,
  "optimizer": {
    "type": "AdamW",
    "lr": 0.003795344412190762,
    "betas": [0.8807374894149752, 0.9475521739040691],
    "weight_decay": 8.7839615436099e-05
  },
  "early_stopping": {
    "enable": true,
    "patience": 15,
    "monitor": "mean_eval_accuracy",
    "min_delta": 0.001
  },
  "model": {
    "num_classes": 3,
    "num_electrodes": 32,
    "sampling_rate": 128,
    "num_T": 15,
    "num_S": 24,
    "hid_channels": 46,
    "dropout": 0.5775407000881697
  }
}
```

#### Training Features
- **Automatic Best Model Saving**: Saves checkpoint with best validation accuracy
- **Early Stopping**: Configurable early stopping with patience and minimum delta
- **Result Logging**: Comprehensive JSON output with metrics and confusion matrix
- **Experiment Organization**: Custom experiment directories and naming

### Hyperparameter Tuning

#### Run Tuning
```bash
python tune.py --model MMTSCeption --dataset DEAP --trials 100 --epochs 30
```

#### Monitor Progress
```bash
# Monitor specific study
optuna-dashboard sqlite:///.exp/optuna_storage/MMTSCeption_DEAP_optimization.db

# Monitor all studies in storage directory
optuna-dashboard .exp/optuna_storage/

# Custom port
optuna-dashboard sqlite:///.exp/optuna_storage/study.db --port 9000
```

### Script Overview

| Script | Purpose | Features |
|--------|---------|----------|
| `train.py` | Model training | Configuration-based training with MMEngine |
| `tune.py` | Hyperparameter tuning | Automated tuning with SQLite storage |
| `benchmark_models.py` | Model benchmarking | Comprehensive performance analysis |

### Model Benchmarking

The project includes a comprehensive benchmarking script that measures all aspects of model performance:

#### **Benchmark Metrics**
- **Accuracy**: Test accuracy from training results
- **Inference Speed**: Throughput (samples/sec) and latency (ms)
- **Memory Usage**: Model size and GPU memory consumption
- **Training Time**: Estimated training duration

#### **Usage Examples**
```bash
# Benchmark all models
python benchmark_models.py --all-models

# Benchmark specific model
python benchmark_models.py --model MMTSCeption

# Generate visualizations
python benchmark_models.py --all-models --visualize

# Use custom output directory
python benchmark_models.py --all-models --output-dir my_benchmark_results

# Compare with README results
python benchmark_models.py --compare-results
```

#### **Output Structure**
```
.benchmark/                          # Default output directory
├── benchmark_report_20250816_225410.json    # Detailed JSON report
├── benchmark_visualization_20250816_225410.png  # Comparison plots
└── ... (timestamped files from previous runs)
```

#### **Benchmark Features**
- **Automatic Configuration Loading**: Uses actual training configurations
- **Trained Weights Loading**: Benchmarks with real trained models
- **Multiple Batch Sizes**: Tests speed across different batch sizes (1, 32, 64, 128)
- **Real-time Analysis**: Measures latency for real-time applications
- **Flexible Output**: Configurable output directories
- **Comprehensive Reporting**: JSON reports with detailed metrics

### Command Line Options

```bash
# Training with configuration
python train.py --config configs/train/mmtsception_complete_config.json

# Override configuration parameters
python train.py --config configs/train/mmtsception_complete_config.json --epochs 50 --batch-size 64

# Disable early stopping
python train.py --config configs/train/mmtsception_complete_config.json --no-early-stopping

# Disable resume from checkpoint
python train.py --config configs/train/mmtsception_complete_config.json --no-resume

# Tuning
python tune.py --help

# Dashboard
optuna-dashboard --help
```

### Key Parameters

**Training Parameters:**
- `--config`: Path to JSON configuration file (required)
- `--model`: Override model type from config
- `--dataset`: Override dataset type from config
- `--epochs`: Override number of epochs from config
- `--batch-size`: Override batch size from config
- `--lr`: Override learning rate from config
- `--work-dir`: Override work directory from config
- `--no-resume`: Disable resume from checkpoint
- `--no-early-stopping`: Disable early stopping

**Tuning Parameters:**
- `--trials`: Number of tuning trials
- `--study-name`: Custom study name
- `--search-space-file`: Custom search space configuration
- `--create-template`: Generate search space template

## Project Structure

```
StressDetector-EEG/
├── configs/train/           # Pre-optimized configurations
│   ├── mmtsception_complete_config.json
│   ├── mmeegnet_complete_config.json
│   ├── mmfbnet_complete_config.json
│   └── mmeegnex_complete_config.json
├── src/
│   ├── models/              # Model implementations
│   │   ├── torcheeg_mmwraper.py  # MMEngine wrappers for torcheeg models
│   │   ├── eegnex.py            # EEGNeX implementation
│   │   └── exp_eegnet.py        # ResEEGNet implementation
│   ├── datasets/            # Dataset loaders
│   │   └── dataset.py           # Custom dataset implementations
│   ├── utils/               # Utilities and metrics
│   │   ├── metrics.py           # MMEngine metrics
│   │   └── transforms.py        # Data transformations
│   └── data/                # Data preprocessing
│       ├── make_dataset.py      # Dataset creation utilities
│       └── preprocess/          # Preprocessing pipelines
├── example/                 # Example notebooks
├── train.py                # Main training script
├── tune.py                 # Hyperparameter optimization
├── benchmark_models.py     # Model benchmarking script
├── requirements.txt        # Dependencies
├── .gitignore              # Git ignore rules
├── .exp/                   # Experiment results and storage
│   ├── optimization_results/    # Best hyperparameters from tuning
│   ├── training_results/        # Training result JSON files
│   └── optuna_storage/          # Optuna database files
├── .benchmark/             # Benchmark results (default output)
│   ├── benchmark_report_*.json  # Detailed benchmark reports
│   └── benchmark_visualization_*.png  # Comparison plots
├── .data/                  # Data storage directory
├── .data_cache/            # Cached data files
├── misc/                   # Miscellaneous files
└── venv/                   # Python virtual environment
```

## Dashboard Features

The Optuna Dashboard provides real-time visualization of tuning progress:

### 📊 Study Overview
- **Best Trial**: Current best performing configuration
- **Study Statistics**: Trial count, completion status
- **Parameter Importance**: Which hyperparameters matter most

### 📈 Tuning History
- **Objective Value Plot**: How accuracy improves over trials
- **Parameter History**: How each parameter evolves
- **Parallel Coordinate Plot**: Multi-dimensional parameter visualization

### 🔍 Trial Details
- **Individual Trial Analysis**: Detailed view of each trial
- **Parameter Distributions**: Histograms of parameter values
- **Correlation Analysis**: How parameters relate to performance

## Results

The models achieve competitive performance on stress detection tasks with comprehensive optimization and benchmarking.

### 🔬 **Experimental Setup**

#### **Hardware Configuration**
- **GPU**: NVIDIA RTX 3060 Ti (8GB VRAM)
- **CPU**: Intel/AMD multi-core processor
- **RAM**: 16GB+ system memory
- **Storage**: SSD for fast data access
- **OS**: Windows 10/11 with CUDA 12.1 support

#### **Dataset Information**
- **Primary Dataset**: DEAP (Database for Emotion Analysis using Physiological signals)
- **Dataset Repurposing**: Originally designed for emotion recognition, repurposed for stress detection through label mapping
- **EEG Configuration**: 32 electrodes, 128 Hz sampling rate
- **Data Split**: Train/Validation/Test split with consistent subject separation
- **Stress Classes**: 3-class classification (Low/Medium/High stress levels)

#### **Training Configuration**
- **Training Epochs**: 200 epochs per model
- **Tuning Trials**: 100 trials per model with 50 epochs each
- **Batch Size**: 128 (production setting)
- **Framework**: PyTorch with MMEngine training framework
- **Tuning**: Optuna with TPE (Tree-structured Parzen Estimator) sampler

### 🏆 **Model Performance Rankings**

All models were tuned using **100 trials with 50 epochs each** on the DEAP dataset.

| Rank | Model | Validation Accuracy | Test Accuracy | Performance Gap |
|------|-------|-------------------|---------------|-----------------|
| **1** | **MMTSCeption** | **98.67%** | **99.01%** | +0.34% |
| **2** | **MMEEGNeX** | **81.54%** | **84.90%** | +3.36% |
| **3** | **MMEEGNet** | **79.77%** | **80.38%** | +0.61% |
| **4** | **MMFBCNet** | **77.58%** | **77.53%** | -0.05% |

### ⚡ **Inference Performance Analysis**

#### **Speed Rankings (Batch Size 128 - Production Setting)**

| Model | Inference Time | Throughput | Speed Efficiency |
|-------|---------------|------------|------------------|
| **MMFBCNet** | **2.1 ms** | **60,523 samples/sec** | ⭐⭐⭐⭐⭐ |
| **MMEEGNet** | **17.4 ms** | **7,360 samples/sec** | ⭐⭐⭐⭐ |
| **MMTSCeption** | **43.5 ms** | **2,943 samples/sec** | ⭐⭐⭐ |
| **MMEEGNeX** | **213.6 ms** | **599 samples/sec** | ⭐⭐ |

#### **Real-time Performance Analysis**
- **Real-time EEG sampling rate**: 128 Hz (128 samples/second)
- **Real-time requirement**: Latency < 1000ms for 128 samples
- **All models** meet real-time requirements
- **MMFBCNet** is most suitable for real-time applications (2.1ms latency)
- **MMTSCeption** provides best accuracy for accuracy-critical applications

#### **Throughput Scaling Analysis**

| Model | Single Sample | Batch 128 | Scaling Efficiency |
|-------|---------------|-----------|-------------------|
| **MMFBCNet** | 3,301 samples/sec | 60,523 samples/sec | ⭐⭐⭐⭐⭐ |
| **MMEEGNet** | 1,139 samples/sec | 7,360 samples/sec | ⭐⭐⭐⭐ |
| **MMTSCeption** | 478 samples/sec | 2,943 samples/sec | ⭐⭐⭐ |
| **MMEEGNeX** | 164 samples/sec | 599 samples/sec | ⭐⭐ |

### 💾 **Resource Utilization**

#### **Memory Consumption**
- **GPU Memory**: ~0.03 MB per inference (negligible across all models)
- **Model Sizes**: 3-11 MB checkpoint files
- **Memory Efficiency**: Excellent across all models

#### **Model Size Comparison**
| Model | Model Size | Memory Efficiency |
|-------|------------|-------------------|
| **MMFBCNet** | ~0.02 MB | ⭐⭐⭐⭐⭐ |
| **MMEEGNet** | ~0.05 MB | ⭐⭐⭐⭐⭐ |
| **MMTSCeption** | ~0.17 MB | ⭐⭐⭐⭐⭐ |
| **MMEEGNeX** | ~0.48 MB | ⭐⭐⭐⭐ |

### ⏱️ **Training Performance**

#### **Training Duration (200 epochs)**
- **MMFBCNet**: ~6 minutes (fastest training)
- **MMEEGNet**: ~51 minutes 
- **MMTSCeption**: ~127 minutes
- **MMEEGNeX**: ~656 minutes (slowest training)
- **Per-Epoch**: Varies significantly by model complexity
- **GPU Utilization**: Excellent (RTX 3060 Ti)
- **Training Consistency**: Very stable across all models



### 📋 **Performance Summary Table**

| Model | Accuracy | Speed | Memory | Training Time |
|-------|----------|-------|--------|---------------|
| **MMTSCeption** | ⭐⭐⭐⭐⭐<br>(99.01%) | ⭐⭐⭐<br>(2,943 samples/sec) | ⭐⭐⭐⭐⭐<br>(0.17 MB) | ⭐⭐⭐<br>(~127 min) |
| **MMEEGNeX** | ⭐⭐⭐⭐<br>(84.90%) | ⭐⭐<br>(599 samples/sec) | ⭐⭐⭐⭐<br>(0.48 MB) | ⭐<br>(~656 min) |
| **MMEEGNet** | ⭐⭐⭐<br>(80.38%) | ⭐⭐⭐⭐<br>(7,360 samples/sec) | ⭐⭐⭐⭐⭐<br>(0.05 MB) | ⭐⭐⭐⭐<br>(~51 min) |
| **MMFBCNet** | ⭐⭐<br>(77.53%) | ⭐⭐⭐⭐⭐<br>(60,523 samples/sec) | ⭐⭐⭐⭐⭐<br>(0.02 MB) | ⭐⭐⭐⭐⭐<br>(~6 min) |

### 🔧 **Tuned Hyperparameters**

### Hyperparameter Tuning Ranges

#### MMTSCeption
- **batch_size**: [32, 64, 128, 256]
- **lr**: [1e-4, 1e-3, 1e-2] (log scale)
- **optimizer**: [Adam, AdamW]
- **beta1**: [0.8, 0.9, 0.95]
- **beta2**: [0.9, 0.95, 0.99]
- **weight_decay**: [1e-5, 1e-4, 1e-3] (log scale)
- **num_T**: [5, 10, 15, 20, 25]
- **num_S**: [10, 15, 20, 25, 30]
- **hid_channels**: [32, 40, 48, 56, 64]
- **dropout**: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#### MMEEGNet
- **batch_size**: [32, 64, 128, 256]
- **lr**: [1e-4, 1e-3, 1e-2] (log scale)
- **optimizer**: [Adam, AdamW]
- **beta1**: [0.8, 0.9, 0.95]
- **beta2**: [0.9, 0.95, 0.99]
- **weight_decay**: [1e-5, 1e-4, 1e-3] (log scale)
- **dropout**: [0.1, 0.2, 0.3, 0.4, 0.5]
- **F1**: [8, 10, 12, 15, 18, 20]
- **F2**: [12, 15, 18, 20, 25, 30]
- **D**: [2, 3, 4]
- **kernel_1**: [64, 96, 128, 160]
- **kernel_2**: [16, 24, 32, 48]

#### MMFBCNet
- **batch_size**: [32, 64, 128, 256]
- **lr**: [1e-4, 1e-3, 1e-2] (log scale)
- **optimizer**: [Adam, AdamW]
- **beta1**: [0.8, 0.9, 0.95]
- **beta2**: [0.9, 0.95, 0.99]
- **eps**: [1e-9, 1e-8, 1e-7] (log scale)
- **num_S**: [32, 40, 48, 56, 64, 72]
- **in_channels**: [1, 2, 4]

#### MMEEGNeX
- **batch_size**: [32, 64, 128, 256]
- **lr**: [1e-4, 1e-3, 1e-2] (log scale)
- **optimizer**: [Adam, AdamW]
- **beta1**: [0.8, 0.9, 0.95]
- **beta2**: [0.9, 0.95, 0.99]
- **weight_decay**: [1e-5, 1e-4, 1e-3] (log scale)
- **F1_1**: [4, 6, 8, 10, 12]
- **F1_2**: [20, 25, 30, 35, 40]
- **F2_1**: [10, 12, 15, 17, 20]
- **F2_2**: [8, 10, 12, 15, 18]
- **dropout**: [0.1, 0.2, 0.3, 0.4, 0.5]
- **D**: [2, 3, 4]
- **kernel_1**: [64, 96, 128, 160]
- **kernel_2**: [8, 12, 16, 20, 24]

### Training Output Example
```
Training completed! Test results: {'accuracy': 0.8819}
Loading best model from .exp/mmtsception_experiments/mmtsception_best_params/best_mean_eval_accuracy_epoch_1.pth
Best model loaded! Final evaluation with best parameters...
Best model test results: {'accuracy': 0.8819}
Results saved to: .exp/training_results/MMTSCeption_DEAP_20250816_170853_results.json
```

## Advanced Usage

### Model Benchmarking
```bash
# Comprehensive benchmark of all models
python benchmark_models.py --all-models --visualize

# Benchmark specific model with custom output
python benchmark_models.py --model MMTSCeption --output-dir single_model_test

# Compare current results with README benchmarks
python benchmark_models.py --compare-results

# Generate detailed report for analysis
python benchmark_models.py --all-models --output-dir detailed_analysis
```

### Custom Configurations
Create custom configuration files for experiments:

```bash
# Copy and modify existing configuration
cp configs/train/mmtsception_complete_config.json my_experiment.json

# Edit my_experiment.json with custom parameters
# Train with custom configuration
python train.py --config my_experiment.json
```

### Multiple Model Comparison
```bash
# Train all models with optimized configurations
python train.py --config configs/train/mmtsception_complete_config.json
python train.py --config configs/train/mmeegnet_complete_config.json
python train.py --config configs/train/mmfbnet_complete_config.json
python train.py --config configs/train/mmeegnex_complete_config.json

# Compare results in .exp/training_results/
# Run comprehensive benchmark
python benchmark_models.py --all-models --visualize
```

### Custom Search Spaces
```bash
# Generate template
python tune.py --model MMEEGNet --create-template --template-output my_config.json

# Use custom search space
python tune.py --model MMEEGNet --search-space-file my_config.json --trials 50
```

### Resume Interrupted Tuning
```bash
# Resume from where it left off
python tune.py --model MMEEGNet --trials 100 --study-name "resume_tuning"
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Activate virtual environment
   ```bash
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```

2. **Configuration file not found**: Check file path
   ```bash
   python train.py --config configs/train/mmtsception_complete_config.json
   ```

3. **CUDA out of memory**: Reduce batch size
   ```bash
   python train.py --config configs/train/mmtsception_complete_config.json --batch-size 32
   ```

4. **Slow training**: Enable GPU acceleration
   ```bash
   # Ensure CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

5. **Dashboard shows no data**: Check storage path
   ```bash
   # List available studies
   ls .exp/optuna_storage/
   
   # Use correct path
   optuna-dashboard sqlite:///.exp/optuna_storage/correct_study.db
   ```

6. **Benchmark script errors**: Check model availability
   ```bash
   # Ensure models have been trained
   ls .exp/*_experiments/
   
   # Check for trained weights
   ls .exp/*_experiments/*_best_params/best_mean_eval_accuracy_*.pth
   ```

7. **Benchmark output directory issues**: Check permissions
   ```bash
   # Use custom output directory if needed
   python benchmark_models.py --all-models --output-dir /path/to/writable/directory
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MMEngine for the training framework
- Optuna for hyperparameter optimization
- TorchEEG for EEG model implementations
- DEAP dataset contributors
- PyTorch for the deep learning framework
- Matplotlib and Seaborn for visualization capabilities
