# StressDetector-EEG

A comprehensive EEG-based stress detection system using deep learning models. This project implements multiple state-of-the-art neural network architectures for stress classification from EEG signals with advanced hyperparameter optimization and real-time visualization capabilities.

## Features

- **Multiple Models**: EEGNet, FBCNet, TSCeption, EEGNeX, and ResEEGNet
- **Dataset Support**: DEAP and SAM40 datasets
- **GPU Acceleration**: Full CUDA support for faster training
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Visualization Dashboard**: Real-time monitoring with Optuna Dashboard
- **Experiment Tracking**: TensorBoard and Weights & Biases integration
- **Modular Design**: Easy to extend with new models and datasets
- **Training Metrics**: Real-time accuracy tracking during training

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

### 1. Basic Training
Train a model on the DEAP dataset:
```bash
python train.py --model MMEEGNet --dataset DEAP --epochs 200
```

### 2. Hyperparameter Optimization
Run automated hyperparameter tuning:
```bash
python tune.py --model MMEEGNet --dataset DEAP --trials 50 --epochs 30
```

### 3. Monitor with Dashboard
Launch Optuna Dashboard to monitor optimization progress:
```bash
optuna-dashboard sqlite:///.exp/optuna_storage/MMEEGNet_DEAP_optimization.db
```

## Usage

### Available Models

- `MMEEGNet`: Standard EEGNet architecture
- `MMFBCNet`: Filter Bank Convolutional Network
- `MMTSCeption`: Temporal-Spatial Convolutional Network
- `MMEEGNeX`: Enhanced EEGNet with residual connections
- `MMResEEGNet`: Residual EEGNet architecture

### Available Datasets

- `DEAP`: Database for Emotion Analysis using Physiological signals
- `SAM40`: Stress Analysis in Motion dataset

### Hyperparameter Optimization

#### Run Optimization
Run automated hyperparameter tuning:
```bash
python tune.py --model MMEEGNet --dataset DEAP --trials 100 --epochs 30
```

#### Monitor Progress
Launch Optuna Dashboard to monitor optimization in real-time:
```bash
# Monitor specific study
optuna-dashboard sqlite:///.exp/optuna_storage/MMEEGNet_DEAP_optimization.db

# Monitor all studies in storage directory
optuna-dashboard .exp/optuna_storage/

# Custom port
optuna-dashboard sqlite:///.exp/optuna_storage/study.db --port 9000

# Network accessible
optuna-dashboard sqlite:///.exp/optuna_storage/study.db --host 0.0.0.0 --port 8080
```

### Script Overview

| Script | Purpose | Features |
|--------|---------|----------|
| `train.py` | Basic model training | Single model training with MMEngine |
| `tune.py` | Hyperparameter optimization | Clean, focused optimization with SQLite storage |

### Command Line Options

```bash
# Training
python train.py --help

# Optimization
python tune.py --help

# Dashboard
optuna-dashboard --help
```

### Key Parameters

**Training Parameters:**
- `--model`: Model architecture to use
- `--dataset`: Dataset to train on
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--work-dir`: Output directory for logs and checkpoints

**Optimization Parameters:**
- `--trials`: Number of optimization trials
- `--study-name`: Custom study name
- `--search-space-file`: Custom search space configuration
- `--create-template`: Generate search space template

**Dashboard Parameters:**
- `--port`: Dashboard port (default: 8080)
- `--host`: Dashboard host (default: 127.0.0.1)
- `--server`: Server type (auto, wsgiref, gunicorn)

## Project Structure

```
StressDetector-EEG/
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
├── requirements.txt        # Dependencies
└── example_search_space.json  # Example search space configuration
```

## Dashboard Features

The Optuna Dashboard provides real-time visualization of your optimization progress:

### 📊 Study Overview
- **Best Trial**: Current best performing configuration
- **Study Statistics**: Trial count, completion status
- **Parameter Importance**: Which hyperparameters matter most

### 📈 Optimization History
- **Objective Value Plot**: How accuracy improves over trials
- **Parameter History**: How each parameter evolves
- **Parallel Coordinate Plot**: Multi-dimensional parameter visualization

### 🔍 Trial Details
- **Individual Trial Analysis**: Detailed view of each trial
- **Parameter Distributions**: Histograms of parameter values
- **Correlation Analysis**: How parameters relate to performance

### ⚙️ Parameter Analysis
- **Parameter Importance**: Which parameters have the most impact
- **Parameter Relationships**: How parameters interact
- **Optimization Landscape**: Visualize the search space

## Results

The models achieve competitive performance on stress detection tasks:

- **EEGNet**: ~77% accuracy on DEAP dataset
- **FBCNet**: Enhanced performance with filter bank approach
- **TSCeption**: Temporal-spatial feature learning
- **EEGNeX**: Improved architecture with residual connections

### Example Results
```
Best trial: 0
Best validation accuracy: 77.0378
Best hyperparameters: {
    'batch_size': 32, 'lr': 0.007144, 'optimizer': 'AdamW',
    'beta1': 0.959, 'beta2': 0.961, 'weight_decay': 0.001721,
    'dropout': 0.110, 'F1': 11, 'F2': 14, 'D': 3,
    'kernel_1': 128, 'kernel_2': 16
}
```

## Advanced Usage

### Custom Search Spaces
Create custom search space configurations:
```bash
# Generate template
python tune.py --model MMEEGNet --create-template --template-output my_config.json

# Use custom search space
python tune.py --model MMEEGNet --search-space-file my_config.json --trials 50
```

### Multiple Model Comparison
Compare different models systematically:
```bash
# Run optimizations for different models
python tune.py --model MMEEGNet --trials 30 --study-name "MMEEGNet_v1"
python tune.py --model MMFBCNet --trials 30 --study-name "MMFBCNet_v1"
python tune.py --model MMTSCeption --trials 30 --study-name "MMTSCeption_v1"

# Monitor all studies
optuna-dashboard .exp/optuna_storage/
```

### Resume Interrupted Optimizations
```bash
# Resume from where it left off
python tune.py --model MMEEGNet --trials 100 --study-name "resume_optimization"
```

### Monitor Long Optimizations
```bash
# Start optimization in background
python tune.py --model MMEEGNet --trials 200 --epochs 50 --study-name "long_optimization" &

# Monitor progress with dashboard
optuna-dashboard sqlite:///.exp/optuna_storage/long_optimization.db
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Activate virtual environment
   ```bash
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```

2. **Dashboard won't start**: Check port availability
   ```bash
   optuna-dashboard sqlite:///.exp/optuna_storage/study.db --port 9000
   ```

3. **CUDA out of memory**: Reduce batch size
   ```bash
   python train.py --model MMEEGNet --batch-size 16
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MMEngine for the training framework
- Optuna for hyperparameter optimization
- TorchEEG for EEG model implementations
- DEAP dataset contributors
