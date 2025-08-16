"""
General training script for StressDetector-EEG
Follows the notebook pattern but makes it reusable for different models and datasets
"""

import argparse
import json
import os
import multiprocessing
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from mmengine.runner import Runner
from mmengine.registry import MODELS, METRICS


def register_models_and_metrics():
    """Register all models and metrics for MMEngine"""
    from src.models.torcheeg_mmwraper import MMEEGNet, MMFBCNet, MMTSCeption
    from src.models.eegnex import MMEEGNeX
    from src.models.exp_eegnet import MMResEEGNet
    from src.utils.metrics import (
        Accuracy, AccuracyWithLoss, TrainingAccuracyWithLoss, 
        ComprehensiveTrainingMetrics, ConfusionMatrix
    )
    
    # Register models (with safety check to prevent duplicate registration)
    model_registrations = [
        ('MMEEGNet', MMEEGNet),
        ('MMFBCNet', MMFBCNet),
        ('MMTSCeption', MMTSCeption),
        ('MMEEGNeX', MMEEGNeX),
        ('MMResEEGNet', MMResEEGNet)
    ]
    
    for name, module in model_registrations:
        if name not in MODELS:
            MODELS.register_module(name=name, module=module)
        else:
            print(f"Model {name} already registered, skipping...")
    
    # Register metrics (with safety check to prevent duplicate registration)
    metric_registrations = [
        ('Accuracy', Accuracy),
        ('AccuracyWithLoss', AccuracyWithLoss),
        ('TrainingAccuracyWithLoss', TrainingAccuracyWithLoss),
        ('ComprehensiveTrainingMetrics', ComprehensiveTrainingMetrics),
        ('ConfusionMatrix', ConfusionMatrix)
    ]
    
    for name, module in metric_registrations:
        if name not in METRICS:
            METRICS.register_module(name=name, module=module)
        else:
            print(f"Metric {name} already registered, skipping...")


def load_config(config_path=None, model_type=None):
    """Load complete configuration from JSON file or use defaults"""
    
    # Default configurations for each model
    default_model_configs = {
        'MMEEGNet': {
            'chunk_size': 128,
            'num_electrodes': 32,
            'dropout': 0.5,
            'kernel_1': 64,
            'kernel_2': 16,
            'F1': 8,
            'F2': 16,
            'D': 2,
            'num_classes': 3
        },
        'MMFBCNet': {
            'num_classes': 3,
            'num_electrodes': 32,
            'chunk_size': 128,
            'in_channels': 1,
            'num_S': 32
        },
        'MMTSCeption': {
            'num_classes': 3,
            'num_electrodes': 32,
            'sampling_rate': 128,
            'num_T': 15,
            'num_S': 15,
            'hid_channels': 32,
            'dropout': 0.5
        },
        'MMEEGNeX': {
            'chunk_size': 128,
            'num_electrodes': 32,
            'dropout': 0.5,
            'F1': [8, 32],
            'F2': [16, 8],
            'D': 2,
            'num_classes': 3,
            'kernel_1': 64,
            'kernel_2': 16
        },
        'MMResEEGNet': {
            'chunk_size': 151,
            'num_electrodes': 60,
            'F1': 8,
            'F2': 16,
            'D': 2,
            'num_classes': 3,
            'kernel_1': 64,
            'kernel_2': 16,
            'dropout': 0.25
        }
    }
    
    # Default training configuration
    default_training_config = {
        'model_type': 'MMEEGNet',
        'dataset_type': 'DEAP',
        'data_root': None,
        'cache_path': None,
        'batch_size': 64,
        'num_epochs': 200,
        'learning_rate': 0.001,
        'work_dir': './.exp',
        'experiment_name': None,  # Will use model_type.lower() if not specified
        'resume': True,
        'split_ratios': [0.6, 0.2, 0.2],
        'random_seed': 42,
        'num_workers': 0,
        'early_stopping': {
            'enable': True,
            'patience': 20,
            'monitor': 'mean_eval_accuracy',
            'min_delta': 0.001
        },

    }
    
    # Start with default training config
    config = default_training_config.copy()
    
    # Add default model config if model_type is specified
    if model_type and model_type in default_model_configs:
        config['model'] = default_model_configs[model_type].copy()
    else:
        # Ensure model config exists with defaults
        config['model'] = default_model_configs['MMEEGNet'].copy()
    
    # Load from config file if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        
        # Update training config
        for key in default_training_config.keys():
            if key in file_config:
                config[key] = file_config[key]
        
        # Update model config
        if 'model' in file_config:
            config['model'] = file_config['model'].copy()
        
        print(f"Loaded configuration from {config_path}")
    elif config_path:
        print(f"Warning: Config file {config_path} not found, using default configuration")
    
    return config


def create_model(model_type, model_config):
    """Create model instance based on model type and configuration"""
    
    if model_type == 'MMEEGNet':
        from src.models.torcheeg_mmwraper import MMEEGNet
        return MMEEGNet(**model_config).float()
    elif model_type == 'MMFBCNet':
        from src.models.torcheeg_mmwraper import MMFBCNet
        return MMFBCNet(**model_config).float()
    elif model_type == 'MMTSCeption':
        from src.models.torcheeg_mmwraper import MMTSCeption
        return MMTSCeption(**model_config).float()
    elif model_type == 'MMEEGNeX':
        from src.models.eegnex import MMEEGNeX
        return MMEEGNeX(**model_config).float()
    elif model_type == 'MMResEEGNet':
        from src.models.exp_eegnet import MMResEEGNet
        return MMResEEGNet(**model_config).float()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_deap_dataset(data_root, cache_path, split_ratios=[0.6, 0.2, 0.2], random_seed=42):
    """Create DEAP dataset with train/val/test split"""
    from torcheeg.datasets import DEAPDataset
    from torcheeg import transforms
    from src.utils.transforms import DeapAVToStress
    
    dataset = DEAPDataset(
        io_path=cache_path,
        root_path=data_root,
        online_transform=transforms.Compose([
            transforms.To2d(),
            transforms.ToTensor()
        ]),
        label_transform=transforms.Compose([
            transforms.Select(['arousal','valence']),
            DeapAVToStress(thresholds=[
                [7.5, 2.5],
                [5.0, 5.0]]),
        ]),
        num_worker=0
    )
    
    # Split dataset
    train_size = int(split_ratios[0] * len(dataset))
    val_size = int(split_ratios[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    return train_dataset, val_dataset, test_dataset


def create_sam40_dataset(data_path, split_ratios=[0.6, 0.2, 0.2], random_seed=42):
    """Create SAM40 dataset with train/val/test split"""
    from src.datasets.dataset import ZekiScalarDataset
    
    dataset = ZekiScalarDataset(data_path)
    
    # Split dataset
    train_size = int(split_ratios[0] * len(dataset))
    val_size = int(split_ratios[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=64, num_workers=0):
    """Create dataloaders for train/val/test datasets"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def train_model(config):
    """Train a model with the given configuration"""
    
    # Register models and metrics
    register_models_and_metrics()
    
    # Extract parameters from config
    model_type = config['model_type']
    dataset_type = config['dataset_type']
    data_root = config['data_root']
    cache_path = config['cache_path']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    work_dir = config['work_dir']
    resume = config['resume']
    split_ratios = config['split_ratios']
    random_seed = config['random_seed']
    num_workers = config['num_workers']
    model_config = config['model']
    
    # Create model
    model = create_model(model_type, model_config)
    num_classes = model_config.get('num_classes', 3)
    
    # Create datasets
    if dataset_type.upper() == 'DEAP':
        if not data_root:
            data_root = '.data/DEAP/data_preprocessed_python-002'
        if not cache_path:
            cache_path = '.data_cache/deap'
        train_dataset, val_dataset, test_dataset = create_deap_dataset(
            data_root, cache_path, split_ratios, random_seed
        )
    elif dataset_type.upper() == 'SAM40':
        if not data_root:
            raise ValueError("data_root must be provided for SAM40 dataset")
        train_dataset, val_dataset, test_dataset = create_sam40_dataset(
            data_root, split_ratios, random_seed
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers
    )
    
    # Create work directory
    experiment_name = config.get('experiment_name', model_type.lower())
    work_dir = Path(work_dir) / experiment_name
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Create runner
    # Get optimizer config
    optimizer_config = config.get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'Adam')
    optimizer_lr = optimizer_config.get('lr', learning_rate)
    optimizer_betas = optimizer_config.get('betas', (0.9, 0.999))
    optimizer_eps = optimizer_config.get('eps', 1e-08)
    optimizer_weight_decay = optimizer_config.get('weight_decay', 0.0)
    
    # Get early stopping config
    early_stopping_config = config.get('early_stopping', {})
    custom_hooks = []
    
    if early_stopping_config is not None and early_stopping_config.get('enable', True):
        early_stopping_patience = early_stopping_config.get('patience', 20)
        early_stopping_monitor = early_stopping_config.get('monitor', 'val/accuracy')
        early_stopping_min_delta = early_stopping_config.get('min_delta', 0.001)
        
        custom_hooks.append(dict(
            type='EarlyStoppingHook',
            monitor=early_stopping_monitor,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta
        ))
    
    # Add checkpoint hook to save best model
    custom_hooks.append(dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        save_best='mean_eval_accuracy',
        greater_keys=['mean_eval_accuracy'],
        less_keys=None,
        out_dir=str(work_dir),
        max_keep_ckpts=3
    ))
    
    runner = Runner(
        model=model,
        work_dir=str(work_dir),
        train_dataloader=train_loader,
        optim_wrapper=dict(
            optimizer=dict(
                type=optimizer_type, 
                lr=optimizer_lr,
                betas=optimizer_betas,
                eps=optimizer_eps,
                weight_decay=optimizer_weight_decay
            )
        ),
        train_cfg=dict(
            by_epoch=True, 
            max_epochs=num_epochs, 
            val_interval=1
        ),
        val_dataloader=val_loader,
        val_cfg=dict(),
        val_evaluator=dict(type='AccuracyWithLoss'),
        test_dataloader=test_loader,
        test_cfg=dict(),
        test_evaluator=[
            dict(type='Accuracy'), 
            dict(type='ConfusionMatrix', num_classes=num_classes)
        ],
        visualizer=dict(
            type='Visualizer', 
            vis_backends=[dict(type='TensorboardVisBackend')]
        ),
        resume=resume,
        custom_hooks=custom_hooks
    )
    
    # Start training
    print(f"Starting training for {model_type} on {dataset_type} dataset...")
    runner.train()
    
    # Test the model
    print(f"Testing {model_type}...")
    test_results = runner.test()
    
    # Load best model for final evaluation
    best_model_path = work_dir / 'best_mean_eval_accuracy.pth'
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        runner.load_checkpoint(str(best_model_path))
        print(f"Best model loaded! Final evaluation with best parameters...")
        best_test_results = runner.test()
        print(f"Best model test results: {best_test_results}")
    else:
        print("No best model checkpoint found, using last epoch model")
        best_test_results = test_results
    
    return runner, best_test_results


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Train EEG stress detection models')
    
    # Configuration
    parser.add_argument('--config', type=str, required=True,
                       help='Path to JSON config file containing all parameters')
    
    # Override options (optional, for quick experiments)
    parser.add_argument('--model', type=str,
                       choices=['MMEEGNet', 'MMFBCNet', 'MMTSCeption', 'MMEEGNeX', 'MMResEEGNet'],
                       help='Override model type from config')
    parser.add_argument('--dataset', type=str,
                       choices=['DEAP', 'SAM40'],
                       help='Override dataset type from config')
    parser.add_argument('--batch-size', type=int,
                       help='Override batch size from config')
    parser.add_argument('--epochs', type=int,
                       help='Override number of epochs from config')
    parser.add_argument('--lr', type=float,
                       help='Override learning rate from config')
    parser.add_argument('--work-dir', type=str,
                       help='Override work directory from config')
    parser.add_argument('--no-resume', action='store_true',
                       help='Disable resume from checkpoint (override config)')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Disable early stopping (override config)')

    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.model)
    
    # Override with command line arguments if provided
    if args.model:
        config['model_type'] = args.model
    if args.dataset:
        config['dataset_type'] = args.dataset
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.work_dir:
        config['work_dir'] = args.work_dir
    if args.no_resume:
        config['resume'] = False
    if args.no_early_stopping:
        if 'early_stopping' not in config:
            config['early_stopping'] = {}
        config['early_stopping']['enable'] = False

    
    # Print configuration
    print(f"Training configuration:")
    print(f"  Model: {config['model_type']}")
    print(f"  Dataset: {config['dataset_type']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Work directory: {config['work_dir']}")
    print(f"  Resume: {config['resume']}")
    if config.get('early_stopping', {}).get('enable', False):
        print(f"  Early stopping: Enabled ({config['early_stopping']})")
    else:
        print(f"  Early stopping: Disabled")

    print(f"  Model parameters: {config.get('model', {})}")
    print()
    
    # Train the model
    runner, test_results = train_model(config)
    
    print(f"Training completed! Test results: {test_results}")
    
    # Save results to file
    results_dir = Path(config['work_dir']) / "training_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{config['model_type']}_{config['dataset_type']}_{timestamp}_results.json"
    
    # Prepare results data
    results_data = {
        "model_type": config['model_type'],
        "dataset_type": config['dataset_type'],
        "config": config,
        "test_results": test_results,
        "timestamp": timestamp,
        "work_dir": str(runner.work_dir)
    }
    
    # Save to JSON file
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    # Fix for Windows multiprocessing issues with scikit-learn
    multiprocessing.set_start_method('spawn', force=True)
    main()
