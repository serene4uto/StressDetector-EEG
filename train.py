"""
General training script for StressDetector-EEG
Follows the notebook pattern but makes it reusable for different models and datasets
"""

import argparse
import os
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
    
    # Register models
    MODELS.register_module(name='MMEEGNet', module=MMEEGNet)
    MODELS.register_module(name='MMFBCNet', module=MMFBCNet)
    MODELS.register_module(name='MMTSCeption', module=MMTSCeption)
    MODELS.register_module(name='MMEEGNeX', module=MMEEGNeX)
    MODELS.register_module(name='MMResEEGNet', module=MMResEEGNet)
    
    # Register metrics
    METRICS.register_module(name='Accuracy', module=Accuracy)
    METRICS.register_module(name='AccuracyWithLoss', module=AccuracyWithLoss)
    METRICS.register_module(name='TrainingAccuracyWithLoss', module=TrainingAccuracyWithLoss)
    METRICS.register_module(name='ComprehensiveTrainingMetrics', module=ComprehensiveTrainingMetrics)
    METRICS.register_module(name='ConfusionMatrix', module=ConfusionMatrix)


def create_model(model_type, **kwargs):
    """Create model instance based on model type"""
    
    model_configs = {
        'MMEEGNet': dict(
            chunk_size=128,
            num_electrodes=32,
            dropout=0.5,
            kernel_1=64,
            kernel_2=16,
            F1=8,
            F2=16,
            D=2,
            num_classes=3
        ),
        'MMFBCNet': dict(
            num_classes=3,
            num_electrodes=32,
            chunk_size=128,
            in_channels=1,
            num_S=32
        ),
        'MMTSCeption': dict(
            num_classes=3,
            num_electrodes=32,
            sampling_rate=128,
            num_T=15,
            num_S=15,
            hid_channels=32,
            dropout=0.5
        ),
        'MMEEGNeX': dict(
            chunk_size=128,
            num_electrodes=32,
            dropout=0.5,
            F1=[8,32],
            F2=[16,8],
            D=2,
            num_classes=3,
            kernel_1=64,
            kernel_2=16
        ),
        'MMResEEGNet': dict(
            chunk_size=151,
            num_electrodes=60,
            F1=8,
            F2=16,
            D=2,
            num_classes=3,
            kernel_1=64,
            kernel_2=16,
            dropout=0.25
        )
    }
    
    config = model_configs.get(model_type, {}).copy()
    config.update(kwargs)  # Override with any provided parameters
    
    if model_type == 'MMEEGNet':
        from src.models.torcheeg_mmwraper import MMEEGNet
        return MMEEGNet(**config).float()
    elif model_type == 'MMFBCNet':
        from src.models.torcheeg_mmwraper import MMFBCNet
        return MMFBCNet(**config).float()
    elif model_type == 'MMTSCeption':
        from src.models.torcheeg_mmwraper import MMTSCeption
        return MMTSCeption(**config).float()
    elif model_type == 'MMEEGNeX':
        from src.models.eegnex import MMEEGNeX
        return MMEEGNeX(**config).float()
    elif model_type == 'MMResEEGNet':
        from src.models.exp_eegnet import MMResEEGNet
        return MMResEEGNet(**config).float()
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
        num_worker=8
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


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=64, num_workers=2):
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


def train_model(
    model_type='MMEEGNet',
    dataset_type='DEAP',
    data_root=None,
    cache_path=None,
    batch_size=64,
    num_epochs=200,
    learning_rate=0.001,
    work_dir='./.exp',
    resume=True,
    **model_kwargs
):
    """Train a model with the given configuration"""
    
    # Register models and metrics
    register_models_and_metrics()
    
    # Create model
    model = create_model(model_type, **model_kwargs)
    num_classes = model_kwargs.get('num_classes', 3)
    
    # Create datasets
    if dataset_type.upper() == 'DEAP':
        if not data_root:
            data_root = '.data/DEAP/data_preprocessed_python-002'
        if not cache_path:
            cache_path = '.data_cache/deap'
        train_dataset, val_dataset, test_dataset = create_deap_dataset(data_root, cache_path)
    elif dataset_type.upper() == 'SAM40':
        if not data_root:
            raise ValueError("data_root must be provided for SAM40 dataset")
        train_dataset, val_dataset, test_dataset = create_sam40_dataset(data_root)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )
    
    # Create work directory
    work_dir = Path(work_dir) / f"{dataset_type.lower()}_stress" / model_type.lower()
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Create runner
    runner = Runner(
        model=model,
        work_dir=str(work_dir),
        train_dataloader=train_loader,
        optim_wrapper=dict(
            optimizer=dict(
                type='Adam', 
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08
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
    )
    
    # Start training
    print(f"Starting training for {model_type} on {dataset_type} dataset...")
    runner.train()
    
    # Test the model
    print(f"Testing {model_type}...")
    test_results = runner.test()
    
    return runner, test_results


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Train EEG stress detection models')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='MMEEGNet',
                       choices=['MMEEGNet', 'MMFBCNet', 'MMTSCeption', 'MMEEGNeX', 'MMResEEGNet'],
                       help='Model type to train')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='DEAP',
                       choices=['DEAP', 'SAM40'],
                       help='Dataset type to use')
    parser.add_argument('--data-root', type=str, 
                       help='Path to dataset root directory')
    parser.add_argument('--cache-path', type=str,
                       help='Path to cache directory (for DEAP)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--work-dir', type=str, default='./.exp',
                       help='Working directory for outputs')
    parser.add_argument('--no-resume', action='store_true',
                       help='Disable resume from checkpoint')
    
    # Model-specific parameters
    parser.add_argument('--chunk-size', type=int, help='Chunk size for EEG data')
    parser.add_argument('--num-electrodes', type=int, help='Number of electrodes')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes')
    
    args = parser.parse_args()
    
    # Prepare model kwargs
    model_kwargs = {'num_classes': args.num_classes}
    if args.chunk_size:
        model_kwargs['chunk_size'] = args.chunk_size
    if args.num_electrodes:
        model_kwargs['num_electrodes'] = args.num_electrodes
    if args.dropout:
        model_kwargs['dropout'] = args.dropout
    
    # Train the model
    runner, test_results = train_model(
        model_type=args.model,
        dataset_type=args.dataset,
        data_root=args.data_root,
        cache_path=args.cache_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        work_dir=args.work_dir,
        resume=not args.no_resume,
        **model_kwargs
    )
    
    print(f"Training completed! Test results: {test_results}")


if __name__ == "__main__":
    main()
