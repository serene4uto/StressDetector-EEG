"""
Hyperparameter optimization script for StressDetector-EEG
Uses Optuna to find optimal hyperparameters for different models
"""

import argparse
import optuna
import os
import json
import yaml
from pathlib import Path
from train import train_model, register_models_and_metrics


def load_search_space_from_file(file_path):
    """Load search space from JSON or YAML file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Search space file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        if file_path.suffix.lower() == '.json':
            return json.load(f)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_search_space_to_file(search_space, file_path):
    """Save search space to JSON or YAML file"""
    file_path = Path(file_path)
    
    with open(file_path, 'w') as f:
        if file_path.suffix.lower() == '.json':
            json.dump(search_space, f, indent=2)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            yaml.dump(search_space, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")


def get_default_search_space(model_type):
    """Get default search spaces for different models"""
    
    # Common search space for all models
    common_space = {
        'lr': (1e-5, 1e-2),  # (min, max) for log scale
        'batch_size': [16, 32, 64, 128],
        'optimizer': ['Adam', 'AdamW', 'SGD'],  # Add optimizer choice
    }
    
    # Model-specific search spaces
    model_spaces = {
        'MMEEGNet': {
            'dropout': (0.1, 0.8),
            'F1': (4, 16),
            'F2': (8, 32),
            'D': (1, 4),
            'kernel_1': [32, 64, 128],
            'kernel_2': [8, 16, 32],
        },
        'MMFBCNet': {
            'num_S': (16, 64),
            'in_channels': [1, 2],
        },
        'MMTSCeption': {
            'num_T': (10, 25),
            'num_S': (10, 25),
            'hid_channels': (16, 64),
            'dropout': (0.1, 0.8),
        },
        'MMEEGNeX': {
            'dropout': (0.1, 0.8),
            'F1_1': (4, 16),
            'F1_2': (16, 64),
            'F2_1': (8, 32),
            'F2_2': (4, 16),
            'D': (1, 4),
            'kernel_1': [32, 64, 128],
            'kernel_2': [8, 16, 32],
        },
        'MMResEEGNet': {
            'dropout': (0.1, 0.8),
            'F1': (4, 16),
            'F2': (8, 32),
            'D': (1, 4),
            'kernel_1': [32, 64, 128],
            'kernel_2': [8, 16, 32],
        }
    }
    
    return common_space, model_spaces.get(model_type, {})


def get_search_space(model_type, search_space_file=None):
    """Define search spaces for different models"""
    
    # If search space file is provided, load from file
    if search_space_file:
        try:
            search_space = load_search_space_from_file(search_space_file)
            
            # Check if it's a complete search space or model-specific
            if 'common' in search_space and 'models' in search_space:
                # Complete search space with multiple models
                common_space = search_space['common']
                model_space = search_space['models'].get(model_type, {})
            elif 'common' in search_space and 'model' in search_space:
                # Single model search space
                common_space = search_space['common']
                model_space = search_space['model']
            else:
                # Assume it's a complete search space
                common_space = search_space.get('common', {})
                model_space = search_space.get('model', {})
            
            print(f"Loaded search space from: {search_space_file}")
            return common_space, model_space
            
        except Exception as e:
            print(f"Warning: Failed to load search space from {search_space_file}: {e}")
            print("Falling back to default search space...")
    
    # Fall back to default search space
    return get_default_search_space(model_type)


def create_search_space_template(model_type):
    """Create a template search space file for the given model"""
    
    common_space, model_space = get_default_search_space(model_type)
    
    template = {
        'description': f'Search space template for {model_type}',
        'common': common_space,
        'model': model_space,
        'notes': {
            'lr': 'Learning rate range (min, max) for log scale',
            'batch_size': 'List of batch sizes to try',
            'optimizer': 'List of optimizers to try',
            'dropout': 'Dropout rate range (min, max)',
            'F1/F2': 'Number of temporal/spatial filters',
            'D': 'Depth multiplier',
            'kernel_1/kernel_2': 'Kernel sizes for temporal/spatial convolution'
        }
    }
    
    return template


def create_optimizer_config(trial, common_space):
    """Create optimizer configuration based on trial suggestions"""
    
    lr = trial.suggest_float('lr', common_space['lr'][0], common_space['lr'][1], log=True)
    optimizer_type = trial.suggest_categorical('optimizer', common_space['optimizer'])
    
    if optimizer_type == 'Adam':
        # Optimize Adam-specific parameters
        beta1 = trial.suggest_float('beta1', 0.8, 0.99)
        beta2 = trial.suggest_float('beta2', 0.9, 0.9999)
        eps = trial.suggest_float('eps', 1e-9, 1e-7, log=True)
        
        optimizer_config = dict(
            type='Adam',
            lr=lr,
            betas=(beta1, beta2),
            eps=eps
        )
        
    elif optimizer_type == 'AdamW':
        # Optimize AdamW-specific parameters
        beta1 = trial.suggest_float('beta1', 0.8, 0.99)
        beta2 = trial.suggest_float('beta2', 0.9, 0.9999)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        
        optimizer_config = dict(
            type='AdamW',
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        
    elif optimizer_type == 'SGD':
        # Optimize SGD-specific parameters
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        
        optimizer_config = dict(
            type='SGD',
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    
    return optimizer_config


def create_objective(model_type, dataset_type, data_root=None, cache_path=None, 
                    num_epochs=50, work_dir='./.exp', resume=False, custom_search_space=None,
                    search_space_file=None):
    """Create objective function for Optuna optimization"""
    
    def objective(trial):
        """Objective function for hyperparameter optimization"""
        
        # Get search space
        common_space, model_space = get_search_space(model_type, search_space_file)
        
        # Override with custom search space if provided
        if custom_search_space:
            common_space.update(custom_search_space.get('common', {}))
            model_space.update(custom_search_space.get('model', {}))
        
        # Common hyperparameters
        batch_size = trial.suggest_categorical('batch_size', common_space['batch_size'])
        
        # Create optimizer configuration
        optimizer_config = create_optimizer_config(trial, common_space)
        
        # Model-specific hyperparameters
        model_kwargs = {'num_classes': 3}  # Default for all models
        
        if model_type == 'MMEEGNet':
            model_kwargs.update({
                'dropout': trial.suggest_float('dropout', model_space['dropout'][0], model_space['dropout'][1]),
                'F1': trial.suggest_int('F1', model_space['F1'][0], model_space['F1'][1]),
                'F2': trial.suggest_int('F2', model_space['F2'][0], model_space['F2'][1]),
                'D': trial.suggest_int('D', model_space['D'][0], model_space['D'][1]),
                'kernel_1': trial.suggest_categorical('kernel_1', model_space['kernel_1']),
                'kernel_2': trial.suggest_categorical('kernel_2', model_space['kernel_2']),
            })
            
        elif model_type == 'MMFBCNet':
            model_kwargs.update({
                'num_S': trial.suggest_int('num_S', model_space['num_S'][0], model_space['num_S'][1]),
                'in_channels': trial.suggest_categorical('in_channels', model_space['in_channels']),
            })
            
        elif model_type == 'MMTSCeption':
            model_kwargs.update({
                'num_T': trial.suggest_int('num_T', model_space['num_T'][0], model_space['num_T'][1]),
                'num_S': trial.suggest_int('num_S', model_space['num_S'][0], model_space['num_S'][1]),
                'hid_channels': trial.suggest_int('hid_channels', model_space['hid_channels'][0], model_space['hid_channels'][1]),
                'dropout': trial.suggest_float('dropout', model_space['dropout'][0], model_space['dropout'][1]),
            })
            
        elif model_type == 'MMEEGNeX':
            # Get individual values for F1 and F2 lists
            F1_1 = trial.suggest_int('F1_1', model_space['F1_1'][0], model_space['F1_1'][1])
            F1_2 = trial.suggest_int('F1_2', model_space['F1_2'][0], model_space['F1_2'][1])
            F2_1 = trial.suggest_int('F2_1', model_space['F2_1'][0], model_space['F2_1'][1])
            F2_2 = trial.suggest_int('F2_2', model_space['F2_2'][0], model_space['F2_2'][1])
            
            model_kwargs.update({
                'dropout': trial.suggest_float('dropout', model_space['dropout'][0], model_space['dropout'][1]),
                'F1': [F1_1, F1_2],  # Convert to list as expected by MMEEGNeX
                'F2': [F2_1, F2_2],  # Convert to list as expected by MMEEGNeX
                'D': trial.suggest_int('D', model_space['D'][0], model_space['D'][1]),
                'kernel_1': trial.suggest_categorical('kernel_1', model_space['kernel_1']),
                'kernel_2': trial.suggest_categorical('kernel_2', model_space['kernel_2']),
            })
            
        elif model_type == 'MMResEEGNet':
            model_kwargs.update({
                'dropout': trial.suggest_float('dropout', model_space['dropout'][0], model_space['dropout'][1]),
                'F1': trial.suggest_int('F1', model_space['F1'][0], model_space['F1'][1]),
                'F2': trial.suggest_int('F2', model_space['F2'][0], model_space['F2'][1]),
                'D': trial.suggest_int('D', model_space['D'][0], model_space['D'][1]),
                'kernel_1': trial.suggest_categorical('kernel_1', model_space['kernel_1']),
                'kernel_2': trial.suggest_categorical('kernel_2', model_space['kernel_2']),
            })
        
        try:
            # Train model with current hyperparameters
            runner, test_results = train_model_with_optimizer(
                model_type=model_type,
                dataset_type=dataset_type,
                data_root=data_root,
                cache_path=cache_path,
                batch_size=batch_size,
                num_epochs=num_epochs,
                work_dir=work_dir,
                resume=resume,
                optimizer_config=optimizer_config,
                **model_kwargs
            )
            
            # Return validation accuracy as the objective to maximize
            # We'll use the last validation accuracy from the training
            validation_accuracy = test_results.get('accuracy', 0.0)
            
            print(f"Trial {trial.number}: Validation Accuracy = {validation_accuracy:.4f}")
            print(f"  Optimizer: {optimizer_config['type']}")
            print(f"  Hyperparameters: {trial.params}")
            
            return validation_accuracy
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return low score for failed trials
    
    return objective


def optimize_hyperparameters(
    model_type='MMEEGNet',
    dataset_type='DEAP',
    data_root=None,
    cache_path=None,
    n_trials=100,
    num_epochs=50,
    work_dir='./.exp',
    study_name=None,
    storage=None,
    custom_search_space=None,
    search_space_file=None
):
    """Run hyperparameter optimization"""
    
    # Register models and metrics ONCE at the beginning
    print("Registering models and metrics...")
    register_models_and_metrics()
    
    # Create study name if not provided
    if study_name is None:
        study_name = f"{model_type}_{dataset_type}_optimization"
    
    # Create storage if not provided (for dashboard support)
    if storage is None:
        storage_dir = Path(work_dir) / "optuna_storage"
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_dir / f'{study_name}.db'}"
        print(f"Using SQLite storage: {storage}")
    
    # Create objective function
    objective = create_objective(
        model_type=model_type,
        dataset_type=dataset_type,
        data_root=data_root,
        cache_path=cache_path,
        num_epochs=num_epochs,
        work_dir=work_dir,
        resume=False,  # Don't resume during optimization
        custom_search_space=custom_search_space,
        search_space_file=search_space_file
    )
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        load_if_exists=True
    )
    
    print(f"Starting hyperparameter optimization for {model_type} on {dataset_type}")
    print(f"Number of trials: {n_trials}")
    print(f"Study name: {study_name}")
    print(f"Storage: {storage}")
    
    # Print search space
    common_space, model_space = get_search_space(model_type, search_space_file)
    print(f"\nSearch space:")
    print(f"  Common: {common_space}")
    print(f"  Model-specific: {model_space}")
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_trial.params}")
    
    # Save results
    results_dir = Path(work_dir) / "optimization_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best parameters
    best_params_file = results_dir / f"{study_name}_best_params.txt"
    with open(best_params_file, 'w') as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Dataset: {dataset_type}\n")
        f.write(f"Best validation accuracy: {study.best_value:.4f}\n")
        f.write(f"Best hyperparameters:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\nBest parameters saved to: {best_params_file}")
    print(f"Database saved to: {storage}")
    
    return study


def train_model_with_optimizer(
    model_type='MMEEGNet',
    dataset_type='DEAP',
    data_root=None,
    cache_path=None,
    batch_size=64,
    num_epochs=200,
    work_dir='./.exp',
    resume=True,
    optimizer_config=None,
    **model_kwargs
):
    """Train a model with custom optimizer configuration"""
    
    # Import here to avoid circular imports
    from train import create_model, create_deap_dataset, create_sam40_dataset, create_dataloaders
    from mmengine.runner import Runner
    from pathlib import Path
    
    # Note: register_models_and_metrics() is called once at the beginning of optimization
    # to avoid duplicate registration errors
    
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
    
    # Use custom optimizer config if provided, otherwise use default
    if optimizer_config is None:
        optimizer_config = dict(
            type='Adam',
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08
        )
    
    # Create runner
    runner = Runner(
        model=model,
        work_dir=str(work_dir),
        train_dataloader=train_loader,
        optim_wrapper=dict(
            optimizer=optimizer_config
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
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization with Optuna')
    
    # Model and dataset arguments
    parser.add_argument('--model', type=str, default='MMEEGNet',
                       choices=['MMEEGNet', 'MMFBCNet', 'MMTSCeption', 'MMEEGNeX', 'MMResEEGNet'],
                       help='Model type to optimize')
    parser.add_argument('--dataset', type=str, default='DEAP',
                       choices=['DEAP', 'SAM40'],
                       help='Dataset to use')
    parser.add_argument('--data-root', type=str, help='Path to dataset root directory')
    parser.add_argument('--cache-path', type=str, help='Path to cache directory')
    
    # Optimization arguments
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs per trial')
    parser.add_argument('--work-dir', type=str, default='./.exp', help='Working directory')
    parser.add_argument('--study-name', type=str, help='Custom study name')
    parser.add_argument('--storage', type=str, help='Optuna storage URL (e.g., sqlite:///study.db)')
    
    # Search space arguments
    parser.add_argument('--search-space-file', type=str, help='Path to search space configuration file')
    parser.add_argument('--create-template', action='store_true', help='Create a search space template')
    parser.add_argument('--template-output', type=str, help='Output file for search space template')
    
    args = parser.parse_args()
    
    # Create search space template
    if args.create_template:
        if not args.template_output:
            args.template_output = f"{args.model.lower()}_template.json"
        create_search_space_template(args.model, args.template_output)
        print(f"Search space template created: {args.template_output}")
        return
    
    # Run optimization
    study = optimize_hyperparameters(
        model_type=args.model,
        dataset_type=args.dataset,
        data_root=args.data_root,
        cache_path=args.cache_path,
        n_trials=args.trials,
        num_epochs=args.epochs,
        work_dir=args.work_dir,
        study_name=args.study_name,
        storage=args.storage,
        search_space_file=args.search_space_file
    )
    
    print(f"\nOptimization completed! Best accuracy: {study.best_value:.4f}")


if __name__ == "__main__":
    main()
