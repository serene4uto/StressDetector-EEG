"""
Example: How to customize Optuna search space with optimizer hyperparameters
"""

from tune import optimize_hyperparameters, get_search_space

# Example 1: Use default search space (now includes optimizer)
print("=== Default Search Space (with Optimizer) ===")
common_space, model_space = get_search_space('MMEEGNet')
print(f"Common: {common_space}")
print(f"Model-specific: {model_space}")

# Example 2: Custom search space with specific optimizer
print("\n=== Custom Search Space (AdamW Focus) ===")
custom_search_space = {
    'common': {
        'lr': (1e-4, 1e-2),  # Narrower range
        'batch_size': [32, 64],  # Fewer options
        'optimizer': ['AdamW'],  # Only AdamW
    },
    'model': {
        'dropout': (0.3, 0.6),  # Narrower range
        'F1': (8, 12),  # Smaller range
        'F2': (16, 24),  # Smaller range
        'D': (2, 3),  # Fewer options
        'kernel_1': [64, 128],  # Fewer options
        'kernel_2': [16, 32],  # Fewer options
    }
}

# Example 3: Run optimization with custom search space
def run_custom_optimization():
    """Example of running optimization with custom search space"""
    
    # Custom search space for faster exploration
    custom_space = {
        'common': {
            'lr': (1e-4, 1e-2),
            'batch_size': [32, 64],
            'optimizer': ['AdamW'],  # Focus on AdamW
        },
        'model': {
            'dropout': (0.3, 0.6),
            'F1': (8, 12),
            'F2': (16, 24),
            'D': (2, 3),
            'kernel_1': [64, 128],
            'kernel_2': [16, 32],
        }
    }
    
    # Run optimization with custom search space
    study = optimize_hyperparameters(
        model_type='MMEEGNet',
        dataset_type='DEAP',
        n_trials=20,  # Fewer trials for faster exploration
        num_epochs=30,  # Shorter training
        work_dir='./.exp',
        study_name='MMEEGNet_AdamW_optimization',
        custom_search_space=custom_space
    )
    
    return study

# Example 4: Different search spaces for different models
def get_model_specific_spaces():
    """Example of different search spaces for different models"""
    
    # Conservative search space for MMFBCNet (SGD focus)
    fbcnet_space = {
        'common': {
            'lr': (1e-4, 1e-2),
            'batch_size': [32, 64],
            'optimizer': ['SGD'],  # Only SGD
        },
        'model': {
            'num_S': (32, 48),  # Narrower range
            'in_channels': [1],  # Only 1 channel
        }
    }
    
    # Aggressive search space for MMTSCeption (all optimizers)
    tsception_space = {
        'common': {
            'lr': (1e-5, 1e-2),
            'batch_size': [16, 32, 64, 128],
            'optimizer': ['Adam', 'AdamW', 'SGD'],  # All optimizers
        },
        'model': {
            'num_T': (15, 30),  # Wider range
            'num_S': (15, 30),  # Wider range
            'hid_channels': (32, 128),  # Wider range
            'dropout': (0.1, 0.9),  # Wider range
        }
    }
    
    return {
        'MMFBCNet': fbcnet_space,
        'MMTSCeption': tsception_space
    }

# Example 5: Show optimizer-specific hyperparameters
def show_optimizer_hyperparameters():
    """Show what hyperparameters are optimized for each optimizer"""
    
    print("\n=== Optimizer-Specific Hyperparameters ===")
    
    optimizers = {
        'Adam': {
            'lr': 'Learning rate (1e-5 to 1e-2, log scale)',
            'beta1': 'First moment decay (0.8 to 0.99)',
            'beta2': 'Second moment decay (0.9 to 0.9999)',
            'eps': 'Numerical stability (1e-9 to 1e-7, log scale)'
        },
        'AdamW': {
            'lr': 'Learning rate (1e-5 to 1e-2, log scale)',
            'beta1': 'First moment decay (0.8 to 0.99)',
            'beta2': 'Second moment decay (0.9 to 0.9999)',
            'weight_decay': 'Weight decay (1e-5 to 1e-2, log scale)'
        },
        'SGD': {
            'lr': 'Learning rate (1e-5 to 1e-2, log scale)',
            'momentum': 'Momentum (0.8 to 0.99)',
            'weight_decay': 'Weight decay (1e-5 to 1e-2, log scale)'
        }
    }
    
    for optimizer, params in optimizers.items():
        print(f"\n{optimizer}:")
        for param, description in params.items():
            print(f"  {param}: {description}")

if __name__ == "__main__":
    print("Optuna Search Space Examples (with Optimizer Hyperparameters)")
    print("=" * 60)
    
    # Show default search spaces
    for model in ['MMEEGNet', 'MMFBCNet', 'MMTSCeption']:
        print(f"\n{model} default search space:")
        common, model_specific = get_search_space(model)
        print(f"  Common: {common}")
        print(f"  Model-specific: {model_specific}")
    
    # Show optimizer-specific hyperparameters
    show_optimizer_hyperparameters()
    
    print("\n=== Usage Examples ===")
    print("1. Optimize with all optimizers:")
    print("   python tune.py --model MMEEGNet --trials 50")
    
    print("\n2. Focus on AdamW only:")
    print("   # Use custom_search_space with 'optimizer': ['AdamW']")
    
    print("\n3. Quick test with fewer trials:")
    print("   python tune.py --model MMEEGNet --trials 20 --epochs 30")
    
    print("\nTo run custom optimization, uncomment the line below:")
    print("# study = run_custom_optimization()")
