#!/usr/bin/env python3
"""
Comprehensive Model Benchmark Script for StressDetector-EEG

This script benchmarks all models based on the metrics compared in README.md:
- Accuracy (validation and test)
- Inference Speed (throughput and latency)
- Memory Consumption (model size and GPU memory)
- Training Time (per epoch and total)

Usage:
    python benchmark_models.py --config configs/train/mmtsception_complete_config.json
    python benchmark_models.py --all-models
    python benchmark_models.py --compare-results
    python benchmark_models.py --output-dir custom_benchmark_results
"""

import torch
import time
import psutil
import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
import sys
sys.path.append('src')

from models.torcheeg_mmwraper import MMTSCeption, MMEEGNet, MMFBCNet
from models.eegnex import MMEEGNeX

class ModelBenchmark:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', output_dir='.benchmark'):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.models = {
            'MMTSCeption': MMTSCeption,
            'MMEEGNet': MMEEGNet,
            'MMFBCNet': MMFBCNet,
            'MMEEGNeX': MMEEGNeX
        }
        
        print(f"📁 Benchmark results will be saved to: {self.output_dir.absolute()}")
        
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def get_gpu_memory_usage(self):
        """Get GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
    
    def create_model_instance(self, model_name, model_config):
        """Create model instance with configuration"""
        model_class = self.models[model_name]
        
        # Load actual model configuration from training results
        config = self.load_model_config_from_training_results(model_name)
        
        # Merge with provided config if any
        if model_config:
            config.update(model_config)
        
        model = model_class(**config)
        
        # Load trained weights if available
        self.load_trained_weights(model, model_name)
        
        return model
    
    def load_model_config_from_training_results(self, model_name):
        """Load actual model configuration from training results"""
        # First try to load from training results
        training_results = self.load_training_results(model_name)
        
        if training_results and 'config' in training_results and 'model' in training_results['config']:
            model_config = training_results['config']['model']
            print(f"  ✅ Loaded model config from training results")
            return model_config
        
        # Fallback to config files if no training results
        print(f"  ⚠️  No training results found for {model_name}, trying config files...")
        return self.load_model_config_from_files(model_name)
    
    def load_model_config_from_files(self, model_name):
        """Load model configuration from config files (fallback)"""
        # Map model names to config files
        config_files = {
            'MMTSCeption': 'configs/train/mmtsception_complete_config.json',
            'MMEEGNet': 'configs/train/mmeegnet_complete_config.json',
            'MMFBCNet': 'configs/train/mmfbnet_complete_config.json',
            'MMEEGNeX': 'configs/train/mmeegnex_complete_config.json'
        }
        
        config_file = config_files.get(model_name)
        if not config_file or not Path(config_file).exists():
            print(f"  ⚠️  Config file {config_file} not found for {model_name}, using defaults")
            return self.get_default_config(model_name)
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Extract model configuration
            model_config = config_data.get('model', {})
            print(f"  ✅ Loaded model config from {config_file}")
            return model_config
            
        except Exception as e:
            print(f"  ⚠️  Failed to load config from {config_file}: {e}, using defaults")
            return self.get_default_config(model_name)
    
    def get_default_config(self, model_name):
        """Get default model configurations"""
        default_configs = {
            'MMTSCeption': {
                'num_classes': 3,
                'num_electrodes': 32,
                'sampling_rate': 128,
                'num_T': 15,
                'num_S': 24,
                'hid_channels': 46,
                'dropout': 0.5775407000881697
            },
            'MMEEGNet': {
                'num_classes': 3,
                'num_electrodes': 32,
                'chunk_size': 128,
                'dropout': 0.5,
                'kernel_1': 64,
                'kernel_2': 16,
                'F1': 15,
                'F2': 19,
                'D': 2
            },
            'MMFBCNet': {
                'num_classes': 3,
                'num_electrodes': 32,
                'chunk_size': 128,
                'in_channels': 1,
                'num_S': 63
            },
            'MMEEGNeX': {
                'num_classes': 3,
                'num_electrodes': 32,
                'chunk_size': 128,
                'dropout': 0.5,
                'F1': [8, 30],
                'F2': [17, 12],
                'D': 2,
                'kernel_1': 64,
                'kernel_2': 16
            }
        }
        return default_configs.get(model_name, {})
    
    def load_trained_weights(self, model, model_name):
        """Load trained weights for the model"""
        # Handle special case for MMFBCNet (directory is mmfbcnet_experiments, not mmfbnet_experiments)
        if model_name == 'MMFBCNet':
            exp_dir = Path(".exp/mmfbcnet_experiments")
        else:
            exp_dir = Path(f".exp/{model_name.lower()}_experiments")
        
        if not exp_dir.exists():
            print(f"  ⚠️  No experiment directory found for {model_name}, using random weights")
            return
        
        # Look for best model checkpoint
        best_params_dir = exp_dir / f"{model_name.lower()}_best_params"
        if not best_params_dir.exists():
            print(f"  ⚠️  No best params directory found for {model_name}, using random weights")
            return
        
        # Look for best model checkpoint in nested directory structure
        nested_dir = best_params_dir / f"{model_name.lower()}_best_params"
        if nested_dir.exists():
            checkpoint_dir = nested_dir
        else:
            checkpoint_dir = best_params_dir
        
        # Find best model checkpoint
        best_checkpoint = None
        for checkpoint_file in checkpoint_dir.glob("best_mean_eval_accuracy_*.pth"):
            best_checkpoint = checkpoint_file
            break
        
        if best_checkpoint is None:
            # Fallback to last checkpoint
            last_checkpoint_file = checkpoint_dir / "last_checkpoint"
            if last_checkpoint_file.exists():
                with open(last_checkpoint_file, 'r') as f:
                    last_checkpoint_path = f.read().strip()
                    if Path(last_checkpoint_path).exists():
                        best_checkpoint = Path(last_checkpoint_path)
        
        if best_checkpoint and best_checkpoint.exists():
            try:
                checkpoint = torch.load(best_checkpoint, map_location='cpu')
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    print(f"  ✅ Loaded trained weights from {best_checkpoint.name}")
                else:
                    print(f"  ⚠️  No state_dict found in {best_checkpoint.name}, using random weights")
            except Exception as e:
                print(f"  ⚠️  Failed to load weights from {best_checkpoint.name}: {e}, using random weights")
        else:
            print(f"  ⚠️  No trained weights found for {model_name}, using random weights")
    
    def benchmark_inference_speed(self, model, model_name, batch_sizes=[1, 32, 64, 128], num_runs=100):
        """Benchmark inference speed for different batch sizes"""
        print(f"\n🔍 Benchmarking {model_name} inference speed...")
        
        # Move model to device
        model = model.to(self.device)
        
        results = {}
        
        for batch_size in batch_sizes:
            # Create dummy input - all models expect 4D input (batch, channels, electrodes, time)
            input_shape = (batch_size, 1, 32, 128)  # (batch, channels, electrodes, time)
            dummy_input = torch.randn(input_shape, dtype=torch.float64, device=self.device)
            dummy_labels = torch.randint(0, 3, (batch_size,), device=self.device)
            
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input, dummy_labels, mode='predict')
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = model(dummy_input, dummy_labels, mode='predict')
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            
            results[batch_size] = {
                'avg_time_ms': avg_time * 1000,
                'throughput_samples_per_sec': throughput,
                'std_time_ms': np.std(times) * 1000
            }
            
            print(f"  Batch {batch_size}: {avg_time*1000:.2f}ms, {throughput:.1f} samples/sec")
        
        return results
    
    def benchmark_memory_usage(self, model, model_name):
        """Benchmark memory usage"""
        print(f"\n💾 Benchmarking {model_name} memory usage...")
        
        # Get model size
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        # Get GPU memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_gpu_memory = self.get_gpu_memory_usage()
            
            # Move model to GPU and measure
            model = model.to(self.device)
            after_load_gpu_memory = self.get_gpu_memory_usage()
            
            # Test inference memory
            dummy_input = torch.randn(1, 1, 32, 128, dtype=torch.float64, device=self.device)
            dummy_labels = torch.randint(0, 3, (1,), device=self.device)
            
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input, dummy_labels, mode='predict')
            
            after_inference_gpu_memory = self.get_gpu_memory_usage()
            
            gpu_memory_usage = after_inference_gpu_memory - initial_gpu_memory
        else:
            gpu_memory_usage = 0
        
        print(f"  Model size: {model_size_mb:.2f} MB")
        print(f"  GPU memory usage: {gpu_memory_usage:.2f} MB")
        
        return {
            'model_size_mb': model_size_mb,
            'gpu_memory_usage_mb': gpu_memory_usage
        }
    
    def load_training_results(self, model_name):
        """Load training results from experiment files"""
        print(f"\n📊 Loading training results for {model_name}...")
        
        # Look for training result files
        # Handle special case for MMFBCNet (directory is mmfbcnet_experiments, not mmfbnet_experiments)
        if model_name == 'MMFBCNet':
            exp_dir = Path(".exp/mmfbcnet_experiments")
        else:
            exp_dir = Path(f".exp/{model_name.lower()}_experiments")
        if not exp_dir.exists():
            print(f"  No experiment directory found for {model_name}")
            return None
        
        # Find training results
        training_results_dir = exp_dir / "training_results"
        if not training_results_dir.exists():
            print(f"  No training results directory found for {model_name}")
            return None
        
        # Find the most recent result file
        result_files = list(training_results_dir.glob(f"{model_name}_DEAP_*.json"))
        if not result_files:
            print(f"  No training result files found for {model_name}")
            return None
        
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        print(f"  Loaded results from {latest_file.name}")
        return results
    
    def estimate_training_time(self, model_name, batch_size, num_epochs=200):
        """Estimate training time based on inference speed"""
        print(f"\n⏱️ Estimating training time for {model_name}...")
        
        # Get inference speed for the batch size
        model = self.create_model_instance(model_name, {})
        # Move model to device first
        model = model.to(self.device)
        speed_results = self.benchmark_inference_speed(model, model_name, [batch_size], num_runs=50)
        
        # Estimate training time
        # Training is typically 2-3x slower than inference due to backward pass
        inference_time_per_batch = speed_results[batch_size]['avg_time_ms'] / 1000  # seconds
        training_time_per_batch = inference_time_per_batch * 2.5  # rough estimate
        
        # Calculate total training time
        # We need to estimate number of batches per epoch
        # This is approximate and should be validated with actual training logs
        estimated_batches_per_epoch = 360  # based on our analysis
        total_batches = estimated_batches_per_epoch * num_epochs
        
        total_training_time_seconds = training_time_per_batch * total_batches
        total_training_time_minutes = total_training_time_seconds / 60
        
        print(f"  Estimated training time: {total_training_time_minutes:.1f} minutes")
        
        return {
            'estimated_training_time_minutes': total_training_time_minutes,
            'inference_time_per_batch_ms': speed_results[batch_size]['avg_time_ms'],
            'training_time_per_batch_ms': training_time_per_batch * 1000
        }
    
    def benchmark_model(self, model_name, model_config=None, batch_size=128):
        """Complete benchmark for a single model"""
        print(f"\n{'='*60}")
        print(f"🚀 BENCHMARKING {model_name}")
        print(f"{'='*60}")
        
        # Create model
        model = self.create_model_instance(model_name, model_config)
        
        # Benchmark inference speed
        speed_results = self.benchmark_inference_speed(model, model_name)
        
        # Benchmark memory usage
        memory_results = self.benchmark_memory_usage(model, model_name)
        
        # Load training results
        training_results = self.load_training_results(model_name)
        
        # Estimate training time
        training_time_results = self.estimate_training_time(model_name, batch_size)
        
        # Compile results
        self.results[model_name] = {
            'speed': speed_results,
            'memory': memory_results,
            'training_results': training_results,
            'training_time': training_time_results
        }
        
        return self.results[model_name]
    
    def benchmark_all_models(self):
        """Benchmark all models"""
        print("🎯 Starting comprehensive model benchmark...")
        
        for model_name in self.models.keys():
            self.benchmark_model(model_name)
        
        return self.results
    
    def generate_comparison_table(self):
        """Generate comparison table similar to README"""
        print(f"\n{'='*100}")
        print("📋 MODEL COMPARISON TABLE")
        print(f"{'='*100}")
        
        # Table header
        print(f"{'Model':<15} {'Accuracy':<12} {'Speed (samples/sec)':<20} {'Latency (ms)':<15} {'Memory':<12} {'Training Time':<15}")
        print("-" * 100)
        
        for model_name in self.models.keys():
            if model_name not in self.results:
                continue
            
            result = self.results[model_name]
            
            # Accuracy
            accuracy = "N/A"
            if result['training_results'] and 'test_results' in result['training_results']:
                acc_value = result['training_results']['test_results'].get('accuracy', 0)
                # Handle tensor string format
                if isinstance(acc_value, str) and 'tensor(' in acc_value:
                    # Extract numeric value from tensor string
                    acc_value = float(acc_value.replace('tensor(', '').replace(')', ''))
                elif isinstance(acc_value, (int, float)):
                    acc_value = float(acc_value)
                else:
                    acc_value = 0
                accuracy = f"{acc_value:.2f}%"
            
            # Speed (batch 128) - samples per second
            speed = "N/A"
            if 128 in result['speed']:
                speed = f"{result['speed'][128]['throughput_samples_per_sec']:.1f}"
            
            # Latency (batch 128) - milliseconds
            latency = "N/A"
            if 128 in result['speed']:
                latency = f"{result['speed'][128]['avg_time_ms']:.1f}"
            
            # Memory
            memory = f"{result['memory']['model_size_mb']:.1f} MB"
            
            # Training time
            training_time = f"{result['training_time']['estimated_training_time_minutes']:.0f} min"
            
            print(f"{model_name:<15} {accuracy:<12} {speed:<20} {latency:<15} {memory:<12} {training_time:<15}")
        
        # Add explanation
        print(f"\n📝 Speed Metrics Explanation:")
        print(f"  • Speed (samples/sec): How many EEG samples the model can process per second")
        print(f"  • Latency (ms): Time to process a batch of 128 EEG samples")
        print(f"  • Real-time EEG sampling rate: 128 Hz (128 samples/second)")
        print(f"  • Real-time requirement: Latency < 1000ms for 128 samples")
    
    def generate_detailed_report(self):
        """Generate detailed benchmark report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'output_directory': str(self.output_dir.absolute()),
            'models': {}
        }
        
        for model_name, result in self.results.items():
            # Extract accuracy properly
            acc_value = 0
            if result['training_results'] and 'test_results' in result['training_results']:
                acc_value = result['training_results']['test_results'].get('accuracy', 0)
                if isinstance(acc_value, str) and 'tensor(' in acc_value:
                    acc_value = float(acc_value.replace('tensor(', '').replace(')', ''))
                elif isinstance(acc_value, (int, float)):
                    acc_value = float(acc_value)
                else:
                    acc_value = 0
            
            report['models'][model_name] = {
                'accuracy': acc_value,
                'speed_batch_128': result['speed'].get(128, {}),
                'latency_ms': result['speed'].get(128, {}).get('avg_time_ms', 0),
                'throughput_samples_per_sec': result['speed'].get(128, {}).get('throughput_samples_per_sec', 0),
                'memory_usage': result['memory'],
                'training_time': result['training_time']
            }
        
        # Save report to .benchmark directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"benchmark_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        return report
    
    def create_visualizations(self):
        """Create visualization plots"""
        if not self.results:
            print("No results to visualize")
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Benchmark Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        accuracies = []
        model_names = []
        for model_name, result in self.results.items():
            if result['training_results'] and 'test_results' in result['training_results']:
                acc_value = result['training_results']['test_results'].get('accuracy', 0)
                if isinstance(acc_value, str) and 'tensor(' in acc_value:
                    acc_value = float(acc_value.replace('tensor(', '').replace(')', ''))
                elif isinstance(acc_value, (int, float)):
                    acc_value = float(acc_value)
                else:
                    acc_value = 0
                accuracies.append(acc_value)
                model_names.append(model_name)
        
        if accuracies:
            axes[0, 0].bar(model_names, accuracies, color='skyblue')
            axes[0, 0].set_title('Test Accuracy (%)')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Speed comparison (batch 128)
        speeds = []
        speed_model_names = []
        for model_name, result in self.results.items():
            if 128 in result['speed']:
                speeds.append(result['speed'][128]['throughput_samples_per_sec'])
                speed_model_names.append(model_name)
        
        if speeds:
            axes[0, 1].bar(speed_model_names, speeds, color='lightgreen')
            axes[0, 1].set_title('Inference Speed (samples/sec, batch=128)')
            axes[0, 1].set_ylabel('Samples per Second')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Memory usage
        memories = []
        memory_model_names = []
        for model_name, result in self.results.items():
            memories.append(result['memory']['model_size_mb'])
            memory_model_names.append(model_name)
        
        if memories:
            axes[1, 0].bar(memory_model_names, memories, color='lightcoral')
            axes[1, 0].set_title('Model Size (MB)')
            axes[1, 0].set_ylabel('Size (MB)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Training time
        training_times = []
        time_model_names = []
        for model_name, result in self.results.items():
            training_times.append(result['training_time']['estimated_training_time_minutes'])
            time_model_names.append(model_name)
        
        if training_times:
            axes[1, 1].bar(time_model_names, training_times, color='gold')
            axes[1, 1].set_title('Estimated Training Time (minutes)')
            axes[1, 1].set_ylabel('Time (minutes)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot to .benchmark directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = self.output_dir / f"benchmark_visualization_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {plot_file}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Benchmark')
    parser.add_argument('--model', type=str, help='Specific model to benchmark')
    parser.add_argument('--all-models', action='store_true', help='Benchmark all models')
    parser.add_argument('--config', type=str, help='Path to model config file')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training time estimation')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--compare-results', action='store_true', help='Compare with README results')
    parser.add_argument('--output-dir', type=str, default='.benchmark', help='Output directory for benchmark results (default: .benchmark)')
    
    args = parser.parse_args()
    
    # Initialize benchmark with output directory
    benchmark = ModelBenchmark(output_dir=args.output_dir)
    
    if args.all_models:
        # Benchmark all models
        benchmark.benchmark_all_models()
    elif args.model:
        # Benchmark specific model
        benchmark.benchmark_model(args.model)
    else:
        # Default: benchmark all models
        benchmark.benchmark_all_models()
    
    # Generate comparison table
    benchmark.generate_comparison_table()
    
    # Generate detailed report
    benchmark.generate_detailed_report()
    
    # Create visualizations if requested
    if args.visualize:
        benchmark.create_visualizations()
    
    # Compare with README results if requested
    if args.compare_results:
        print(f"\n{'='*80}")
        print("📖 COMPARISON WITH README RESULTS")
        print(f"{'='*80}")
        
        readme_results = {
            'MMTSCeption': {'accuracy': 99.01, 'speed': 23.1, 'memory': 5.6, 'training_time': 47},
            'MMEEGNeX': {'accuracy': 84.90, 'speed': 4.7, 'memory': 6.2, 'training_time': 51},
            'MMEEGNet': {'accuracy': 80.38, 'speed': 57.5, 'memory': 5.4, 'training_time': 51},
            'MMFBCNet': {'accuracy': 77.53, 'speed': 473.0, 'memory': 11.0, 'training_time': 51}
        }
        
        print(f"{'Model':<15} {'Accuracy Diff':<15} {'Speed Diff':<15} {'Memory Diff':<15}")
        print("-" * 70)
        
        for model_name in benchmark.results.keys():
            if model_name in readme_results:
                readme = readme_results[model_name]
                current = benchmark.results[model_name]
                
                # Calculate differences
                acc_diff = "N/A"
                if current['training_results'] and 'test_results' in current['training_results']:
                    acc_value = current['training_results']['test_results'].get('accuracy', 0)
                    if isinstance(acc_value, str) and 'tensor(' in acc_value:
                        current_acc = float(acc_value.replace('tensor(', '').replace(')', ''))
                    elif isinstance(acc_value, (int, float)):
                        current_acc = float(acc_value)
                    else:
                        current_acc = 0
                    acc_diff = f"{current_acc - readme['accuracy']:+.2f}%"
                
                speed_diff = "N/A"
                if 128 in current['speed']:
                    current_speed = current['speed'][128]['throughput_samples_per_sec']
                    speed_diff = f"{current_speed - readme['speed']:+.1f}"
                
                memory_diff = f"{current['memory']['model_size_mb'] - readme['memory']:+.1f} MB"
                
                print(f"{model_name:<15} {acc_diff:<15} {speed_diff:<15} {memory_diff:<15}")

if __name__ == "__main__":
    main()
