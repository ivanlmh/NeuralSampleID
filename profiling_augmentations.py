import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")

from modules.data import NeuralfpDataset
from modules.transformations import GPUTransformNeuralfp
from torch_audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse
from util import load_config, load_augmentation_index

def profile_batch_augmentations(batch, cfg, ir_dir, noise_dir):
    """Profile IR and noise augmentations on a full batch"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    times = {
        'ir_only': [],
        'noise_only': [],
        'combined': []
    }
    
    try:
        # Create individual augmentations
        ir_transform = Compose([
            ApplyImpulseResponse(ir_paths=ir_dir, p=1.0)
        ])
        
        noise_transform = Compose([
            AddBackgroundNoise(
                background_paths=noise_dir,
                min_snr_in_db=cfg['tr_snr'][0],
                max_snr_in_db=cfg['tr_snr'][1],
                p=1.0
            )
        ])
        
        combined_transform = Compose([
            ApplyImpulseResponse(ir_paths=ir_dir, p=1.0),
            AddBackgroundNoise(
                background_paths=noise_dir,
                min_snr_in_db=cfg['tr_snr'][0],
                max_snr_in_db=cfg['tr_snr'][1],
                p=1.0
            )
        ])
        
        # Convert batch to expected shape [batch_size, channels, samples]
        batch_tensor = torch.tensor(batch).unsqueeze(1)
        batch_tensor = batch_tensor.to(device)
        
        # Profile IR only
        start_time = time.time()
        _ = ir_transform(batch_tensor, sample_rate=cfg['fs'])
        times['ir_only'].append((time.time() - start_time) / len(batch))

        # Profile noise only
        start_time = time.time()
        _ = noise_transform(batch_tensor, sample_rate=cfg['fs'])
        times['noise_only'].append((time.time() - start_time) / len(batch))

        # Profile combined
        start_time = time.time()
        _ = combined_transform(batch_tensor, sample_rate=cfg['fs'])
        times['combined'].append((time.time() - start_time) / len(batch))

    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        print(f"Batch shape: {batch_tensor.shape}")
    
    return times

def load_batches(cfg, batch_sizes=[32, 64, 128], num_batches=5):
    """Load multiple batches of different sizes for testing"""
    print(f"Loading batches for testing... Testing batch sizes: {batch_sizes}")
    batch_data = {}
    
    for batch_size in batch_sizes:
        print(f"\nLoading batches of size {batch_size}")
        dataset = NeuralfpDataset(
            cfg=cfg,
            path=cfg['train_dir'],
            train=True,
            transform=None
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            drop_last=True
        )
        
        batches = []
        for i, (x_i, x_j) in enumerate(loader):
            if i >= num_batches:
                break
            batches.append([x.numpy() for x in x_i])
            print(f"Loaded batch {i+1}/{num_batches} of size {batch_size}")
        
        batch_data[batch_size] = batches
    
    return batch_data

def calculate_statistics(times):
    """Calculate statistics for timing results"""
    if not times:
        return {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'count': 0
        }
    
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'count': len(times)
    }

def print_results(results):
    """Print profiling results in a readable format"""
    print("\nIR and Noise Augmentation Profiling Results")
    print("=" * 50)
    
    for batch_size, batch_results in results.items():
        print(f"\nResults for batch size {batch_size}:")
        print("-" * 30)
        
        print("\nAugmentations (seconds per audio file):")
        for aug_type, stats in batch_results.items():
            print(f"\n{aug_type.replace('_', ' ').title()}:")
            if stats['mean'] is not None:
                print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Trials: {stats['count']}")
            else:
                print("  No successful trials")

def main():
    # Load configuration
    cfg = load_config('config/grafp.yaml')
    print("Loaded config:", cfg)
    
    # Load augmentation indices
    print("\nLoading augmentation indices...")
    noise_train_idx = load_augmentation_index(cfg['noise_dir'], splits=0.8)["train"]
    ir_train_idx = load_augmentation_index(cfg['ir_dir'], splits=0.8)["train"]
    print(f"Loaded {len(noise_train_idx)} noise files and {len(ir_train_idx)} IR files")
    
    # Test different batch sizes
    batch_sizes = [32, 64, 128]
    num_batches = 5
    
    # Load batches
    batch_data = load_batches(cfg, batch_sizes, num_batches)
    
    # Results dictionary
    results = {}
    
    # Profile each batch size
    for batch_size in batch_sizes:
        print(f"\nProfiling batch size: {batch_size}")
        batches = batch_data[batch_size]
        
        aug_times = {
            'ir_only': [],
            'noise_only': [],
            'combined': []
        }
        
        for i, batch in enumerate(tqdm(batches, desc=f"Processing batches of size {batch_size}")):
            # Profile augmentations
            batch_times = profile_batch_augmentations(batch, cfg, ir_train_idx, noise_train_idx)
            for aug_type, times in batch_times.items():
                aug_times[aug_type].extend(times)
        
        results[batch_size] = {
            aug_type: calculate_statistics(times)
            for aug_type, times in aug_times.items()
        }
    
    # Print results
    print_results(results)
    
    # Save results
    with open('audio_augmentation_profile_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()