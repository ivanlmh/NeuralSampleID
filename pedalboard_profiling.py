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
from modules.transformations import GPUTransformSamples
from pedalboard import (
    Pedalboard,
    Chorus,
    Reverb,
    Distortion,
    PitchShift,
    time_stretch
)
from util import load_config

def profile_batch_effects(batch, cfg, effects_config):
    """Profile effects on a full batch of audio"""
    times = {
        'time_stretch': [],
        'pitch_shift': [],
        'chorus': [],
        'reverb': [],
        'distortion': [],
        'complete_chain': []
    }
    
    try:
        # Time stretch
        start_time = time.time()
        for audio in batch:
            _ = time_stretch(audio, cfg['fs'], 1.1)
        times['time_stretch'].append((time.time() - start_time) / len(batch))

        # Setup individual effects
        chorus = Chorus(**effects_config['chorus'])
        reverb = Reverb(**effects_config['reverb'])
        distortion = Distortion(**effects_config['distortion'])
        pitch_shifter = Pedalboard([PitchShift(semitones=2)])
        full_chain = Pedalboard([chorus, reverb, distortion])
        
        # Pitch shift
        start_time = time.time()
        for audio in batch:
            _ = pitch_shifter.process(audio, cfg['fs'])
        times['pitch_shift'].append((time.time() - start_time) / len(batch))

        # Chorus
        start_time = time.time()
        for audio in batch:
            _ = chorus.process(audio, cfg['fs'])
        times['chorus'].append((time.time() - start_time) / len(batch))

        # Reverb
        start_time = time.time()
        for audio in batch:
            _ = reverb.process(audio, cfg['fs'])
        times['reverb'].append((time.time() - start_time) / len(batch))

        # Distortion
        start_time = time.time()
        for audio in batch:
            _ = distortion.process(audio, cfg['fs'])
        times['distortion'].append((time.time() - start_time) / len(batch))

        # Complete chain
        start_time = time.time()
        for audio in batch:
            _ = full_chain.process(audio, cfg['fs'])
        times['complete_chain'].append((time.time() - start_time) / len(batch))

    except Exception as e:
        print(f"Error processing batch: {str(e)}")
    
    return times

def profile_random_batch_augmentations(batch, cfg):
    """Profile random augmentations on a full batch"""
    times = []
    transform = GPUTransformSamples(cfg=cfg, train=True)
    
    try:
        start_time = time.time()
        for audio in batch:
            audio_copy = audio.copy()
            if torch.rand(1).item() < 0.5:
                audio_copy = transform.apply_random_time_stretch(audio_copy)
            if torch.rand(1).item() < 0.5:
                audio_copy = transform.apply_random_pitch_shift(audio_copy)
            audio_copy = transform.train_transform.process(audio_copy, cfg['fs'])
        times.append((time.time() - start_time) / len(batch))
        
    except Exception as e:
        print(f"Error in random augmentation: {str(e)}")
    
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
    print("\nPedalboard Augmentation Profiling Results")
    print("=" * 50)
    
    for batch_size, batch_results in results.items():
        print(f"\nResults for batch size {batch_size}:")
        print("-" * 30)
        
        print("\nIndividual Effects (seconds per audio file):")
        for effect, stats in batch_results['individual_effects'].items():
            print(f"\n{effect.replace('_', ' ').title()}:")
            if stats['mean'] is not None:
                print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Trials: {stats['count']}")
            else:
                print("  No successful trials")
        
        print("\nRandom Augmentation Chains (seconds per audio file):")
        stats = batch_results['random_chains']
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
    
    # Effect configuration
    effects_config = {
        'chorus': {
            'rate_hz': 1.0,
            'depth': 0.25,
            'centre_delay_ms': 7.0,
            'feedback': 0.0,
            'mix': 0.5
        },
        'reverb': {
            'room_size': 0.8,
            'damping': 0.1,
            'wet_level': 0.5,
            'dry_level': 0.5
        },
        'distortion': {
            'drive_db': 25
        }
    }
    
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
        
        individual_times = {
            'time_stretch': [],
            'pitch_shift': [],
            'chorus': [],
            'reverb': [],
            'distortion': [],
            'complete_chain': []
        }
        random_times = []
        
        for i, batch in enumerate(tqdm(batches, desc=f"Processing batches of size {batch_size}")):
            # Profile individual effects
            batch_times = profile_batch_effects(batch, cfg, effects_config)
            for effect, times in batch_times.items():
                individual_times[effect].extend(times)
            
            # Profile random augmentations
            random_batch_times = profile_random_batch_augmentations(batch, cfg)
            random_times.extend(random_batch_times)
        
        results[batch_size] = {
            'individual_effects': {
                name: calculate_statistics(times)
                for name, times in individual_times.items()
            },
            'random_chains': calculate_statistics(random_times)
        }
    
    # Print results
    print_results(results)
    
    # Save results
    with open('pedalboard_profile_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()