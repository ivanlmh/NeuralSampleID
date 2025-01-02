import os
from pathlib import Path
import torch
import torchaudio
import demucs.separate
import time
from datetime import datetime, timedelta

# Set environment variable for PyTorch memory management
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

def format_time(seconds):
    """Convert seconds to human readable time format"""
    return str(timedelta(seconds=int(seconds)))

def check_stems_exist(filepath, output_dir):
    """
    Check if all four stems already exist for a given file.
    Returns True if all stems exist, False otherwise.
    """
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(filepath))[0]
    
    # Define expected stems
    stems = ['drums', 'bass', 'vocals', 'other']
    
    # Check for each stem
    stem_path = os.path.join(output_dir, 'htdemucs', filename)
    return all(os.path.exists(os.path.join(stem_path, f"{stem}.mp3")) for stem in stems)

def process_audio_files(input_dir, output_dir):
    """
    Process all audio files in input_dir and its subdirectories using Demucs.
    Uses high quality settings with parallel processing.
    """
    start_time = time.time()
    
    # Log start time
    print(f"Started processing at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Calculate number of jobs based on available CPU cores
    n_jobs = min(4, os.cpu_count() // 2) if device == "cuda" else os.cpu_count()
    
    # Collect all MP3 files first
    collection_start = time.time()

    # Open file with empty .bpm files
    with open("/data/home/eez083/datasets/fma_small/beats/empty_bpm_files.txt", 'r') as f:
        empty_files = f.readlines()
    empty_files = [f.strip().split("/")[-1].split(".")[0] for f in empty_files]

    mp3_files = []
    skipped_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                input_path = os.path.join(root, file)
                # Skip files that contain specific strings or empty bpm
                if any(x in input_path for x in empty_files + ["108925", "099134", "133297"]):
                    skipped_files.append(input_path)
                    continue
                # Check if stems already exist
                if check_stems_exist(input_path, output_dir):
                    skipped_files.append(input_path)
                    continue
                mp3_files.append(input_path)

    mp3_files.sort()
    collection_time = time.time() - collection_start
    print(f"Found {len(mp3_files)} MP3 files to process and {len(skipped_files)} already processed files")
    print(f"Collection took {format_time(collection_time)}")
    print(f"Using {n_jobs} parallel jobs")
    
    # Process files in batches of 10
    batch_size = 1 #10
    processing_start = time.time()
    total_processed = 0
    
    for i in range(0, len(mp3_files), batch_size):
        batch = mp3_files[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(mp3_files) + batch_size - 1)//batch_size}")
        print(f"Files {i+1} to {min(i + batch_size, len(mp3_files))} of {len(mp3_files)}")
        
        try:
            demucs.separate.main([
                "-n", "htdemucs",        #"htdemucs_ft",   # best quality model
                "-d", device,          # use GPU if available
                # "--segment", "10",     # process in 10-second segments
                "--mp3",              # save as MP3
                "--mp3-bitrate", "320",  # high quality MP3
                "-j", str(n_jobs),    # parallel processing
                *batch,              # current batch of files
                "-o", output_dir      # output directory
            ])
            total_processed += len(batch)
            print(f"Successfully processed batch. Total processed: {total_processed}/{len(mp3_files)}")
            
        except Exception as e:
            print(f"Error during processing batch: {str(e)}")
            print("Skipping to next batch...")
    
    processing_time = time.time() - processing_start
    
    # Calculate and display final statistics
    total_time = time.time() - start_time
    print("\nFinal Statistics:")
    print(f"Total files processed: {total_processed}")
    print(f"Files skipped (already processed): {len(skipped_files)}")
    print(f"Processing time: {format_time(processing_time)}")
    print(f"Total execution time: {format_time(total_time)}")
    print(f"Average time per file: {format_time(processing_time / total_processed) if total_processed > 0 else 'N/A'}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # input_dir = "/data/home/acw723/datasets/fma/fma_small"
    # output_dir = "/data/home/eez083/datasets/fma_small"
    input_dir = "/data/home/eez083/sample_100/audio"
    output_dir = "/data/home/eez083/sample_100"
    
    process_audio_files(input_dir, output_dir)