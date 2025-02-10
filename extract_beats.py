import os
from beat_this.utils import save_beat_tsv
from beat_this.inference import File2Beats
file2beats = File2Beats(checkpoint_path="final1", device="cpu", dbn=False)


def get_beats(fma_paths, fma_path, beats_path):
    # create directory to save beats
    if not os.path.exists(beats_path):
        os.makedirs(beats_path)

    count = 0

    for file in fma_paths:
        count += 1
        beat_path = os.path.join(beats_path, file.replace(fma_path, "").replace(".mp3", ".beats"))
        if os.path.exists(beat_path):
            print(f"Skipping {file} due to existing beats file")
            continue
        print(f"Processing {file}")
        beats, downbeats = file2beats(file)
        # print(f"Beats: {beats}")
        # print(f"Downbeats: {downbeats}")
        # if beats or downbeats are empty, skip the file
        if len(beats) == 0 or len(downbeats) == 0:
            print(f"Skipping {file} due to empty beats or downbeats")
            # create empty file to indicate that the file was processed
            if not os.path.exists(beat_path):
                os.makedirs(os.path.dirname(beat_path), exist_ok=True)
            with open(beat_path, "w") as f:
                f.write("")
        else:
            # save beats and downbeats to file
            save_beat_tsv(beats, downbeats, beat_path)
            print(f"Beats saved to {beat_path}")

    print(f"Processed {count} files")

def calculate_bpm(file_path, start_time=0, start_at_downbeat=False, 
                 length_in_beats=None, length_in_downbeats=None, min_required_beats=8):
    """
    Calculate the average tempo in BPM from a beat file with various filtering options.
    
    Args:
        file_path (str or Path): Path to the beat file
        start_time (float): Start reading beats at or after this time (seconds)
        start_at_downbeat (bool): If True, start at the next downbeat after start_time
        length_in_beats (int, optional): Number of beats to process
        length_in_downbeats (int, optional): Number of downbeats to process
        min_required_beats (int): Minimum number of beats required to calculate tempo
        
    Returns:
        float or None: Average tempo in beats per minute, None if calculation fails
    """
    try:
        # Read all lines from the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            print(f"Warning: {file_path} is empty")
            return None
            
        # Parse all beats into a list of (time, beat_number) tuples
        beats = []
        for line in lines:
            try:
                time_str, beat_str = line.strip().split('\t')
                beats.append((float(time_str), int(beat_str)))
            except ValueError as e:
                print(f"Warning: Invalid line format in {file_path}: {line.strip()}")
                continue
                
        # Require at least 8 beats to calculate tempo
        if len(beats) < min_required_beats:
            print(f"Warning: Not enough beats in {file_path} to calculate tempo")
            return None
            
        # Filter beats based on start time and downbeat requirement
        start_idx = 0
        for i, (time, beat_num) in enumerate(beats):
            if time >= start_time:
                if start_at_downbeat:
                    # Look for the next downbeat (beat_num == 1)
                    while i < len(beats) and beats[i][1] != 1:
                        i += 1
                    if i >= len(beats):
                        print(f"Warning: No downbeat found after start_time {start_time}")
                        return None
                start_idx = i
                break
                
        filtered_beats = beats[start_idx:]
        
        # Apply length filters
        end_idx = len(filtered_beats)
        
        if length_in_beats is not None:
            end_idx = min(end_idx, start_idx + length_in_beats)
            
        if length_in_downbeats is not None:
            downbeats_found = 0
            for i, (_, beat_num) in enumerate(filtered_beats):
                if beat_num == 1:
                    downbeats_found += 1
                    if downbeats_found == length_in_downbeats:
                        end_idx = min(end_idx, start_idx + i + 1)
                        break
                        
        # Get final filtered beat sequence
        final_beats = filtered_beats[:end_idx]
        
        if len(final_beats) < 2:
            print(f"Warning: Not enough beats after filtering in {file_path}")
            return None
            
        # Calculate time differences between consecutive beats
        times = [time for time, _ in final_beats]
        time_diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
        
        # Calculate average time difference and convert to BPM
        avg_time_diff = sum(time_diffs) / len(time_diffs)
        bpm = 60 / avg_time_diff
        
        return round(bpm, 1)
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


from pathlib import Path

def process_beats_directory(beats_dir):
    """
    Recursively process all .beats files in directory and save .bpm files
    in the same location, calculating BPM starting at second downbeat.
    
    Args:
        beats_dir (str or Path): Directory containing .beats files
    """
    beats_path = Path(beats_dir)
    
    # Counter for progress tracking
    total_files = 0
    processed_files = 0
    error_files = 0
    
    # First count total files
    for _ in beats_path.rglob("*.beats"):
        total_files += 1
    
    print(f"Found {total_files} .beats files to process")
    
    # Process each .beats file
    for beats_file in beats_path.rglob("*.beats"):
        processed_files += 1
        
        # Create corresponding .bpm file path in same directory
        bpm_file = beats_file.with_suffix('.bpm')
        
        # Calculate BPM starting at second downbeat
        bpm = calculate_bpm(beats_file, start_at_downbeat=True, min_required_beats=32)
        # start at downbeat = true and min required beats=32 was used for fma_small
        
        # Write result to output file
        with open(bpm_file, 'w') as f:
            if bpm is not None:
                f.write(f"{bpm}\n")
            else:
                error_files += 1
        # if bpm is None:
        #     error_files += 1

        # Print progress
        if processed_files % 100 == 0 or processed_files == total_files:
            print(f"Processed {processed_files}/{total_files} files")

    print(f"Processing complete: {processed_files} files processed, {error_files} errors")



if __name__ == "__main__":

    fma_path = "/data/EECS-Studiosync/datasets/fma_medium/"
    beats_path = "/data/EECS-Studiosync/datasets/fma_medium/beats"

    # list of files to be processed
    # fma has subdirectories with audio files
    fma_paths = []
    for root, dirs, files in os.walk(fma_path):
        print(f"Processing {root}")
        for file in files:
            if file.endswith(".mp3"):
                fma_paths.append(os.path.join(root, file))
                # print(f"Added {file} to processing list")
                
    # sort by name
    fma_paths.sort()
    print(f"Processing {len(fma_paths)} files")

    corrupted_files = ["001486", "005574", "065753", "080391", "098558", "098559", "098560", "098565", "098566",
                        "098567", "098568", "098569", "098571", "099134", "105247", "108925", "126981", "127336",
                        "133297", "143992"]
    fma_paths = [f for f in fma_paths if not any(cf in f for cf in corrupted_files)]
    print(f"Removed corrupted files {corrupted_files}")

    get_beats(fma_paths, fma_path, beats_path)


    process_beats_directory(beats_path)

    # Get list of all empty .bpm files
    beats_path = Path(beats_path)
    print(f"Fetching bpm files from {beats_path}")
    empty_bpm_files = [f for f in beats_path.rglob("*.bpm") if f.stat().st_size == 0]
    print(f"Found {len(empty_bpm_files)} empty .bpm files")
    print(empty_bpm_files[:5])
    # save list of empty files to text file
    empty_files_path = beats_path / "empty_bpm_files.txt"
    with open(empty_files_path, 'w') as f:
        for file in empty_bpm_files:
            f.write(f"{file}\n")
    print(f"Saved list of emptu bpm files at {empty_files_path}")

