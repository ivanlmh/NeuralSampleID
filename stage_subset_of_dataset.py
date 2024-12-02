import os
import random
import shutil

import pandas as pd
import numpy as np
import librosa


def load_audio_files(dir, ext=".mp3"):
    audio_files = []
    for file in os.listdir(dir):
        if file.endswith(ext):
            audio_files.append(os.path.join(dir, file))
    audio_files.sort()
    return audio_files


sr = 22050

base_track_length_s = 60

# load base tracks from /home/ivan/Documents/FIng/QueenMary/
base_tracks_paths = load_audio_files(
    "/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/base tracks/"
)
base_tracks = []
for track in base_tracks_paths:
    print(f"Loading {track}...")
    y, _ = librosa.load(track, sr=sr)
    # # cut base song to base_track_length_s
    y = y[: base_track_length_s * sr]
    base_tracks.append(y)


samples_path = "/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/samples/"
sample_tracks_paths = load_audio_files(samples_path, ext=".wav")
sample_tracks = []
for track in sample_tracks_paths:
    print(f"Loading {track}...")
    y, _ = librosa.load(track, sr=sr)
    # pad song with 1s of silence before and after
    print(y.shape)
    y = np.concatenate([np.zeros(4 * sr), y, np.zeros(int(0.2 * sr))])
    print(y.shape)
    sample_tracks.append(y)


# load mp3 files that contain stem separated samples
separated_samples_path = (
    "/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/samples/"
)
separated_sample_tracks_paths = load_audio_files(separated_samples_path, ext=".mp3")
separated_sample_tracks = []
for track in separated_sample_tracks_paths:
    print(f"Loading {track}...")
    y, _ = librosa.load(track, sr=sr)
    # pad song with 1s of silence before and after
    y = np.concatenate([np.zeros(4 * sr), y, np.zeros(int(0.2 * sr))])
    separated_sample_tracks.append(y)


config_array = [
    {"gain": {"gain_db": 0}},
    {"gain": {"gain_db": -3}},
    {"gain": {"gain_db": -6}},
    {"gain": {"gain_db": 1}},
    {"sample_length": {"seconds": None, "beats": 4}},  # 5,
    {"sample_length": {"seconds": None, "beats": 8}},  # 5,
    {"tempo_sync": {"multiplier": 1}},
    {"tempo_change": 0.98},
    {"tempo_change": 1.02},
    {"tempo_change": 1.05},
    {"tempo_change": 1.5},
    {"tempo_change": 2},
    {"pitch_sync": True},
    {"pitch_shift": -1},
    {"pitch_shift": 1},
    {"pitch_shift": -3},
    {"pitch_shift": 3},
    {"pitch_shift": -7},
    {"pitch_shift": 7},
    {"pitch_shift": -12},
    {"pitch_shift": 12},
    {"pitch_sync": True, "tempo_sync": {"multiplier": 1}},
    {
        "effects": {
            "compressor": {
                "threshold_db": -6,
                "ratio": 6,
                "attack_ms": 0.001,
                "release_ms": 0.2,
            },
            "reverb": False,
            "delay": False,
            "distortion": False,
        }
    },
    {
        "effects": {
            "reverb": {
                "room_size": 0.8,
                "damping": 0.1,
                "wet_level": 0.5,
                "dry_level": 0.5,
            },
            "compressor": False,
            "delay": False,
            "distortion": False,
        }
    },
    {
        "effects": {
            "delay": {"delay_seconds": 0.3, "feedback": 0.4, "mix": 0.3},
            "compressor": False,
            "reverb": False,
            "distortion": False,
        }
    },
    {
        "effects": {
            "distortion": {"drive_db": 20},
            "compressor": False,
            "reverb": False,
            "delay": False,
        }
    },
    {
        "effects": {
            "compressor": {
                "threshold_db": -6,
                "ratio": 6,
                "attack_ms": 0.001,
                "release_ms": 0.2,
            },
            "reverb": {
                "room_size": 0.8,
                "damping": 0.1,
                "wet_level": 0.5,
                "dry_level": 0.5,
            },
            "delay": {"delay_seconds": 0.3, "feedback": 0.4, "mix": 0.3},
            "distortion": {"drive_db": 20},
        }
    },
    # {'drums_separation': True},
    # {'vocal_separation': True},
    # {'bass_separation': True}
]


# Ok, now I want to create a subset of the artificial dataset which I moved to /home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2/
# create folder to save subset
os.makedirs(
    f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2_subset",
    exist_ok=True,
)
# create folder to save subset audio
os.makedirs(
    f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2_subset/audio_wav",
    exist_ok=True,
)

# move the "mix" mp3 files to the subset folder
for file in os.listdir(
    "/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2/audio_wav"
):
    if "mix" in file:
        shutil.copy(
            f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2/audio_wav/{file}",
            f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2_subset/audio_wav/{file}",
        )

# and for a random 25% i want to move the effect and original sample files
files = os.listdir(
    "/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2/audio_wav"
)
# keep only files that are effected or original samples
files = [file for file in files if "effected" in file]
print(len(files))

# seed 42
random.seed(42)
random.shuffle(files)
files = files[: int(len(files) * 0.25)]

# append the original files
files += [file.replace("effected", "original") for file in files]

for file in files:
    shutil.copy(
        f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2/audio_wav/{file}",
        f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2_subset/audio_wav/{file}",
    )

    # and the config files for those files
    shutil.copy(
        f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2/config_{file.split('_')[2]}_{file.split('_')[3]}_{file.split('_')[4].split('.')[0]}.json",
        f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2_subset/config_{file.split('_')[2]}_{file.split('_')[3]}_{file.split('_')[4].split('.')[0]}.json",
    )
