import os
import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
import torch.nn as nn
import warnings

from util import (
    load_index,
    load_sample100_index,
    get_frames,
    qtile_normalize,
    qtile_norm,
)


def neural_collate_fn(batch):
    """Collate function that can handle both training and validation data"""
    # Check if we're in validation mode (second element is None)
    if batch[0][1] is None:
        # Validation/test mode
        x_i = torch.stack([item[0] for item in batch])
        metadata = [item[2] for item in batch]
        return x_i, None, metadata
    else:
        # Training mode
        x_i = torch.stack([item[0] for item in batch])
        x_j = torch.stack([item[1] for item in batch])
        metadata = [item[2] for item in batch]

    return x_i, x_j, metadata


class NeuralfpDataset(Dataset):
    def __init__(self, cfg, path, transform=None, train=False):
        self.path = path
        self.transform = transform
        self.train = train
        self.norm = cfg["norm"]
        self.offset = cfg["offset"]
        self.sample_rate = cfg["fs"]
        self.dur = cfg["dur"]
        self.n_frames = cfg["n_frames"]
        self.silence = cfg["silence"]
        self.error_threshold = cfg["error_threshold"]
        if "stem" in cfg:
            self.stem = cfg["stem"]
        else:
            self.stem = None
        self.keys_dir = cfg.get("train_keys_dir" if train else "val_keys_dir", None)
        self.key_data = None
        if self.keys_dir:
            self.key_data = np.load(self.keys_dir, allow_pickle=True)

        self.beats_dir = cfg.get("train_beats_dir" if train else "val_beats_dir", None)
        self.min_beats_required = int(cfg.get("min_beats_required", 0))

        if train:
            self.filenames = load_index(cfg, path, mode="train", stem=self.stem)
        else:
            self.filenames = load_index(cfg, path, mode="valid", stem=self.stem)

        # Filter out files with insufficient beats
        if self.beats_dir:
            filtered_filenames = {}
            original_count = len(self.filenames)
            new_idx = 0  # Keep track of new continuous indices
            for idx, filepath in self.filenames.items():
                beats_data = self.load_beats_file(filepath)
                if (
                    beats_data is not None
                    and len(beats_data["times"]) >= self.min_beats_required
                ):
                    filtered_filenames[str(new_idx)] = (
                        filepath  # Use str(new_idx) as key
                    )
                    new_idx += 1
            self.filenames = filtered_filenames
            print(
                f"Filtered out {original_count - len(self.filenames)} files with insufficient beats"
            )

        print(f"Loaded {len(self.filenames)} files from {path}")
        self.ignore_idx = []
        self.error_counts = {}

    def get_key_for_file(self, filepath):
        if self.key_data is None:
            return -1  # Unknown key

        # Extract relative path from full filepath to match key_data format
        rel_path = "/".join(filepath.split("/")[-2:])  # Gets "091/091869.mp3" format
        return int(self.key_data.get(rel_path, -1))

    def load_beats_file(self, filepath):
        """Load beats file corresponding to audio file"""
        # Convert audio filepath to beats filepath
        # e.g. /path/to/000/000002.mp3 -> /path/to/beats/000/000002.beats
        beats_path = os.path.join(
            self.beats_dir,
            os.path.basename(os.path.dirname(filepath)),
            os.path.splitext(os.path.basename(filepath))[0] + ".beats",
        )

        if not os.path.exists(beats_path):
            return None

        # Load beats file
        beats = []
        beat_numbers = []
        try:
            with open(beats_path) as f:
                for line in f:
                    time, beat = line.strip().split("\t")
                    beats.append(float(time))
                    beat_numbers.append(int(beat))
            return {"times": beats, "numbers": beat_numbers}
        except:
            return None

    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]

        datapath = self.filenames[str(idx)]
        try:
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            audio, sr = torchaudio.load(datapath)

        except Exception:
            print("Error loading:" + self.filenames[str(idx)])
            self.error_counts[idx] = self.error_counts.get(idx, 0) + 1
            if self.error_counts[idx] > self.error_threshold:
                self.ignore_idx.append(idx)
            return self[idx + 1]

        audio_mono = audio.mean(dim=0)

        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
        audio_resampled = resampler(audio_mono)

        clip_frames = int(self.sample_rate * self.dur)

        if len(audio_resampled) <= clip_frames:
            # self.ignore_idx.append(idx)
            return self[idx + 1]

        key = self.get_key_for_file(datapath)
        beats = self.load_beats_file(datapath)
        metadata = {
            "file_path": datapath,
            "idx": idx,
            "original_sr": sr,
            "length": len(audio_resampled),
            "key": key,
            "beats": beats,
        }

        #   For training pipeline, output a random frame of the audio
        if self.train:
            a_i = audio_resampled
            a_j = a_i.clone()

            offset_mod = int(self.sample_rate * (self.offset) + clip_frames)
            if len(audio_resampled) < offset_mod:
                print(
                    "Audio too short (offset_mod > len(audio resampled)). Skipping..."
                )
                return self[idx + 1]
            r = np.random.randint(0, len(audio_resampled) - offset_mod)
            ri = np.random.randint(0, offset_mod - clip_frames)
            rj = np.random.randint(0, offset_mod - clip_frames)

            # Add timestamps to metadata
            metadata.update(
                {"start_i": r + ri, "start_j": r + rj, "clip_length": clip_frames}
            )

            clip_i = a_i[r : r + offset_mod]
            clip_j = a_j[r : r + offset_mod]
            x_i = clip_i[ri : ri + clip_frames]
            x_j = clip_j[rj : rj + clip_frames]

            if x_i.abs().max() < self.silence or x_j.abs().max() < self.silence:
                print("Silence detected. Skipping...")
                return self[idx + 1]

            if self.norm is not None:
                norm_val = qtile_norm(audio_resampled, q=self.norm)
                x_i = x_i / norm_val
                x_j = x_j / norm_val

            if self.transform is not None:
                x_i, x_j, transform_metadata = self.transform(x_i, x_j, metadata)

            if x_i is None or x_j is None:
                return self[idx + 1]

            # Pad or truncate to sample_rate * dur
            if len(x_i) < clip_frames:
                x_i = F.pad(x_i, (0, clip_frames - len(x_i)))
            else:
                x_i = x_i[:clip_frames]

            if len(x_j) < clip_frames:
                x_j = F.pad(x_j, (0, clip_frames - len(x_j)))
            else:
                x_j = x_j[:clip_frames]

            return x_i, x_j, metadata

        #   For validation / test, output consecutive (overlapping) frames
        else:
            return audio_resampled, None, metadata
            # return audio_resampled

    def __len__(self):
        return len(self.filenames)


# create class for NeuralSampleIDDataset
class NeuralSampleIDDataset(Dataset):
    def __init__(self, cfg, path, transform=None, train=False):
        self.path = path
        self.transform = transform
        self.train = train
        self.norm = cfg["norm"]
        self.offset = cfg["offset"]
        self.sample_rate = cfg["fs"]
        self.dur = cfg["dur"]
        self.n_frames = cfg["n_frames"]
        self.silence = cfg["silence"]
        self.error_threshold = cfg["error_threshold"]

        if train:
            # self.filenames = load_index(cfg, path, mode="train")
            self.filenames = load_sample100_index(cfg, path, mode="train")
        else:
            self.filenames = load_sample100_index(cfg, path, mode="valid")

        print(f"Loaded {len(self.filenames)} files from {path}")
        self.ignore_idx = []
        self.error_counts = {}

    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]

        datapath = self.filenames[str(idx)]
        datapath_sample = datapath[0]
        datapath_original = datapath[1]
        try:
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            audio_sample, sr = torchaudio.load(datapath_sample)
            audio_original, sr_original = torchaudio.load(datapath_original)
            assert (
                sr == sr_original
            ), "Sample and original audio have different sample rates"

        except Exception:
            print("Error loading:" + self.filenames[str(idx)])
            self.error_counts[idx] = self.error_counts.get(idx, 0) + 1
            if self.error_counts[idx] > self.error_threshold:
                self.ignore_idx.append(idx)
            return self[idx + 1]

        # audio mono shape is (1, n_samples)
        audio_sample_mono = audio_sample.mean(dim=0)
        audio_original_mono = audio_original.mean(dim=0)

        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
        audio_sample_resampled = resampler(audio_sample_mono)
        audio_original_resampled = resampler(audio_original_mono)

        # csv columns are sample_id,original_track_id,sample_track_id,t_original,t_sample,n_repetitions,sample_type,interpolation,comments
        # open csv and get t_sample
        csv_path = os.path.join(self.path, "samples.csv")
        with open(csv_path, encoding="latin-1") as f:
            lines = f.readlines()
            t_sample = int(lines[idx + 1].split(",")[4])  # +1 because of header of csv
            t_original = int(lines[idx + 1].split(",")[3])
            # print(f"t_sample: {t_sample}, t_original: {t_original}")
        clip_frames = int(self.sample_rate * self.dur)
        # print(
        #     f"clip_frames: {clip_frames}, sample_rate: {self.sample_rate}, dur: {self.dur}"
        # )
        # print(
        #     f"audio_sample_resampled.shape: {audio_sample_resampled.shape}, audio_original_resampled.shape: {audio_original_resampled.shape}"
        # )

        audio_sample_cut = audio_sample_resampled[t_sample : t_sample + clip_frames]
        audio_original_cut = audio_original_resampled[
            t_original : t_original + clip_frames
        ]

        #   TODO:
        if self.train:
            return audio_sample_cut, audio_original_cut
        #   For validation / test, output consecutive (overlapping) frames
        else:
            # return audio_sample_resampled, audio_original_resampled
            return audio_sample_cut, audio_original_cut
            # return audio_resampled

            # # I will return the cut verions, so as to focus on the time around which the sample starts
            # return audio_sample_resampled[t_sample : t_sample + int(self.sample_rate * 10)], audio_original_resampled[t_sample : t_sample + int(self.sample_rate * 10)]

    def __len__(self):
        return len(self.filenames)
