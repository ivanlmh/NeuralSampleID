import os
import json
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import pandas as pd

from util import (
    create_fp_dir,
    load_config,
    query_len_from_seconds,
    seconds_from_query_len,
)
from modules.transformations import GPUTransformNeuralSampleid
from encoder.graph_encoder import GraphEncoder
from simclr.simclr import SimCLR
from eval import eval_faiss

torchaudio.set_audio_backend("soundfile")


class SampleIDDataset(Dataset):
    """Dataset for sample identification using JSON annotations"""

    def __init__(self, annotations_dir, audio_dir, sample_length=None, sr=22050):
        self.audio_dir = audio_dir
        self.sr = sr
        self.sample_length = sample_length

        # Load all JSON files from annotations directory
        json_files = os.listdir(annotations_dir)
        json_files.sort()
        print(json_files[:5])
        self.annotations = []
        for filename in json_files:
            # print(f"Loading {filename}")
            if filename.endswith(".json") and not filename.startswith("extra"):
                with open(os.path.join(annotations_dir, filename)) as f:
                    try:
                        annotation = json.load(f)
                        if isinstance(annotation, dict):  # Ensure it's a dictionary
                            annotation["id"] = filename.split(".")[0]
                            self.annotations.append(annotation)
                        else:
                            print(f"Skipping {filename}: Not a dictionary")
                    except json.JSONDecodeError:
                        print(f"Error decoding {filename}")

        print(f"Loaded {len(self.annotations)} annotations")

        # Load metadata if available
        try:
            print("Loading metadata from CSVs in", audio_dir)
            self.tracks_df = pd.read_csv(
                os.path.join(audio_dir, "tracks.csv"), encoding="latin-1"
            )
            self.samples_df = pd.read_csv(
                os.path.join(audio_dir, "samples.csv"), encoding="latin-1"
            )
        except:
            print("Warning: Metadata files not found")
            self.tracks_df = None
            self.samples_df = None

    def __len__(self):
        return len(self.annotations)

    def load_audio(self, filepath, start_time=None, duration=None):
        try:
            if start_time is not None and duration is not None:
                # Load specific segment
                offset = start_time
                num_frames = int(duration * self.sr)
                audio, loaded_sr = torchaudio.load(
                    filepath, frame_offset=int(offset * self.sr), num_frames=num_frames
                )
            else:
                # Load full file
                audio, loaded_sr = torchaudio.load(filepath)

            # Convert to mono and resample if needed
            audio = audio.mean(dim=0)
            if loaded_sr != self.sr:
                resampler = torchaudio.transforms.Resample(loaded_sr, self.sr)
                audio = resampler(audio)

            return audio

        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None

    def find_sample_window(self, annotation):
        """Determine the correct audio window to use based on annotation type"""
        base_times = annotation["base_time_annotations"]
        sample_times = annotation["sample_time_annotations"]

        # Default to first annotation if no type specified
        if len(sample_times) == 0 or "type" not in sample_times[0]:
            return base_times[0], sample_times[0]

        sample_type = sample_times[0].get("type", "").lower()

        if "absence" in sample_type:
            # For absence annotations, find the first non-absence segment after
            for base_time in base_times:
                if base_time.get("type", "").lower() != "absence":
                    if base_time["start_time"] > sample_times[0]["end_time"]:
                        return base_time, {
                            "start_time": base_time["start_time"],
                            "end_time": base_time["end_time"],
                        }
            # If no valid segment found after absence, use first non-absence segment
            for base_time in base_times:
                if base_time.get("type", "").lower() != "absence":
                    return base_time, {
                        "start_time": base_time["start_time"],
                        "end_time": base_time["end_time"],
                    }

        # For occurrence/presence annotations, use the annotated times
        return base_times[0], sample_times[0]

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # Get audio paths
        base_filepath = os.path.join(self.audio_dir, "audio", annotation["base_file"])
        sample_filepath = os.path.join(
            self.audio_dir, "audio", annotation["sample_file"]
        )

        # Get appropriate time windows based on annotation type
        base_time, sample_time = self.find_sample_window(annotation)

        # Load audio segments
        base_audio = self.load_audio(
            base_filepath,
            start_time=base_time["start_time"],
            duration=base_time["end_time"] - base_time["start_time"],
        )
        sample_audio = self.load_audio(
            sample_filepath,
            start_time=sample_time["start_time"],
            duration=sample_time["end_time"] - sample_time["start_time"],
        )

        if base_audio is None or sample_audio is None:
            # Return a default item or handle error
            return (
                torch.zeros(1, int(self.sr * self.sample_length)),
                torch.zeros(1, int(self.sr * self.sample_length)),
                {},
            )

        # For now just cut or pad to sample_length second segments
        if base_audio.shape[0] < int(self.sr * self.sample_length):
            base_audio = torch.nn.functional.pad(
                base_audio, (0, int(self.sr * self.sample_length) - base_audio.shape[0])
            )
        else:
            base_audio = base_audio[: int(self.sr * self.sample_length)]

        if sample_audio.shape[0] < int(self.sr * self.sample_length):
            sample_audio = torch.nn.functional.pad(
                sample_audio,
                (0, int(self.sr * self.sample_length) - sample_audio.shape[0]),
            )
        else:
            sample_audio = sample_audio[: int(self.sr * self.sample_length)]
        # TODO: Implement more sophisticated handling of audio through various segments with hop

        metadata = {
            "annotation_id": annotation["id"],
            "base_file": annotation["base_file"],
            "sample_file": annotation["sample_file"],
            "base_time": base_time,
            "sample_time": sample_time,
        }

        print(
            f"Retrieved {base_filepath.split('/')[-1]} and {sample_filepath.split('/')[-1]} for {annotation['id']}"
        )

        return base_audio, sample_audio, metadata


class SampleIDDummyDataset(Dataset):
    """Dataset that loads segments from all the audios starting with the letter N in the samples directory"""

    def __init__(self, audio_dir, sample_length=None, sr=22050):
        self.audio_dir = audio_dir
        self.sr = sr
        self.sample_length = sample_length

        # Load all JSON files from annotations directory
        audio_files = os.listdir(os.path.join(audio_dir, "audio"))
        audio_files.sort()
        print(audio_files[:5])
        self.audio_files = []
        for filename in audio_files:
            if filename.startswith("N"):
                self.audio_files.append(filename)
        print(f"Loaded {len(self.audio_files)} audio files")

    def __len__(self):
        return len(self.audio_files)

    def load_audio(self, filepath, start_time=None, duration=None):
        try:
            if start_time is not None and duration is not None:
                # Load specific segment
                offset = start_time
                num_frames = int(duration * self.sr)
                audio, loaded_sr = torchaudio.load(
                    filepath, frame_offset=int(offset * self.sr), num_frames=num_frames
                )
            else:
                # Load full file
                audio, loaded_sr = torchaudio.load(filepath)

            # Convert to mono and resample if needed
            audio = audio.mean(dim=0)
            if loaded_sr != self.sr:
                resampler = torchaudio.transforms.Resample(loaded_sr, self.sr)
                audio = resampler(audio)

            return audio

        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        audio_filepath = os.path.join(self.audio_dir, "audio", audio_file)
        audio = self.load_audio(audio_filepath)

        if audio is None:
            # Return a default item or handle error
            return torch.zeros(1, int(self.sr * self.sample_length), {})

        # Get a random segment of sample_length seconds
        start_time = np.random.uniform(
            0, audio.shape[0] - (self.sr * self.sample_length)
        )
        audio = audio[int(start_time) : int(start_time + self.sr * self.sample_length)]

        metadata = {"audio_file": audio_file, "start_time": start_time}

        return audio, audio, metadata


def create_embeddings(dataloader, augment, model, output_root_dir, verbose=True):
    """Create embeddings for base and sample audio segments"""
    emb_orig = []
    emb_sample = []

    print("=> Creating embeddings...")
    for idx, (base_audio, sample_audio, metadata) in enumerate(dataloader):
        # base_audio = base_audio.to(device)
        # sample_audio = sample_audio.to(device)

        base_audio = base_audio.to(device)  # Remove channel dim for augmentation
        sample_audio = sample_audio.to(device)
        print(base_audio.shape, sample_audio.shape)
        # Apply augmentations
        # x_i, x_j, aug_metadata = augment(x_i, x_j, metadata)
        x_i, _, _ = augment(base_audio, base_audio, metadata)
        x_j, _, _ = augment(sample_audio, sample_audio, metadata)
        print(x_i.shape, x_j.shape)

        # Generate embeddings
        with torch.no_grad():
            _, _, z_i, z_j = model(x_i.to(device), x_j.to(device))

        emb_orig.append(z_i.detach().cpu().numpy())
        emb_sample.append(z_j.detach().cpu().numpy())

        if verbose and idx % 10 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {z_i.shape}")

    # Concatenate embeddings
    emb_orig = np.concatenate(emb_orig)
    emb_sample = np.concatenate(emb_sample)
    arr_shape = (len(emb_orig), z_i.shape[-1])

    # Save query embeddings
    arr_q = np.memmap(
        f"{output_root_dir}/query.mm", dtype="float32", mode="w+", shape=arr_shape
    )
    arr_q[:] = emb_sample[:]
    arr_q.flush()
    del arr_q

    np.save(f"{output_root_dir}/query_shape.npy", arr_shape)

    # Save database embeddings
    arr_db = np.memmap(
        f"{output_root_dir}/db.mm", dtype="float32", mode="w+", shape=arr_shape
    )
    arr_db[:] = emb_orig[:]
    arr_db.flush()
    del arr_db

    np.save(f"{output_root_dir}/db_shape.npy", arr_shape)


def main():
    parser = argparse.ArgumentParser(description="Sample ID Evaluation")
    parser.add_argument("--config", default="config/grafp.yaml", type=str)
    parser.add_argument("--test_config", default="config/test_config.yaml", type=str)
    parser.add_argument("--annotations_dir", required=True, type=str)
    parser.add_argument("--sample_dir", required=True, type=str)
    parser.add_argument("--encoder", default="grafp", type=str)
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    args = parser.parse_args()

    # Load configs
    cfg = load_config(args.config)
    test_cfg = load_config(args.test_config)

    # Set up model
    print("Creating model...")
    model = SimCLR(
        cfg, encoder=GraphEncoder(cfg=cfg, in_channels=cfg["n_filters"], k=args.k)
    )
    model = model.to(device)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # Create dataset and dataloader
    dataset = SampleIDDataset(
        args.annotations_dir,
        args.sample_dir,
        sample_length=cfg["dur"],
        sr=cfg["fs"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    # Set up augmentation
    test_augment = GPUTransformNeuralSampleid(cfg, train=False, cpu=True).to(device)

    # Evaluate each checkpoint
    for ckp_name, epochs in test_cfg.items():
        if not isinstance(epochs, list):
            epochs = [epochs]

        writer = SummaryWriter(f"runs/{ckp_name}")

        for epoch in epochs:
            ckp = os.path.join("checkpoint", f"model_{ckp_name}_{str(epoch)}.pth")

            # Load checkpoint
            if os.path.isfile(ckp):
                print(f"=> Loading checkpoint '{ckp}'")
                checkpoint = torch.load(ckp, map_location=device)

                # Handle DataParallel state dict
                if "module" in list(checkpoint["state_dict"].keys())[0] and (
                    torch.cuda.device_count() == 1
                ):
                    checkpoint["state_dict"] = {
                        key.replace("module.", ""): value
                        for key, value in checkpoint["state_dict"].items()
                    }
                model.load_state_dict(checkpoint["state_dict"])
            else:
                print(f"=> No checkpoint found at '{ckp}'")
                continue

            # Create embeddings directory
            fp_dir = create_fp_dir(resume=ckp, train=False)

            # Generate embeddings
            create_embeddings(
                dataloader,
                augment=test_augment,
                model=model,
                output_root_dir=fp_dir,
                verbose=True,
            )

            # Evaluate using FAISS
            hit_rates = eval_faiss(
                emb_dir=fp_dir, test_ids="all", index_type="l2", nogpu=True
            )

            # Log results
            print("-------Test hit-rates-------")
            print(f"Top-1 exact hit rate = {hit_rates[0]}")
            print(f"Top-1 near hit rate = {hit_rates[1]}")
            print(f"Top-3 exact hit rate = {hit_rates[2]}")
            print(f"Top-10 exact hit rate = {hit_rates[3]}")

            label = epoch if isinstance(epoch, int) else 0
            writer.add_scalar("exact_hit_rate", hit_rates[0], label)
            writer.add_scalar("near_hit_rate", hit_rates[1], label)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
