import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse
from torchaudio.transforms import (
    MelSpectrogram,
    TimeMasking,
    FrequencyMasking,
    AmplitudeToDB,
    Resample,
)
import warnings

from pedalboard import Pedalboard, Chorus, Reverb, Distortion, PitchShift, time_stretch
from music2latent import EncoderDecoder


class GPUTransformNeuralfp(nn.Module):
    """
    Transform includes noise addition and impulse response.
    """

    def __init__(self, cfg, ir_dir, noise_dir, train=True, cpu=False):
        super(GPUTransformNeuralfp, self).__init__()
        self.sample_rate = cfg["fs"]
        self.ir_dir = ir_dir
        self.noise_dir = noise_dir
        self.overlap = cfg["overlap"]
        self.arch = cfg["arch"]
        self.n_frames = cfg["n_frames"]
        self.train = train
        self.cpu = cpu
        self.cfg = cfg

        self.train_transform = Compose(
            [
                ApplyImpulseResponse(ir_paths=self.ir_dir, p=cfg["ir_prob"]),
                AddBackgroundNoise(
                    background_paths=self.noise_dir,
                    min_snr_in_db=cfg["tr_snr"][0],
                    max_snr_in_db=cfg["tr_snr"][1],
                    p=cfg["noise_prob"],
                ),
            ]
        )

        self.val_transform = Compose(
            [
                ApplyImpulseResponse(ir_paths=self.ir_dir, p=1),
                AddBackgroundNoise(
                    background_paths=self.noise_dir,
                    min_snr_in_db=cfg["val_snr"][0],
                    max_snr_in_db=cfg["val_snr"][1],
                    p=1,
                ),
            ]
        )

        self.logmelspec = nn.Sequential(
            MelSpectrogram(
                sample_rate=self.sample_rate,
                win_length=cfg["win_len"],
                hop_length=cfg["hop_len"],
                n_fft=cfg["n_fft"],
                n_mels=cfg["n_mels"],
            ),
            AmplitudeToDB(),
        )

        self.spec_aug = nn.Sequential(
            TimeMasking(cfg["time_mask"], True),
            FrequencyMasking(cfg["freq_mask"], True),
        )

        self.melspec = MelSpectrogram(
            sample_rate=self.sample_rate,
            win_length=cfg["win_len"],
            hop_length=cfg["hop_len"],
            n_fft=cfg["n_fft"],
            n_mels=cfg["n_mels"],
        )

    def forward(self, x_i, x_j):
        if self.cpu:
            try:
                x_j = self.train_transform(
                    x_j.view(1, 1, x_j.shape[-1]), sample_rate=self.sample_rate
                )
            except ValueError:
                print("Error loading noise file. Hack to solve issue...")
                # Increase length of x_j by 1 sample
                x_j = F.pad(x_j, (0, 1))
                x_j = self.train_transform(
                    x_j.view(1, 1, x_j.shape[-1]), sample_rate=self.sample_rate
                )
            return x_i, x_j.flatten()[: int(self.sample_rate * self.cfg["dur"])]

        if self.train:
            X_i = self.logmelspec(x_i)
            assert X_i.device == torch.device("cuda:0"), f"X_i device: {X_i.device}"
            X_j = self.logmelspec(x_j)

        else:
            X_i = self.logmelspec(x_i.squeeze(0)).transpose(1, 0)
            X_i = X_i.unfold(
                0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            )

            if x_j is None:
                # Dummy db does not need augmentation
                return X_i, X_i
            try:
                x_j = self.val_transform(
                    x_j.view(1, 1, x_j.shape[-1]), sample_rate=self.sample_rate
                )
            except ValueError:
                print("Error loading noise file. Retrying...")
                x_j = self.val_transform(
                    x_j.view(1, 1, x_j.shape[-1]), sample_rate=self.sample_rate
                )

            X_j = self.logmelspec(x_j.flatten()).transpose(1, 0)
            X_j = X_j.unfold(
                0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            )

        return X_i, X_j


class GPUTransformNeuralfpM2L(nn.Module):
    """
    Transform includes noise addition and impulse response.
    """

    def __init__(self, cfg, ir_dir, noise_dir, train=True, cpu=False):
        super(GPUTransformNeuralfp, self).__init__()
        self.sample_rate = cfg["fs"]
        self.ir_dir = ir_dir
        self.noise_dir = noise_dir
        self.overlap = cfg["overlap"]
        self.arch = cfg["arch"]
        self.n_frames = cfg["n_frames"]
        self.train = train
        self.cpu = cpu
        self.cfg = cfg

        self.train_transform = Compose(
            [
                ApplyImpulseResponse(ir_paths=self.ir_dir, p=cfg["ir_prob"]),
                AddBackgroundNoise(
                    background_paths=self.noise_dir,
                    min_snr_in_db=cfg["tr_snr"][0],
                    max_snr_in_db=cfg["tr_snr"][1],
                    p=cfg["noise_prob"],
                ),
            ]
        )

        self.val_transform = Compose(
            [
                ApplyImpulseResponse(ir_paths=self.ir_dir, p=1),
                AddBackgroundNoise(
                    background_paths=self.noise_dir,
                    min_snr_in_db=cfg["val_snr"][0],
                    max_snr_in_db=cfg["val_snr"][1],
                    p=1,
                ),
            ]
        )

        self.logmelspec = nn.Sequential(
            MelSpectrogram(
                sample_rate=self.sample_rate,
                win_length=cfg["win_len"],
                hop_length=cfg["hop_len"],
                n_fft=cfg["n_fft"],
                n_mels=cfg["n_mels"],
            ),
            AmplitudeToDB(),
        )
        # default sr = 44100
        self.encdec = EncoderDecoder()

        self.resampler = Resample(cfg["fs"], 44100)

    def forward(self, x_i, x_j):
        if self.cpu:
            try:
                x_j = self.train_transform(
                    x_j.view(1, 1, x_j.shape[-1]), sample_rate=self.sample_rate
                )
            except ValueError:
                print("Error loading noise file. Hack to solve issue...")
                # Increase length of x_j by 1 sample
                x_j = F.pad(x_j, (0, 1))
                x_j = self.train_transform(
                    x_j.view(1, 1, x_j.shape[-1]), sample_rate=self.sample_rate
                )
            return x_i, x_j.flatten()[: int(self.sample_rate * self.cfg["dur"])]

        if self.train:
            # X_i = self.logmelspec(x_i)
            # assert X_i.device == torch.device("cuda:0"), f"X_i device: {X_i.device}"
            # X_j = self.logmelspec(x_j)
            if cfg["fs"] != 44100:
                x_i = self.resampler(x_i)
                x_j = self.resampler(x_j)
            X_i = self.encdec(x_i)
            assert X_i.device == torch.device("cuda:0"), f"X_i device: {X_i.device}"
            X_j = self.encdec(x_j)

        else:
            if cfg["fs"] != 44100:
                x_i = self.resampler(x_i)
                x_j = self.resampler(x_j)
            # X_i = self.logmelspec(x_i.squeeze(0)).transpose(1, 0)
            X_i = self.encdec(x_i.squeeze(0)).squeeze(0).transpose(1, 0)

            # take X_i from shape ([64, 1, 1634]) to ([x, 64, 32])
            # X_i = X_i.unfold(
            #     0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            # )
            X_i = X_i.unfold(
                0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            )

            if x_j is None:
                # Dummy db does not need augmentation
                return X_i, X_i
            try:
                x_j = self.val_transform(
                    x_j.view(1, 1, x_j.shape[-1]), sample_rate=self.sample_rate
                )
            except ValueError:
                print("Error loading noise file. Retrying...")
                x_j = self.val_transform(
                    x_j.view(1, 1, x_j.shape[-1]), sample_rate=self.sample_rate
                )

            # X_j = self.logmelspec(x_j.flatten()).transpose(1, 0)
            X_j = self.encdec(x_j.flatten()).transpose(1, 0)
            X_j = X_j.unfold(
                0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            )

        return X_i, X_j


class GPUTransformSamples(nn.Module):
    """
    Transform includes reverb and chorus.
    """

    def __init__(self, cfg, train=True, cpu=False):
        super(GPUTransformSamples, self).__init__()
        self.train = train
        self.cfg = cfg
        self.sample_rate = cfg["fs"]
        self.overlap = cfg["overlap"]
        self.arch = cfg["arch"]
        self.n_frames = cfg["n_frames"]
        self.cpu = cpu
        self.reverb = Reverb(room_size=0.25)
        self.chorus = Chorus()

        self.train_transform = Pedalboard([self.reverb, self.chorus])
        self.val_transform = Pedalboard([self.reverb, self.chorus])

        self.logmelspec = nn.Sequential(
            MelSpectrogram(
                sample_rate=self.sample_rate,
                win_length=cfg["win_len"],
                hop_length=cfg["hop_len"],
                n_fft=cfg["n_fft"],
                n_mels=cfg["n_mels"],
            ),
            AmplitudeToDB(),
        )

    def forward(self, x_i, x_j):
        if self.cpu:
            return x_i, x_j

        if self.train:
            X_i = self.logmelspec(x_i)
            assert X_i.device == torch.device("cuda:0"), f"X_i device: {X_i.device}"
            X_j = self.logmelspec(x_j)

        else:
            X_i = self.logmelspec(x_i.squeeze(0)).transpose(1, 0)
            X_i = X_i.unfold(
                0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            )

            if x_j is None:
                # Dummy db does not need augmentation
                return X_i, X_i

            X_j = self.logmelspec(x_j.flatten()).transpose(1, 0)
            X_j = X_j.unfold(
                0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            )

        return X_i, X_j


class GPUTransformNeuralSampleid(nn.Module):
    """
    Transform includes pedalboard effects and song mixing.
    Handles batched data and keeps track of song metadata.
    """

    def __init__(self, cfg, train=True, cpu=False):
        super(GPUTransformNeuralSampleid, self).__init__()
        self.sample_rate = cfg["fs"]
        self.train = train
        self.cpu = cpu
        self.cfg = cfg

        # Define effects configuration
        self.effects_config = {
            "chorus": {
                "rate_hz": 1.0,
                "depth": 0.25,
                "centre_delay_ms": 7.0,
                "feedback": 0.0,
                "mix": 0.5,
            },
            "reverb": {
                "room_size": 0.8,
                "damping": 0.1,
                "wet_level": 0.5,
                "dry_level": 0.5,
            },
            "distortion": {"drive_db": 15},
        }

        # Create pedalboard effects
        self.chorus = Chorus(**self.effects_config["chorus"])
        self.reverb = Reverb(**self.effects_config["reverb"])
        self.distortion = Distortion(**self.effects_config["distortion"])

        self.mix_prob = float(cfg.get("mix_prob", 0.95))
        self.mix_gain_range = cfg.get("mix_gain_range", [0.05, 0.5])  # Narrower range
        self.mix_gain_range = [float(i) for i in self.mix_gain_range]

        # Keep melspec transform
        self.logmelspec = nn.Sequential(
            MelSpectrogram(
                sample_rate=self.sample_rate,
                win_length=cfg["win_len"],
                hop_length=cfg["hop_len"],
                n_fft=cfg["n_fft"],
                n_mels=cfg["n_mels"],
            ),
            AmplitudeToDB(),
        )

    def get_transpose_semitones(self, from_key, to_key):
        """
        Calculate semitones needed to transpose from one key to another

        Keys 0-11 are major keys (A to G#)
        Keys 12-23 are minor keys (B to Bb)
        Key -1 is unknown
        """
        if from_key == -1 or to_key == -1:  # If either key is unknown
            print(
                "Warning: Unknown key when trying to calculate transpose, returning 0"
            )
            return 0

        # Convert minor keys to their relative major
        if from_key > 11:  # Minor key
            from_key = (from_key - 7) % 12  # -7 to get relative major
        if to_key > 11:
            to_key = (to_key - 7) % 12

        # Calculate the smallest semitone difference needed
        difference = (to_key - from_key) % 12
        if difference > 6:
            difference -= 12
        return difference

    def analyze_tempo(self, beats_data):
        """Calculate tempo and time between beats"""
        if not beats_data or len(beats_data["times"]) < 2:
            print("Warning: Missing beat data, when trying to calculate tempo")
            return None

        beat_times = beats_data["times"]
        # intervals = np.diff(beat_times)
        intervals = torch.diff(torch.tensor(beat_times))
        # median_interval = np.median(intervals)
        median_interval = torch.median(intervals)
        tempo = 60.0 / median_interval

        return {"tempo": tempo, "beat_interval": median_interval}

    def get_tempo_ratio(self, source_tempo, target_tempo):
        """Calculate ratio to match tempos"""
        if not source_tempo or not target_tempo:
            print("Warning: Missing tempo data, using 1.0 ratio")
            return 1.0

        raw_ratio = target_tempo / source_tempo
        
        # Find the closest power of 2 multiple/divisor that keeps ratio between 0.5 and 2.0
        while raw_ratio > 2.0:
            raw_ratio /= 2.0
        while raw_ratio < 0.5:
            raw_ratio *= 2.0
            
        return raw_ratio

    def find_alignment_beats(self, main_beats, other_beats):
        """
        Try to find downbeats (beat 1) in both segments, fall back to first beats if not found.
        Returns tuple of (main_beat_time, other_beat_time)
        """
        try:
            # Try to find downbeats first
            main_idx = main_beats["numbers"].index(1)
            other_idx = other_beats["numbers"].index(1)
            return main_beats["times"][main_idx], other_beats["times"][other_idx]
        except ValueError:
            # If no downbeat found, use first available beats
            print("Warning: No downbeat found in one or both segments, using first beats")
            return main_beats["times"][0], other_beats["times"][0]

    def process_audio_batch(self, batch_audio, metadata):
        """Process a batch of audio with random effects and mixing"""
        batch_size = batch_audio.shape[0]
        processed_batch = []

        for i in range(batch_size):
            # Convert to numpy for pedalboard processing IS THIS NEEDED? I HOPE NOT
            audio = batch_audio[i].cpu().numpy()  # .cpu().numpy()

            # Create random effect chain for this sample
            active_effects = []
            if torch.rand(1).item() < 0.5:
                active_effects.append(self.chorus)
            if torch.rand(1).item() < 0.5:
                active_effects.append(self.reverb)
            if torch.rand(1).item() < 0.5:
                active_effects.append(self.distortion)

            if active_effects:
                board = Pedalboard(active_effects)
                audio = board.process(audio, self.sample_rate)

            # Mix with another random sample from batch if desired
            if torch.rand(1).item() < 0.95:
                # Choose random sample from batch (not self)
                other_idx = (i + torch.randint(1, batch_size, (1,)).item()) % batch_size
                other_audio = batch_audio[other_idx].cpu().numpy()  # TODO: WHY CPU?

                # Get keys from metadata
                main_key = metadata[i]["key"]
                other_key = metadata[other_idx]["key"]

                # Transpose other audio to match main audio's key
                semitones = self.get_transpose_semitones(other_key, main_key)
                # Im transposing at the same time as time stretching
                # if semitones != 0:
                #     pitch_shifter = Pedalboard([PitchShift(semitones=semitones)])
                #     other_audio = pitch_shifter.process(other_audio, self.sample_rate)

                # Get beats information and segment start times
                main_beats = metadata[i].get("beats")
                other_beats = metadata[other_idx].get("beats")

                # Start time of the segment in samples
                main_start = metadata[i].get("start_i")
                other_start = metadata[other_idx].get("start_j")

                # Convert sample indices to seconds
                main_start_sec = main_start / self.sample_rate
                other_start_sec = other_start / self.sample_rate

                # print("Main start", main_start)
                # print("Other start", other_start)
                # print("Main start sec", main_start_sec)
                # print("Other start sec", other_start_sec)
                # print("Main beats", main_beats)
                # print("Other beats", other_beats)

                # Cut beats to next beat after start until next after start+duration
                first_beat_i = np.searchsorted(main_beats["times"], main_start_sec)
                last_beat_i = np.searchsorted(
                    main_beats["times"], main_start_sec + len(audio) / self.sample_rate
                )
                main_beats = {
                    "times": (
                        np.array(main_beats["times"][first_beat_i:last_beat_i])
                        - main_start_sec
                    ).tolist(),
                    "numbers": main_beats["numbers"][first_beat_i:last_beat_i],
                }

                first_beat_i = np.searchsorted(other_beats["times"], other_start_sec)
                last_beat_i = np.searchsorted(
                    other_beats["times"],
                    other_start_sec + len(audio) / self.sample_rate,
                )
                other_beats = {
                    "times": (
                        np.array(other_beats["times"][first_beat_i:last_beat_i])
                        - other_start_sec
                    ).tolist(),
                    "numbers": other_beats["numbers"][first_beat_i:last_beat_i],
                }

                # print("Main beats", main_beats)
                # print("Other beats", other_beats)

                tempo_ratio = 1.0
                # If we have more than 2 beats, calculate tempo and time stretch
                if len(main_beats["times"]) > 2 and len(other_beats["times"]) > 2:
                    # Calculate tempos if we have beat information
                    main_tempo_data = self.analyze_tempo(main_beats)
                    other_tempo_data = self.analyze_tempo(other_beats)

                    # Handle beat synchronization
                    # if main_tempo_data and other_tempo_data:
                    # Calculate tempo ratio for time stretching
                    tempo_ratio = self.get_tempo_ratio(
                        other_tempo_data["tempo"], main_tempo_data["tempo"]
                    )
                    # If tempo_ratio is a tensor, convert to float
                    if isinstance(tempo_ratio, torch.Tensor):
                        tempo_ratio = tempo_ratio.item()

                    # Time stretch using pedalboard if needed
                    if (
                        abs(1 - tempo_ratio) > 0.02
                    ):  # Only stretch if difference is significant
                        # Convert to audio segment and stretch
                        other_audio = time_stretch(
                            input_audio=other_audio,
                            samplerate=self.sample_rate,
                            stretch_factor=tempo_ratio,
                            pitch_shift_in_semitones=semitones,
                            high_quality=True,
                            transient_mode="crisp",
                            transient_detector="compound",
                        )[0]
                        # Stretch beats times
                        other_beats["times"] = [
                            t * tempo_ratio for t in other_beats["times"]
                        ]

                    # Find alignment points (downbeats or first beats)
                    main_nearest_beat, other_nearest_beat = self.find_alignment_beats(main_beats, other_beats)
        
                    # print("Main nearest beat", main_nearest_beat)
                    # print("Other nearest beat", other_nearest_beat)

                    # Calculate offset needed to align beats
                    offset = int(
                        (main_nearest_beat - other_nearest_beat) * self.sample_rate
                    )

                    # print("Offset", offset)

                    # Apply offset and padding/trimming to same length
                    target_length = len(audio)
                    if offset >= 0:
                        # Add offset zeros at the start
                        other_audio = np.pad(other_audio, (offset, 0))
                        # Then trim/pad to target length
                        if len(other_audio) > target_length:
                            other_audio = other_audio[:target_length]
                        else:
                            other_audio = np.pad(
                                other_audio, (0, target_length - len(other_audio))
                            )
                    elif offset < 0:
                        other_audio = other_audio[-offset:]
                        if len(other_audio) > target_length:
                            # If longer than target, trim the end
                            other_audio = other_audio[:target_length]
                        else:
                            # If shorter than target, pad the end
                            other_audio = np.pad(
                                other_audio, (0, target_length - len(other_audio))
                            )
                    
                    # Verify lengths match before mixing
                    assert len(audio) == len(other_audio), f"Length mismatch: {len(audio)} vs {len(other_audio)}, after messing with time stretch. Offset was: {offset}."

                else:
                    # If we don't have enough beats, just transpose and mix
                    print(
                        "Warning: Not enough beats in one of the segments to calculate tempo, transposing only"
                    )

                    main_tempo_data = None
                    other_tempo_data = None
                    offset = None
                    # Transpose other audio to match main audio's key
                    if semitones != 0:
                        pitch_shifter = Pedalboard([PitchShift(semitones=semitones)])
                        other_audio = pitch_shifter.process(
                            other_audio, self.sample_rate
                        )
                    # Verify lengths match before mixing
                    assert len(audio) == len(other_audio), f"Length mismatch: {len(audio)} vs {len(other_audio)}, WITHOUT messing with time stretch"


                # Normalize both audios before mixing
                audio = audio / np.abs(audio).max() + 1e-8
                other_audio = other_audio / np.abs(other_audio).max() + 1e-8

                # Random gain between 0.05 and 0.5
                gain = (
                    torch.rand(1).item()
                    * (self.mix_gain_range[1] - self.mix_gain_range[0])
                    + self.mix_gain_range[0]
                )
                # print("Audio shape", audio.shape)
                # print("Other audio shape", other_audio.shape)
                # print("Gain", gain)

                audio = (1 - gain) * audio + gain * other_audio

                # Update metadata to note the mixing
                metadata[i].update(
                    {
                        "mixed_with": metadata[other_idx]["file_path"],
                        "transpose_semitones": semitones,
                        "mix_gain": gain,
                        "tempo_ratio": tempo_ratio,
                        "main_tempo": (
                            main_tempo_data["tempo"] if main_tempo_data else None
                        ),
                        "other_tempo": (
                            other_tempo_data["tempo"] if other_tempo_data else None
                        ),
                        "beat_offset_samples": (
                            offset if main_beats and other_beats else None
                        ),
                    }
                )

            # Normalize the final mix
            audio = (audio / np.abs(audio).max() + 1e-8)
            processed_batch.append(torch.from_numpy(audio))

        return torch.stack(processed_batch).to(batch_audio.device)

    def forward(self, x_i, x_j, metadata=None):
        """
        Args:
            x_i: First set of audio segments [B, T]
            x_j: Second set of audio segments [B, T]
            metadata: List of dicts containing info about each sample
        """
        if self.cpu:
            return x_i, x_j, metadata

        if self.train:
            x_j_processed = self.process_audio_batch(x_j, metadata)


            # Add safety checks
            if torch.isnan(x_j_processed).any():
                print("Warning: NaN detected in processed batch")
                # Return original audio as fallback
                x_j_processed = x_j


            # Convert to mel spectrograms
            X_i = self.logmelspec(x_i)
            # X_i = self.logmelspec(x_i_processed)
            X_j = self.logmelspec(x_j_processed)
            # X_i = x_i
            # X_j = x_j_processed


            # Add more safety checks
            if torch.isnan(X_i).any() or torch.isnan(X_j).any():
                print("Warning: NaN detected in spectrograms")

            # Update metadata with mixing info if needed
            if metadata is not None:
                for meta in metadata:
                    meta["augmented"] = True

            return X_i, X_j, metadata

        else:
            # Validation
            # SHOULDNT THIS BE AUGMENTED?
            X_i = self.logmelspec(x_i.squeeze(0)).transpose(1, 0)
            X_i = X_i.unfold(
                0,
                size=self.cfg["n_frames"],
                step=int(self.cfg["n_frames"] * (1 - self.cfg["overlap"])),
            )

            if x_j is None:
                return X_i, X_i, metadata

            X_j = self.logmelspec(x_j.squeeze(0)).transpose(1, 0)
            X_j = X_j.unfold(
                0,
                size=self.cfg["n_frames"],
                step=int(self.cfg["n_frames"] * (1 - self.cfg["overlap"])),
            )

            return X_i, X_j, metadata
