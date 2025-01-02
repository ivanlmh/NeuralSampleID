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
        super(GPUTransformNeuralfpM2L, self).__init__()
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
            if self.sample_rate != 44100:
                x_i = self.resampler(x_i)
                x_j = self.resampler(x_j)
            X_i = self.encdec.encode(x_i)
            assert X_i.device == torch.device("cuda:0"), f"X_i device: {X_i.device}"
            X_j = self.encdec.encode(x_j)

        else:
            X_i = self.logmelspec(x_i.squeeze(0)).transpose(1, 0)
            # print("mel shape xi", X_i.shape)
            X_i = X_i.unfold(
                0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            )
            # print("mel shape xi", X_i.shape)

            if self.sample_rate != 44100:
                x_i = self.resampler(x_i)
            X_i = self.encdec.encode(x_i.squeeze(0)).squeeze(0).transpose(1, 0)
            # print("m2l shape xi", X_i.shape)
            # take from shape ([64, 1, 1634]) to 
            X_i = X_i.unfold(
                0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            )
            # print("m2l shape xi", X_i.shape)

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
            # print("mel shape xj", X_j.shape)
            X_j = X_j.unfold(
                0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            )
            # print("mel shape xj", X_j.shape)
            if self.sample_rate != 44100:
                x_j = self.resampler(x_j)
            X_j = self.encdec.encode(x_j.flatten()).squeeze(0).transpose(1, 0)
            # print("m2l shape xj", X_j.shape)
            X_j = X_j.unfold(
                0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            )
            # print("m2l shape xj", X_j.shape)

        return X_i, X_j


# class GPUTransformSamples(nn.Module):
#     """
#     Transform includes reverb and chorus.
#     """

#     def __init__(self, cfg, train=True, cpu=False):
#         super(GPUTransformSamples, self).__init__()
#         self.train = train
#         self.cfg = cfg
#         self.sample_rate = cfg["fs"]
#         self.overlap = cfg["overlap"]
#         self.arch = cfg["arch"]
#         self.n_frames = cfg["n_frames"]
#         self.cpu = cpu
#         self.reverb = Reverb(room_size=0.25)
#         self.chorus = Chorus()

#         self.train_transform = Pedalboard([self.reverb, self.chorus])
#         self.val_transform = Pedalboard([self.reverb, self.chorus])

#         self.logmelspec = nn.Sequential(
#             MelSpectrogram(
#                 sample_rate=self.sample_rate,
#                 win_length=cfg["win_len"],
#                 hop_length=cfg["hop_len"],
#                 n_fft=cfg["n_fft"],
#                 n_mels=cfg["n_mels"],
#             ),
#             AmplitudeToDB(),
#         )

#     def forward(self, x_i, x_j):
#         if self.cpu:
#             return x_i, x_j

#         if self.train:
#             X_i = self.logmelspec(x_i)
#             assert X_i.device == torch.device("cuda:0"), f"X_i device: {X_i.device}"
#             X_j = self.logmelspec(x_j)

#         else:
#             X_i = self.logmelspec(x_i.squeeze(0)).transpose(1, 0)
#             X_i = X_i.unfold(
#                 0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
#             )

#             if x_j is None:
#                 # Dummy db does not need augmentation
#                 return X_i, X_i

#             X_j = self.logmelspec(x_j.flatten()).transpose(1, 0)
#             X_j = X_j.unfold(
#                 0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
#             )

#         return X_i, X_j
class GPUTransformSamples(nn.Module):
    """
    Transform includes time stretching, pitch shifting, distortion, reverb and chorus.
    Uses Pedalboard library for audio effects and transforms.
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

        # Define effects with default parameters
        self.reverb = Reverb(
            room_size=0.8,
            damping=0.1, 
            wet_level=0.5,
            dry_level=0.5
        )
        self.chorus = Chorus(
            rate_hz=1.0,
            depth=0.25,
            centre_delay_ms=7.0,
            feedback=0.0,
            mix=0.5
        )
        self.distortion = Distortion(drive_db=25)

        # Create pedalboard chain for training and validation
        self.train_transform = Pedalboard([
            self.reverb,
            self.chorus,
            self.distortion
        ])
        self.val_transform = Pedalboard([
            self.reverb,
            self.chorus,
            self.distortion
        ])

        # Create mel spectrogram transform
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

    def apply_random_time_stretch(self, audio):
        """Apply random time stretching between 0.8x and 1.2x"""
        stretch_factor = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
        return time_stretch(audio, self.sample_rate, stretch_factor)

    def apply_random_pitch_shift(self, audio):
        """Apply random pitch shifting between -4 and +4 semitones"""
        pitch_shift = Pedalboard([PitchShift(
            semitones=torch.FloatTensor(1).uniform_(-4, 4).item()
        )])
        return pitch_shift.process(audio, self.sample_rate)

    def forward(self, x_i, x_j):
        if self.cpu:
            return x_i, x_j

        if self.train:
            # Apply random augmentations during training
            if torch.rand(1).item() < 0.5:
                x_i = self.apply_random_time_stretch(x_i)
            if torch.rand(1).item() < 0.5:
                x_j = self.apply_random_time_stretch(x_j)
            
            if torch.rand(1).item() < 0.5:
                x_i = self.apply_random_pitch_shift(x_i)
            if torch.rand(1).item() < 0.5:
                x_j = self.apply_random_pitch_shift(x_j)

            # Apply effects chain
            x_i = self.train_transform.process(x_i, self.sample_rate)
            x_j = self.train_transform.process(x_j, self.sample_rate)

            # Convert to mel spectrograms
            X_i = self.logmelspec(x_i)
            assert X_i.device == torch.device("cuda:0"), f"X_i device: {X_i.device}"
            X_j = self.logmelspec(x_j)

        else:
            # Validation/test mode
            X_i = self.logmelspec(x_i.squeeze(0)).transpose(1, 0)
            X_i = X_i.unfold(
                0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            )

            if x_j is None:
                # Dummy db does not need augmentation
                return X_i, X_i

            # Apply validation transform chain
            x_j = self.val_transform.process(x_j.flatten(), self.sample_rate)
            
            X_j = self.logmelspec(x_j).transpose(1, 0)
            X_j = X_j.unfold(
                0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap))
            )

        return X_i, X_j