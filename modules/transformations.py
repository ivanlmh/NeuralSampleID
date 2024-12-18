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

from pedalboard import Pedalboard, Chorus, Reverb, Distortion
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
