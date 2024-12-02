import os
import numpy as np

import IPython.display as ipd

from pedalboard import (
    Compressor,
    Gain,
    Pedalboard,
    Reverb,
    Distortion,
    Delay,
    time_stretch,
    PitchShift,
)  # , Limiter, NoiseGate,

import librosa
import numpy as np

import pyloudnorm as pyln

import soundfile as sf
import json

# import to save as mp3
import pydub

import pandas as pd

import soundfile as sf
import json
import pydub
import os
import shutil
import random


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
# sort alphabetically


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


def effect_sample(sample, sr, cfg):
    assert cfg["gain"] is not None, "Gain is required"
    gain = Gain(gain_db=cfg["gain"]["gain_db"])
    pedalboard = Pedalboard([gain])
    if cfg["effects"]:
        if cfg["effects"]["compressor"]:
            compressor = Compressor(
                threshold_db=cfg["effects"]["compressor"]["threshold_db"],
                ratio=cfg["effects"]["compressor"]["ratio"],
                attack_ms=cfg["effects"]["compressor"]["attack_ms"],
                release_ms=cfg["effects"]["compressor"]["release_ms"],
            )
            pedalboard.append(compressor)
        if cfg["effects"]["reverb"]:
            reverb = Reverb(
                room_size=cfg["effects"]["reverb"]["room_size"],
                damping=cfg["effects"]["reverb"]["damping"],
                wet_level=cfg["effects"]["reverb"]["wet_level"],
                dry_level=cfg["effects"]["reverb"]["dry_level"],
            )
            pedalboard.append(reverb)
        # if cfg["effects"]["limiter"]:
        #     limiter = Limiter(threshold_db=cfg["effects"]["limiter"]["threshold_db"])
        #     pedalboard.append(limiter)
        # if cfg["effects"]["noise_gate"]:
        #     noise_gate = NoiseGate(threshold_db=cfg["effects"]["noise_gate"]["threshold_db"], attack_ms=cfg["effects"]["noise_gate"]["attack_ms"], release_ms=cfg["effects"]["noise_gate"]["release_ms"])
        #     pedalboard.append(noise_gate)
        if cfg["effects"]["delay"]:
            delay = Delay(
                delay_seconds=cfg["effects"]["delay"]["delay_seconds"],
                feedback=cfg["effects"]["delay"]["feedback"],
                mix=cfg["effects"]["delay"]["mix"],
            )
            pedalboard.append(delay)
        if cfg["effects"]["distortion"]:
            distortion = Distortion(drive_db=cfg["effects"]["distortion"]["drive_db"])
            pedalboard.append(distortion)

    return pedalboard.process(sample, sr)


def loudness_normalize(y, sr, target_loudness=-12.0):
    # peak_normalized_y = pyln.normalize.peak(y, -1.0)
    meter = pyln.Meter(
        sr, block_size=0.200
    )  # create BS.1770 meter, default block size is 400ms
    loudness = meter.integrated_loudness(y)
    y = pyln.normalize.loudness(y, loudness, target_loudness)
    return y


# create song by adding sample to base
def create_song(base, sample_source, cfg=None):
    # print(f"Peak value of base: {np.max(base)}, Peak value of sample: {np.max(sample)}")
    sample_tempo, sample_beats = librosa.beat.beat_track(y=sample_source, sr=sr)
    print(f"Sample tempo: {sample_tempo}")
    base_tempo, base_beats = librosa.beat.beat_track(y=base, sr=sr)
    assert cfg is not None, "Configuration is required"

    sample_time_annotation = []
    base_time_annotations = []

    # extract sample from source
    print("Extracting sample from sample source...")
    assert cfg["sample_start"] is not None, "Sample start is required"
    assert cfg["sample_length"] is not None, "Sample length is required"
    if cfg["sample_start"]["seconds"] is not None:
        sample_source = sample_source[cfg["sample_start"]["seconds"] * sr :]
        sample_time_annotation.append(
            {"start_time": cfg["sample_start"]["seconds"], "end_time": None}
        )
    elif cfg["sample_start"]["beats"] is not None:
        sample_source = sample_source[
            librosa.frames_to_samples(sample_beats[cfg["sample_start"]["beats"]]) :
        ]
        sample_time_annotation.append(
            {
                "start_time": librosa.frames_to_time(
                    sample_beats[cfg["sample_start"]["beats"]], sr=sr
                ),
                "end_time": None,
            }
        )
        sample_beats -= sample_beats[cfg["sample_start"]["beats"]]
    if cfg["sample_length"]["seconds"] is not None:
        sample = sample_source[: cfg["sample_length"]["seconds"] * sr]
        sample_time_annotation[-1]["end_time"] = cfg["sample_length"]["seconds"]
    elif cfg["sample_length"]["beats"] is not None:
        try:
            sample = sample_source[
                : librosa.frames_to_samples(
                    sample_beats[
                        cfg["sample_start"]["beats"] + cfg["sample_length"]["beats"]
                    ]
                )
            ]
            sample_time_annotation[-1]["end_time"] = sample_time_annotation[-1][
                "start_time"
            ] + librosa.frames_to_time(
                sample_beats[
                    cfg["sample_start"]["beats"] + cfg["sample_length"]["beats"]
                ],
                sr=sr,
            )
        except IndexError:
            print(
                "Sample length exceeds sample source length. Truncating sample to end of source."
            )
            sample = sample_source[:]
            sample_time_annotation[-1]["end_time"] = sample_time_annotation[-1][
                "start_time"
            ] + librosa.frames_to_time(len(sample), sr=sr)

    # loudness normalize audio to -12 dB LUFS
    base = loudness_normalize(base, sr, target_loudness=-12.0)
    sample = loudness_normalize(sample, sr, target_loudness=-12.0)
    original_sample = sample

    # half the volume
    gain = Gain(-6)

    board = Pedalboard([gain])

    base = board.process(base, sr)
    sample = board.process(sample, sr)

    # base_start_sample = 0
    pitch_shift_semitones = 0

    # if cfg["masking"]:
    #     # take the signal to silence for n beats
    #     pass

    if cfg["tempo_sync"]:
        stretch_rate = (base_tempo / sample_tempo)[0] * cfg["tempo_sync"]["multiplier"]
        cfg["total_tempo_stretch_rate"] = float(stretch_rate)
        sample = time_stretch(sample, sr, stretch_rate)[0]

    if cfg["tempo_change"] != 1:
        sample = time_stretch(sample, sr, cfg["tempo_change"])[0]

    if cfg["pitch_sync"]:
        sample_chromagram = librosa.feature.chroma_stft(y=sample, sr=sr)
        base_chromagram = librosa.feature.chroma_stft(y=base, sr=sr)
        mean_sample_chroma = np.mean(sample_chromagram, axis=1)
        mean_base_chroma = np.mean(base_chromagram, axis=1)
        sample_estimated_key = np.argmax(mean_sample_chroma)
        base_estimated_key = np.argmax(mean_base_chroma)
        pitch_shift_semitones = base_estimated_key - sample_estimated_key

    if cfg["pitch_shift"]:
        pitch_shift_semitones += cfg["pitch_shift"]

    cfg["total_pitch_shift"] = int(pitch_shift_semitones)

    pitch_shift = Pedalboard([PitchShift(semitones=pitch_shift_semitones)])
    sample = pitch_shift.process(sample, sr)

    print("Applying effects to sample...")
    sample = effect_sample(sample, sr, cfg)
    effected_sample = sample

    mix = base
    # mix[:len(sample)] += sample
    assert (
        cfg["looping"] is not None
    ), "Looping configuration is required (even if n_loops=1)"
    start_beat = cfg["beat_sync"]["start_beat"]
    for _ in range(cfg["looping"]["n_loops"]):
        loop_start_sample = librosa.frames_to_samples(base_beats[start_beat])
        mix[loop_start_sample : loop_start_sample + len(sample)] += sample

        start_beat += cfg["looping"]["loop_every_n_beats"]

        base_time_annotations.append(
            {
                "start_time": librosa.frames_to_time(base_beats[start_beat], sr=sr),
                "end_time": librosa.frames_to_time(
                    base_beats[start_beat] + librosa.samples_to_frames(len(sample)),
                    sr=sr,
                ),
            }
        )

        if (
            cfg["looping"]["n_loops"] != 1
            and cfg["looping"]["loop_every_n_beats"] < cfg["sample_length"]["beats"]
        ):
            print(
                "Warning: Loops will overlap as sample_length in beats is greater than loop_every_n_beats"
            )

        # # # find closest beat to start_sample
        # start_beat += cfg["looping"]["loop_every_n_beats"]
        # # loop_start_sample = librosa.frames_to_samples(base_beats[end_beat + cfg["looping"]["loop_every_n_beats"]])
        # loop_start_sample = librosa.frames_to_samples(base_beats[start_beat])

    mix[len(sample) :] += base[len(sample) :]
    # mix[base_start_sample+len(sample):] += base[base_start_sample+len(sample):]

    mix = loudness_normalize(mix, sr, target_loudness=-12.0)

    return (
        mix,
        original_sample,
        effected_sample,
        base_time_annotations,
        sample_time_annotation,
        cfg,
    )


cfg = {
    "sample_start": {"seconds": None, "beats": 0},  # 0,  # 24
    "sample_length": {"seconds": None, "beats": 16},  # 5,
    "gain": {"gain_db": 0},
    "effects": {
        "compressor": False,
        "reverb": False,
        "delay": False,
        "noise_gate": False,
        "distortion": False,
        "limiter": False,
        "noise_gate": False,
    },
    "beat_sync": {
        "start_beat": 12,  # 0 is the first beat
    },
    "tempo_sync": False,
    "tempo_change": 1,  # 1 is no change
    "pitch_sync": False,  # detects key of the base track and adjusts pitch of sample to match
    "pitch_shift": 0,  # in semitones
    "looping": {"n_loops": 2, "loop_every_n_beats": 16},  # 1 is no loop
}

# [4 suno base tracks (Pop, HipHop/Rap, EDM-instrumental, House-instrumental) + 1 stableAudio base track (EDM-instrumental)] x 10 samples from sample100 sample set (4 drum-based beats, 4 instrumental riffs, 2 with vocals) x 30 augmentations
# = 1500 tracks
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

print(f"Creating {len(config_array)} songs...")

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


# create folder to save songs
# go to file directory
os.chdir("/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/")
os.makedirs(f"artificial_dataset_v2", exist_ok=True)
os.chdir(f"artificial_dataset_v2")
os.makedirs("audio", exist_ok=True)
os.makedirs("audio_wav", exist_ok=True)

# all the base-samples combinations
version = 0
for config in config_array:
    for base_index, base_track in enumerate(base_tracks):
        for sample_index, sample_track in enumerate(sample_tracks):
            print(f"Creating song {base_index}_{sample_index}_v{version}...")
            # if json exists, skip
            if os.path.exists(f"config_{base_index}_{sample_index}_v{version}.json"):
                print(
                    f"config_{base_index}_{sample_index}_v{version}.json already exists. Skipping..."
                )
                continue
            print(f"Using base track {base_tracks_paths[base_index]}")
            print(f"Using sample track {sample_tracks_paths[sample_index]}")
            aux_cfg = cfg.copy()
            aux_cfg.update(config)
            (
                mix,
                original_sample,
                effected_sample,
                base_time_annotations,
                sample_time_annotation,
                aux_cfg,
            ) = create_song(base_track, sample_track, aux_cfg)

            # save audio together with config used
            # save audio as wav
            sf.write(
                f"audio_wav/mix_{base_index}_{sample_index}_v{version}.wav", mix, sr
            )
            sf.write(
                f"audio_wav/original_sample_{base_index}_{sample_index}_v{version}.wav",
                original_sample,
                sr,
            )
            sf.write(
                f"audio_wav/effected_sample_{base_index}_{sample_index}_v{version}.wav",
                effected_sample,
                sr,
            )
            # convert to mp3
            sound = pydub.AudioSegment.from_wav(
                f"audio_wav/mix_{base_index}_{sample_index}_v{version}.wav"
            )
            sound.export(
                f"audio/mix_{base_index}_{sample_index}_v{version}.mp3", format="mp3"
            )
            print(f"Created mix_{base_index}_{sample_index}_v{version}.mp3")
            sound = pydub.AudioSegment.from_wav(
                f"audio_wav/original_sample_{base_index}_{sample_index}_v{version}.wav"
            )
            sound.export(
                f"audio/original_sample_{base_index}_{sample_index}_v{version}.mp3",
                format="mp3",
            )
            print(f"Created original_sample_{base_index}_{sample_index}_v{version}.mp3")
            sound = pydub.AudioSegment.from_wav(
                f"audio_wav/effected_sample_{base_index}_{sample_index}_v{version}.wav"
            )
            sound.export(
                f"audio/effected_sample_{base_index}_{sample_index}_v{version}.mp3",
                format="mp3",
            )
            print(f"Created effected_sample_{base_index}_{sample_index}_v{version}.mp3")

            # add time annotations to config
            aux_cfg["base_time_annotations"] = base_time_annotations
            aux_cfg["sample_time_annotations"] = sample_time_annotation
            # add base file name to config
            aux_cfg["base_file"] = base_tracks_paths[base_index]
            aux_cfg["sample_file"] = sample_tracks_paths[sample_index]
            # save config
            with open(f"config_{base_index}_{sample_index}_v{version}.json", "w") as f:
                json.dump(aux_cfg, f)
                print("Config saved")

            print(f"Song {base_index}_{sample_index}_v{version} created!")

    version += 1

# all the base-separated_samples combinations
for base_index, base_track in enumerate(base_tracks):
    for sample_index, sample_track in enumerate(separated_sample_tracks):
        print(f"Creating song {base_index}_{sample_index}_v{version}...")
        if os.path.exists(f"config_{base_index}_{sample_index}_v{version}.json"):
            print(
                f"config_{base_index}_{sample_index}_v{version}.json already exists. Skipping..."
            )
            continue
        print(f"Using base track {base_tracks_paths[base_index]}")
        print(f"Using sample track {separated_sample_tracks_paths[sample_index]}")
        aux_cfg = cfg.copy()
        (
            mix,
            original_sample,
            effected_sample,
            base_time_annotations,
            sample_time_annotation,
            aux_cfg,
        ) = create_song(base_track, sample_track, aux_cfg)

        # save audio together with config used
        # save audio as wav
        sf.write(f"audio_wav/mix_{base_index}_{sample_index}_v{version}.wav", mix, sr)
        sf.write(
            f"audio_wav/original_sample_{base_index}_{sample_index}_v{version}.wav",
            original_sample,
            sr,
        )
        sf.write(
            f"audio_wav/effected_sample_{base_index}_{sample_index}_v{version}.wav",
            effected_sample,
            sr,
        )
        # convert to mp3
        sound = pydub.AudioSegment.from_wav(
            f"audio_wav/mix_{base_index}_{sample_index}_v{version}.wav"
        )
        sound.export(
            f"audio/mix_{base_index}_{sample_index}_v{version}.mp3", format="mp3"
        )
        sound = pydub.AudioSegment.from_wav(
            f"audio_wav/original_sample_{base_index}_{sample_index}_v{version}.wav"
        )
        sound.export(
            f"audio/original_sample_{base_index}_{sample_index}_v{version}.mp3",
            format="mp3",
        )
        sound = pydub.AudioSegment.from_wav(
            f"audio_wav/effected_sample_{base_index}_{sample_index}_v{version}.wav"
        )
        sound.export(
            f"audio/effected_sample_{base_index}_{sample_index}_v{version}.mp3",
            format="mp3",
        )

        # add time annotations to config
        aux_cfg["base_time_annotations"] = base_time_annotations
        aux_cfg["sample_time_annotations"] = sample_time_annotation
        # add base file name to config
        aux_cfg["base_file"] = base_tracks_paths[base_index]
        aux_cfg["sample_file"] = separated_sample_tracks_paths[sample_index]
        # save config
        with open(f"config_{base_index}_{sample_index}_v{version}.json", "w") as f:
            json.dump(aux_cfg, f)

        print(f"Song {base_index}_{sample_index}_v{version} created!")

    version += 1


# save df that maps version to config used
df = pd.DataFrame(config_array)
# add version column
df["version"] = list(range(len(config_array)))
# add separated_samples as extra rows of augmentation
df = pd.concat(
    [df] * len(base_tracks) * len(separated_sample_tracks), ignore_index=True
)
df.to_csv("config_versions.csv", index=False)

# save df that maps base and sample files to index
df = pd.DataFrame({"base_files": base_tracks_paths})
df["base_index"] = df.index
# concatenate with sample files
df2 = pd.DataFrame({"sample_files": sample_tracks_paths})
df2["sample_index"] = df2.index
# and separated sample files
df3 = pd.DataFrame({"sample_files": separated_sample_tracks_paths})
df3["separated_sample_index"] = df3.index
# concatenate
df = pd.concat([df, df2, df3], axis=1)

df.to_csv("base_sample_files.csv", index=False)


# # Ok, now I want to create a subset of the artificial dataset which I moved to /home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2/
# # create folder to save subset
# os.makedirs(
#     f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2_subset",
#     exist_ok=True,
# )
# # create folder to save subset audio
# os.makedirs(
#     f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2_subset/audio",
#     exist_ok=True,
# )

# # move the "mix" mp3 files to the subset folder
# for file in os.listdir(
#     "/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2/audio"
# ):
#     if "mix" in file:
#         shutil.move(
#             f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2/audio/{file}",
#             f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2_subset/audio/{file}",
#         )

# # and for a random 25% i want to move the effect and original sample files
# files = os.listdir(
#     "/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2/audio"
# )
# # keep only files that are effected or original samples
# files = [file for file in files if "effected" in file or "original" in file]
# print(len(files))

# # seed 42
# random.seed(42)
# random.shuffle(files)
# files = files[: int(len(files) * 0.25)]

# for file in files:
#     shutil.move(
#         f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2/audio/{file}",
#         f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2_subset/audio/{file}",
#     )

#     # and the config files for those files
#     shutil.move(
#         f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2/config_{file.split('_')[2]}_{file.split('_')[3]}_{file.split('_')[4].split('.')[0]}.json",
#         f"/home/ivanmeresman-higgs/Documents/QueenMary/tracks_for_stage/artificial_dataset_v2_subset/config_{file.split('_')[2]}_{file.split('_')[3]}_{file.split('_')[4].split('.')[0]}.json",
#     )
