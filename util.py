import os
import torch
import numpy as np
import json
import glob
import soundfile as sf
import shutil
import yaml
from prettytable import PrettyTable


class DummyScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        return optimizer.step()

    def update(self):
        pass


def load_index(
    cfg, data_dir, ext=["mp3"], shuffle_dataset=True, mode="train", stem=None
):  # "wav", "mp3"], shuffle_dataset=True, mode="train"):
    if data_dir.endswith(".json"):
        print(f"=>Loading indices from index file {data_dir}")
        with open(data_dir, "r") as fp:
            dataset = json.load(fp)
        return dataset

    print(f"=>Loading indices from {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found")

    if stem is None:
        # json path for saving indices
        json_path = os.path.join(
            cfg["data_dir"], os.path.normpath(data_dir.split("/")[-1]) + ".json"
        )
    else:
        print(f"Going to load indices with stem {stem}")
        json_path = os.path.join(
            cfg["data_dir"], os.path.normpath(data_dir.split("/")[-1]) + f"_{stem}.json"
        )

    if os.path.exists(json_path):
        print(f"Loading indices from {json_path}")
        with open(json_path, "r") as fp:
            dataset = json.load(fp)
        return dataset

    fpaths = glob.glob(os.path.join(data_dir, "**/*.*"), recursive=True)
    fpaths = [p for p in fpaths if p.split(".")[-1] in ext]

    if stem is not None:
        print(f"Filtering files with stem {stem}")
        fpaths = [p for p in fpaths if stem in p]
    else:
        # If no stem is specified, discard files in htdemucs subdirectory
        print("No stem specified, discarding htdemucs files")
        fpaths = [p for p in fpaths if "htdemucs" not in p]

    dataset_size = len(fpaths)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(42)
        np.random.shuffle(indices)
    if mode == "train":
        size = cfg["train_sz"]
    else:
        size = cfg["val_sz"]
    # returns a dictionary, the key is the index and the value is the file path
    dataset = {str(i): fpaths[ix] for i, ix in enumerate(indices[:size])}

    with open(json_path, "w") as fp:
        json.dump(dataset, fp)

    return dataset


def load_sample100_index(
    cfg, data_dir, ext=["mp3"], shuffle_dataset=True, mode="train"
):  # , mode="train"):
    print("Called load_sample100_index with:", data_dir)
    # verify data_dir is called sample_100, has an audio subdirectory and a samples.csv file
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found")
    if not os.path.exists(os.path.join(data_dir, "audio")):
        raise FileNotFoundError(
            f"Directory {data_dir} does not contain an audio subdirectory"
        )

    if data_dir.endswith(".json"):
        print(f"=>Loading indices from index file {data_dir}")
        with open(data_dir, "r") as fp:
            dataset = json.load(fp)
        return dataset
    
    # Add stem handling
    stem = cfg.get('stem', None)  # Get stem from config if it exists
    if stem:
        json_path = os.path.join(
            cfg["data_dir"],
            os.path.normpath(data_dir.split("/")[-1]) + f"_querysamples_{stem}.json",
        )
    else:
        json_path = os.path.join(
            cfg["data_dir"],
            os.path.normpath(data_dir.split("/")[-1]) + "_querysamples.json",
        )

    if os.path.exists(json_path):
        print(f"Loading indices from {json_path}")
        with open(json_path, "r") as fp:
            dataset = json.load(fp)
        return dataset

    print(f"=>Loading indices from {data_dir}")

    if not os.path.exists(os.path.join(data_dir, "samples.csv")):
        raise FileNotFoundError(
            f"Directory {data_dir} does not contain a samples.csv file"
        )

    # csv columns are sample_id,original_track_id,sample_track_id,t_original,t_sample,n_repetitions,sample_type,interpolation,comments
    # fpaths are data_dir/audio/{sample_track_id}.mp3
    # open csv encode as latin-1
   # Adjust file paths to include stem subdirectory if specified
    with open(os.path.join(data_dir, "samples.csv"), encoding="latin-1") as f:
        lines = f.readlines()
        if stem:
            # Modify paths to point to stem files in htdemucs subdirectory
            fpaths_sample = [
                os.path.join(data_dir, "htdemucs", line.split(",")[2], f"{stem}.mp3")
                for line in lines[1:]
            ]
            fpaths_original = [
                os.path.join(data_dir, "htdemucs", line.split(",")[1], f"{stem}.mp3")
                for line in lines[1:]
            ]
        else:
            fpaths_sample = [
                os.path.join(data_dir, "audio", line.split(",")[2] + ".mp3")
                for line in lines[1:]
            ]
            fpaths_original = [
                os.path.join(data_dir, "audio", line.split(",")[1] + ".mp3")
                for line in lines[1:]
            ]
            
    # assert file exists
    for fpath in fpaths_sample + fpaths_original:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File {fpath} does not exist")

    assert len(fpaths_sample) == len(
        fpaths_original
    ), "Different number of sample and original files"
    dataset_size = len(fpaths_sample)
    # indices = list(range(dataset_size))
    # if shuffle_dataset:
    #     np.random.seed(42)
    #     np.random.shuffle(indices)
    # if mode == "train":
    #     size = cfg["train_sz"]
    # else:
    #     size = cfg["val_sz"]
    # returns a dictionary, the key is the index and the value is the file path
    # TO DO: FIX the problem is that the indices are always cut off at the same point if no shuffle
    # print("Size: ", dataset_size, "Length: ", len(fpaths_sample), len(fpaths_original))
    dataset = {
        str(i): (fpaths_sample[i], fpaths_original[i]) for i in range(dataset_size)
    }

    with open(json_path, "w") as fp:
        json.dump(dataset, fp)

    return dataset


def load_augmentation_index(
    data_dir, splits, json_path=None, ext=["wav", "mp3"], shuffle_dataset=True
):
    dataset = {"train": [], "test": [], "validate": []}
    if json_path is None:
        json_path = os.path.join(data_dir, data_dir.split("/")[-1] + ".json")
    if not os.path.exists(json_path):
        fpaths = glob.glob(os.path.join(data_dir, "**/*.*"), recursive=True)
        fpaths = [p for p in fpaths if p.split(".")[-1] in ext]
        dataset_size = len(fpaths)
        indices = list(range(dataset_size))
        if shuffle_dataset:
            np.random.seed(42)
            np.random.shuffle(indices)
        if type(splits) == list or type(splits) == np.ndarray:
            splits = [int(splits[ix] * dataset_size) for ix in range(len(splits))]
            train_idxs, valid_idxs, test_idxs = (
                indices[: splits[0]],
                indices[splits[0] : splits[0] + splits[1]],
                indices[splits[1] :],
            )
            dataset["validate"] = [fpaths[ix] for ix in valid_idxs]
        else:
            splits = int(splits * dataset_size)
            train_idxs, test_idxs = indices[:splits], indices[splits:]

        dataset["train"] = [fpaths[ix] for ix in train_idxs]
        dataset["test"] = [fpaths[ix] for ix in test_idxs]

        with open(json_path, "w") as fp:
            json.dump(dataset, fp)

    else:
        with open(json_path, "r") as fp:
            dataset = json.load(fp)

    return dataset


def get_frames(y, frame_length, hop_length):
    # frames = librosa.util.frame(y.numpy(), frame_length, hop_length, axis=0)
    frames = y.unfold(0, size=frame_length, step=hop_length)
    return frames


def qtile_normalize(y, q, eps=1e-8):
    return y / (eps + torch.quantile(y.abs(), q=q))


def qtile_norm(y, q, eps=1e-8):
    return eps + torch.quantile(y.abs(), q=q)


def query_len_from_seconds(seconds, overlap, dur):
    hop = dur * (1 - overlap)
    return int((seconds - dur) / hop + 1)


def seconds_from_query_len(query_len, overlap, dur):
    hop = dur * (1 - overlap)
    return int((query_len - 1) * hop + dur)


def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    # # Check if dataparallel is used
    # if 'module' in list(checkpoint['state_dict'].keys())[0]:
    #     print("Loading model with dataparallel...")
    #     checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return (
        model,
        optimizer,
        scheduler,
        checkpoint["epoch"],
        checkpoint["loss"],
        checkpoint["valid_acc"],
    )


def save_ckp(state, model_name, model_folder, text):
    if not os.path.exists(model_folder):
        print("Creating checkpoint directory...")
        os.mkdir(model_folder)
    torch.save(state, "{}/model_{}_{}.pth".format(model_folder, model_name, text))


def load_config(config_path):
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    return config


def override(config_val, arg):
    return arg if arg is not None else config_val


def create_fp_dir(resume=None, ckp=None, epoch=1, train=True):
    if train:
        parent_dir = "logs/emb/valid"
    else:
        parent_dir = "logs/emb/test"

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    if resume is not None:
        ckp_name = resume.split("/")[-1].split(".pt")[0]
    else:
        ckp_name = f"model_{ckp}_epoch_{epoch}"
    fp_dir = os.path.join(parent_dir, ckp_name)
    if not os.path.exists(fp_dir):
        os.mkdir(fp_dir)
    return fp_dir


def create_train_set(data_dir, dest, size=10000):
    if not os.path.exists(dest):
        os.mkdir(dest)
        print(data_dir)
        print(dest)
    for ix, fname in enumerate(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if ix <= size and fpath.endswith("mp3"):
            shutil.move(fpath, dest)
            print(ix)
        if len(os.listdir(dest)) >= size:
            return dest

    return dest


def create_downstream_set(data_dir, size=5000):
    src = os.path.join(data_dir, f"fma_downstream")
    dest = data_dir
    # if not os.path.exists(dest):
    #     os.mkdir(dest)
    # if len(os.listdir(dest)) >= size:
    #     return dest
    for ix, fname in enumerate(os.listdir(src)):
        fpath = os.path.join(src, fname)
        if not fpath.endswith("mp3"):
            continue
        # if ix < size:
        if len(os.listdir(src)) > 500:
            shutil.move(fpath, dest)

    return dest


def preprocess_aug_set_sr(data_dir, sr=22050):
    for fpath in glob.iglob(os.path.join(data_dir, "**/*.wav"), recursive=True):
        y, sr = sf.read(fpath)
        print(sr)
        break
        # sf.write(fpath, data=y, samplerate=sr)
    return


def count_parameters(model, encoder):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    # Write table in text file
    with open(f"model_summary_{encoder}.txt", "w") as f:
        f.write(str(table))
    return total_params


def calculate_output_sparsity(output):
    total_elements = torch.numel(output)
    zero_elements = torch.sum((output == 0).int()).item()

    sparsity = zero_elements / total_elements * 100
    return sparsity

    # Get paths of files not in the index


def get_test_index(data_dir):
    train_idx = load_index(data_dir)
    all_file_list = glob.glob(os.path.join(data_dir, "**/*.mp3"), recursive=True)
    print(f"Number of files in {data_dir}: {len(all_file_list)}")
    # test_idx = {str(i):f for i,f in enumerate(all_file_list) if f not in train_idx.values()}
    idx = 0
    test_idx = {}
    for i, fpath in enumerate(all_file_list):
        if i % 200 == 0:
            print(f"Processed {i}/{len(all_file_list)} files")
        if fpath not in train_idx.values():
            test_idx[str(idx)] = fpath
            idx += 1

    return test_idx
