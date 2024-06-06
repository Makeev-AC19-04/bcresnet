# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import random
from glob import glob
import shutil
import requests
import tarfile
import noisereduce as nr

from constants import KEYWORD, TRAIN_RATIO, VAL_RATIO, SAMPLE_RATE

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

### GSC
label_dict = {
    KEYWORD: 1,
    "filler": 0,
}
print("labels:\t", label_dict)


def ScanAudioFiles(root_dir):
    audio_paths, labels = [], []
    for path, _, files in sorted(os.walk(root_dir, followlinks=True)):
        random.shuffle(files)
        for idx, filename in enumerate(files):
            if not filename.endswith(".wav"):
                continue
            dataset, class_name = path.split("/")[-2:]
            audio_paths.append(os.path.join(path, filename))
            labels.append(label_dict[class_name])
    return audio_paths, labels


class SpeechCommand(Dataset):
    """GSC"""

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data_list, self.labels = ScanAudioFiles(root_dir) # Получаем Список файлов и их классы

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = self.data_list[idx]
        sample, _ = torchaudio.load(audio_path)
        sample = torch.tensor(nr.reduce_noise(y=sample, sr=SAMPLE_RATE, stationary=True))
        if self.transform:
            sample = self.transform(sample)
        label = self.labels[idx]
        return sample, label


def spec_augment(
    x, frequency_masking_para=20, time_masking_para=20, frequency_mask_num=0, time_mask_num=0
):
    lenF, lenT = x.shape[1:3]
    # Frequency masking
    for _ in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, int((lenF - f)/10)) #lenF - f
        x[:, f0 : f0 + f, :] = 0
    # Time masking
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, int((lenT - t)/10)) #lenF - f
        x[:, :, t0 : t0 + t] = 0
    return x


class Preprocess:
    def __init__(
        self,
        noise_loc,
        device,
        hop_length=160,
        win_length=480,
        n_fft=512,
        n_mels=40,
        specaug=False,
        sample_rate=SAMPLE_RATE,
        frequency_masking_para=7,
        time_masking_para=20,
        frequency_mask_num=2,
        time_mask_num=2,
    ):
        if noise_loc is None:
            self.background_noise = []
        else:
            self.background_noise = [
                torchaudio.load(file_name)[0] for file_name in glob(noise_loc + "/*.wav")
            ]
            assert len(self.background_noise) != 0
        self.feature = LogMel(
            device,
            sample_rate=sample_rate,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )
        self.sample_len = sample_rate
        self.specaug = specaug
        self.device = device
        if self.specaug:
            self.frequency_masking_para = frequency_masking_para
            self.time_masking_para = time_masking_para
            self.frequency_mask_num = frequency_mask_num
            self.time_mask_num = time_mask_num
            print(
                "frequency specaug %d %d" % (self.frequency_mask_num, self.frequency_masking_para)
            )
            print("time specaug %d %d" % (self.time_mask_num, self.time_masking_para))

    def __call__(self, x, labels, augment=True, noise_prob=0.8, is_train=True):
        assert len(x.shape) == 3
        if augment:
            for idx in range(x.shape[0]):
                if labels[idx] != 0 and (not is_train or random.random() > noise_prob):
                    continue
                noise_amp = (
                    np.random.uniform(0, 0.1) if labels[idx] != 0 else np.random.uniform(0, 1)
                )
                noise = random.choice(self.background_noise).to(self.device)
                sample_loc = random.randint(0, noise.shape[-1] - self.sample_len)
                noise = noise_amp * noise[:, sample_loc : sample_loc + SAMPLE_RATE]

                if is_train:
                    x_shift = int(np.random.uniform(-0.1, 0.1) * SAMPLE_RATE)
                    zero_padding = torch.zeros(1, np.abs(x_shift)).to(self.device)
                    if x_shift < 0:
                        temp_x = torch.cat([zero_padding, x[idx, :, :x_shift]], dim=-1)
                    else:
                        temp_x = torch.cat([x[idx, :, x_shift:], zero_padding], dim=-1)
                    x[idx] = temp_x + noise
                else:  # valid
                    x[idx] = x[idx] + noise
                x[idx] = torch.clamp(x[idx], -1.0, 1.0)

        x = self.feature(x)
        if self.specaug:
            for i in range(x.shape[0]):
                x[i] = spec_augment(
                    x[i],
                    self.frequency_masking_para,
                    self.time_masking_para,
                    self.frequency_mask_num,
                    self.time_mask_num,
                )
        return x


class LogMel:
    def __init__(
        self, device, sample_rate=SAMPLE_RATE, hop_length=160, win_length=480, n_fft=512, n_mels=40
    ):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
        )
        self.device = device

    def __call__(self, x):
        self.mel = self.mel.to(self.device)
        output = (self.mel(x) + 1e-6).log()
        return output


class Padding:
    """zero pad to have 1 sec len"""

    def __init__(self):
        self.output_len = SAMPLE_RATE

    def __call__(self, x):
        pad_len = self.output_len - x.shape[-1]
        if pad_len > 0:
            x = torch.cat([x, torch.zeros([x.shape[0], pad_len])], dim=-1)
        elif pad_len < 0:
            raise ValueError("no sample exceed 1sec in GSC.")
        return x


def SplitDataset(path):
    TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO

    for folder in os.listdir(path):
        if folder != "_background_noise_":
            train_path = f"{path}/train/{folder}"
            val_path = f"{path}/val/{folder}"
            test_path = f"{path}/test/{folder}"

            os.makedirs(train_path, exist_ok=True)
            os.makedirs(val_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            files = os.listdir(os.path.join(path, folder))
            
            # Перемешиваем аудиофайлы
            random.shuffle(files)
            
            # Разделяем аудиофайлы на тренировочную, валидационную и тестовую выборки
            train_size = int(len(files) * TRAIN_RATIO)
            val_size = int(len(files) * VAL_RATIO)
            test_size = len(files) - train_size - val_size
            
            train_files = files[:train_size]
            val_files = files[train_size:train_size + val_size]
            test_files = files[train_size + val_size:]
            
            # Копируем аудиофайлы в соответствующие папки
            for file in train_files:
                shutil.move(os.path.join(path, folder, file), os.path.join(train_path, file))
            for file in val_files:
                shutil.move(os.path.join(path, folder, file), os.path.join(val_path, file))
            for file in test_files:
                shutil.move(os.path.join(path, folder, file), os.path.join(test_path, file))

            # Удаляем исходные папки
            os.rmdir(os.path.join(path, folder))

