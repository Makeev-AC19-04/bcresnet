# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from argparse import ArgumentParser
import shutil
from glob import glob
import time
import psutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from bcresnet import BCResNets
from constants import TOTAL_EPOCH, KEYWORD, TRAIN_RATIO, VAL_RATIO, WARMUP_EPOCH, BATCH_SIZE
from utils import Padding, Preprocess, SpeechCommand, SplitDataset

DATASET_PATH = f"./data/{KEYWORD}"

class Trainer:
    def __init__(self):
        """
        Constructor for the Trainer class.

        Initializes the trainer object with default values for the hyperparameters and data loaders.
        """
        parser = ArgumentParser()
        parser.add_argument(
            "--tau", default=1, help="model size", type=float, choices=[1, 1.5, 2, 3, 6, 8]
        )
        parser.add_argument("--gpu", default=0, help="gpu device id", type=int)
        parser.add_argument("--split_dataset", help="split dataset to train, val and test", action="store_true") # Не понял почему тут работает только с store_true, c split_true выдает ошибку
        args = parser.parse_args()
        self.__dict__.update(vars(args))
        self.device = torch.device("cuda:%d" % self.gpu if torch.cuda.is_available() else "cpu")
        self._load_data()
        self._load_model()

    def __call__(self):
        """
        Method that allows the object to be called like a function.

        Trains the model and presents the train/test progress.
        """
        # train hyperparameters
        total_epoch = TOTAL_EPOCH
        warmup_epoch = WARMUP_EPOCH
        init_lr = 1e-1
        lr_lower_limit = 0

        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0, weight_decay=1e-3, momentum=0.9)
        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0

        print(len(self.train_loader))
        # train
        for epoch in range(total_epoch):
            self.model.train()
            for sample in tqdm(self.train_loader, desc="epoch %d, iters" % (epoch + 1)):
                # lr cos schedule
                iterations += 1
                if iterations < n_step_warmup:
                    lr = init_lr * iterations / n_step_warmup
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                inputs, labels = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = self.preprocess_train(inputs, labels, augment=True)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

            # valid
            print("cur lr check ... %.4f" % lr)
            with torch.no_grad():
                self.model.eval()

                valid_acc = self.Test(self.valid_dataset, self.valid_loader, augment=True)
                print("valid acc: %.3f" % (valid_acc))

        print(f"Test dataset size: {len(self.test_dataset)}")
        memory_before = psutil.virtual_memory().used
        start_time = time.time()

        test_acc = self.Test(self.test_dataset, self.test_loader, augment=False)  # official testset

        memory_after = psutil.virtual_memory().used
        end_time = time.time()
        
        torch.save(self.model.state_dict(), f'models/{KEYWORD}_model.pth')
        print(f"model saved at models/{KEYWORD}_model.pth")
        print("test acc: %.3f" % (test_acc))

        execution_time = end_time - start_time
        memory_used = (memory_after - memory_before)/1024
        print(f"Execution time: {execution_time:.4f} seconds")
        print(f"Memory used: {memory_used:.4f} Kb")

        print(f"Average execution time to predict 1 file: {execution_time/len(self.test_dataset):.4f} seconds")
        print(f"Average memory used to predict 1 file: {memory_used/len(self.test_dataset):.4f} Kb")

        print("End.")

    def Test(self, dataset, loader, augment):
        """
        Tests the model on a given dataset.

        Parameters:
            dataset (Dataset): The dataset to test the model on.
            loader (DataLoader): The data loader to use for batching the data.
            augment (bool): Flag indicating whether to use data augmentation during testing.

        Returns:
            float: The accuracy of the model on the given dataset.
        """
        true_count = 0.0
        num_testdata = float(len(dataset))
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs = self.preprocess_test(inputs, labels=labels, is_train=False, augment=augment)
            outputs = self.model(inputs)
            prediction = torch.argmax(outputs, dim=-1)
            true_count += torch.sum(prediction == labels).detach().cpu().numpy()
        acc = true_count / num_testdata * 100.0  # percentage
        return acc

    def _load_data(self):
        """
        Private method that loads data into the object.

        Downloads and splits the data if necessary.
        """
        if not os.path.isdir("./data"):
            os.mkdir("./data")
        base_dir = DATASET_PATH

        print(str(self.split_dataset))

        if self.split_dataset:
            SplitDataset(base_dir)

        # Define data loaders
        train_dir = "%s/train" % base_dir
        valid_dir = "%s/val" % base_dir
        test_dir = "%s/test" % base_dir
        noise_dir = "%s/_background_noise_" % base_dir

        transform = transforms.Compose([Padding()])
        # Read audiofiles and define their's labels
        self.train_dataset = SpeechCommand(train_dir, transform=transform) 
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False
        )
        self.valid_dataset = SpeechCommand(valid_dir, transform=transform)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=BATCH_SIZE, num_workers=0)
        self.test_dataset = SpeechCommand(test_dir, transform=transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=BATCH_SIZE, num_workers=0)

        print(
            "check num of data train/valid/test %d/%d/%d"
            % (len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))
        )

        specaugment = self.tau >= 1.5
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        # Define preprocessors
        self.preprocess_train = Preprocess(
            noise_dir,
            self.device,
            specaug=specaugment,
            frequency_masking_para=frequency_masking_para[self.tau],
        )
        self.preprocess_test = Preprocess(noise_dir, self.device)

    def _load_model(self):
        """
        Private method that loads the model into the object.
        """
        print("model: BC-ResNet-%.1f" % self.tau)
        self.model = BCResNets(int(self.tau * 8)).to(self.device)


if __name__ == "__main__":
    _trainer = Trainer()
    _trainer()
