import os

import torch
import numpy as np
import random
from PIL import Image
import pandas as pd
import math

class Dataset(object):
    def __init__(self, dir_dataset, data_frame, input_shape=(3, 512, 512),
                 augmentation=True, preallocate=True, inference=False, wsi_list=[], config=0):

        self.dir_dataset = dir_dataset
        self.data_frame = data_frame
        self.config = config
        self.augmentation = augmentation
        self.preallocate = preallocate
        self.inference = inference
        self.wsi_list = wsi_list
        self.input_shape = input_shape
        self.images = []
        self.Y = []

        # Recursively find all image files within subdirectories
        if self.inference:

            if not os.path.exists(os.path.join(dir_dataset, [f for f in os.listdir(dir_dataset) if f.endswith('.csv')][0])):
                print('')
            else:
                tile_filename_list = os.listdir(dir_dataset)
                for name in tile_filename_list:
                    self.images.append(os.path.join(dir_dataset, name))
                    self.Y.append(-1)

        else: # Training
            wsi_count = 0
            for path in os.listdir(dir_dataset):
                print('{}/{}: {}'.format(wsi_count, len(self.wsi_list), path))
                if len([f for f in os.listdir(dir_dataset+ path) if f.endswith('.csv')]) == 0:
                    print('Empty file: {}'.format(path))
                    continue
                wsi_count += 1

                # Directories for multiscale dependence
                current_tile_folder = os.path.join(dir_dataset, path)
                current_tile_info_dataframe = pd.read_csv(os.path.join(dir_dataset + path,
                                [f for f in os.listdir(dir_dataset+ path) if f.endswith('.csv')][0]))
                current_bcg_failure_label = (1 if self.data_frame.loc[self.data_frame['SID'] == int(path),
                              'BCG_failure'].item() == 'Yes' else 0)

                # Iterate over tiles in the repository
                tile_filename_list = os.listdir(current_tile_folder)
                for name in tile_filename_list:

                    if self.config in [1, 2, 3]:

                        current_grade_label = self.tile_info_dataframe.loc[
                            (self.tile_info_dataframe['X_coor'] == int(name.split('_')[0])) & \
                            (self.tile_info_dataframe['Y_coor'] == int(name.split('_')[1])), 'Grade'].item()
                        current_til_label = self.tile_info_dataframe.loc[
                            (self.tile_info_dataframe['X_coor'] == int(name.split('_')[0])) & \
                            (self.tile_info_dataframe['Y_coor'] == int(name.split('_')[1])), 'TILs'].item()

                        # If there are not either Grade or TIL labels, then it is an unlabeled sample
                        current_grade_is_nan = math.isnan(current_grade_label)
                        current_til_is_nan = math.isnan(current_til_label)

                        if not current_grade_is_nan:
                            self.Y.append(int(current_grade_label))
                        elif not current_til_is_nan:
                            self.Y.append(int(current_til_label))
                        else:
                            self.Y.append(-1)

                    elif config == 4:
                        self.Y.append(-1)

                    elif self.config == 5: # BCG weak
                        self.Y.append(current_bcg_failure_label)

                    # Append filenames
                    self.images.append(os.path.join(current_tile_folder, name))

        self.indexes = np.arange(len(self.images))
        self.Y = np.array(self.Y)

        if self.preallocate:
            # Load, and normalize images
            self.X = np.zeros((len(self.images), self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)

            print('[INFO]: Training on ram: Loading images')
            for i in np.arange(len(self.indexes)):
                print(str(i) + '/' + str(len(self.indexes)), end='\r')

                x = Image.open(self.images[self.indexes[i]])
                x = np.asarray(x)
                x = np.transpose(x, (2, 0, 1))

                # Normalization
                self.X[self.indexes[i], :, :, :] = x / 255

            print('[INFO]: Images loaded')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images_400x)

    def __getitem__(self, index):
        'Generates one sample of data'

        if self.preallocate:
            x = self.X[index, :, :, :]
        else:

            x = Image.open(self.images[self.indexes[index]])
            x = np.asarray(x)
            x = np.transpose(x, (2, 0, 1))

            # Normalization
            x = x / 255

        y = self.Y[index]

        x = torch.tensor(x).float().cuda()
        y = torch.tensor(y).float().cuda()

        return x, y


class Generator(object):
    def __init__(self, dataset, batch_size, shuffle=True, augmentation=False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.indexes = np.arange(0, len(self.dataset.images))
        self._idx = 0

        self._reset()

    def __len__(self):
        N = len(self.dataset.images)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def __next__(self):

        if self._idx == len(self.dataset.images):
            self._reset()
            raise StopIteration()

        if self._idx + self.batch_size > len(self.dataset.images):
            iteration_batch_size = len(self.dataset.images) - self._idx
        else:
            iteration_batch_size = self.batch_size

        # Load images and include into the batch
        X, Y = [], []

        # Try preallocating numpy array
        for i in range(self._idx, self._idx + iteration_batch_size):
            x, y = self.dataset.__getitem__(self.indexes[i])

            X.append(x.unsqueeze(0))
            Y.append(y.unsqueeze(0))

        # Update index iterator
        self._idx += iteration_batch_size

        X = torch.cat(X, 0)
        Y = torch.cat(Y, 0)

        if self.augmentation:
            X = self.dataset.transforms(X)

        return X, Y

    def _reset(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0

