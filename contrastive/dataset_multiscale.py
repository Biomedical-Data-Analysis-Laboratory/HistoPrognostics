import os

import torch
import numpy as np
import random
from PIL import Image
import pandas as pd
import math

class Dataset(object):
    def __init__(self, dir_dataset, data_frame,
                 augmentation=True, preallocate=True, inference=False, wsi_list=[], config=0):

        self.dir_dataset = dir_dataset
        self.data_frame = data_frame
        self.config = config
        self.augmentation = augmentation
        self.preallocate = preallocate
        self.inference = inference
        self.wsi_list = wsi_list
        self.images = []
        self.images_400x = []
        self.images_100x = []
        self.images_25x = []
        self.Y = []

        # Recursively find all image files within subdirectories
        if self.inference:

            current_tile_folder_400x = os.path.join(dir_dataset, '400x')
            current_tile_folder_100x = os.path.join(dir_dataset, '100x')
            current_tile_folder_25x = os.path.join(dir_dataset, '25x')
            if not os.path.exists(os.path.join(dir_dataset, [f for f in os.listdir(dir_dataset) if f.endswith('.csv')][0])):
                print('')
            else:
                current_tile_info_dataframe = pd.read_csv(os.path.join(dir_dataset, [f for f in os.listdir(dir_dataset) if f.endswith('.csv')][0]))

                tile_filename_list_400x = os.listdir(current_tile_folder_400x)
                tile_dict_400x = dict.fromkeys(tile_filename_list_400x, 0)

                for name_400x in tile_dict_400x.keys():

                    current_dataframe_row = current_tile_info_dataframe.loc[(current_tile_info_dataframe['400X_coor'] == int(name_400x.split('_')[0])) &
                                                                (current_tile_info_dataframe['400Y_coor'] == int(name_400x.split('_')[1].split('.')[0]))]

                    if len(current_dataframe_row) == 1:
                        coor_x_100x = current_dataframe_row['100X_coor'].item()
                        coor_y_100x = current_dataframe_row['100Y_coor'].item()
                    else:
                        coor_x_100x = current_dataframe_row['100X_coor'].iloc[0]
                        coor_y_100x = current_dataframe_row['100Y_coor'].iloc[0]

                    name_100x = current_tile_folder_100x + '/' + str(int(coor_x_100x)) + '_' + str(int(coor_y_100x)) + '.jpeg'
                    if not os.path.isfile(name_100x):
                        print(name_100x)
                        exit()

                    if len(current_dataframe_row) == 1:
                        coor_x_25x = current_dataframe_row['25X_coor'].item()
                        coor_y_25x = current_dataframe_row['25Y_coor'].item()
                    else:
                        coor_x_25x = current_dataframe_row['25X_coor'].iloc[0]
                        coor_y_25x = current_dataframe_row['25Y_coor'].iloc[0]

                    name_25x = current_tile_folder_25x + '/' + str(int(coor_x_25x)) + '_' + str(int(coor_y_25x))+ '.jpeg'
                    if not os.path.isfile(name_25x):
                        print(name_25x)
                        exit()

                    self.images_400x.append(os.path.join(current_tile_folder_400x, name_400x))
                    self.images_100x.append(os.path.join(current_tile_folder_100x, name_100x))
                    self.images_25x.append(os.path.join(current_tile_folder_25x, name_25x))
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
                current_tile_folder_400x = os.path.join(dir_dataset + path, '400x')
                current_tile_folder_100x = os.path.join(dir_dataset + path, '100x')
                current_tile_folder_25x = os.path.join(dir_dataset + path, '25x')
                current_tile_info_dataframe = pd.read_csv(os.path.join(dir_dataset + path,
                                [f for f in os.listdir(dir_dataset+ path) if f.endswith('.csv')][0]))
                current_bcg_failure_label = (1 if self.data_frame.loc[self.data_frame['SID'] == int(path),
                              'BCG_failure'].item() == 'Yes' else 0)

                tile_filename_list_400x = os.listdir(current_tile_folder_400x)
                tile_dict_400x = dict.fromkeys(tile_filename_list_400x, 0)

                # Iterate over tiles in the repository
                for name_400x in tile_dict_400x.keys():

                    current_dataframe_row = current_tile_info_dataframe.loc[
                        (current_tile_info_dataframe['400X_coor'] == int(name_400x.split('_')[0])) &
                        (current_tile_info_dataframe['400Y_coor'] == int(name_400x.split('_')[1]))]
                    coor_x_100x = current_dataframe_row['100X_coor'].item()
                    coor_y_100x = current_dataframe_row['100Y_coor'].item()

                    name_100x = current_tile_folder_100x + '/' + str(int(coor_x_100x)) + '_' + str(
                        int(coor_y_100x)) + '.jpeg'
                    if not os.path.isfile(name_100x):
                        print(name_100x)
                        exit()

                    coor_x_25x = current_dataframe_row['25X_coor'].item()
                    coor_y_25x = current_dataframe_row['25Y_coor'].item()

                    name_25x = current_tile_folder_25x + '/' + str(int(coor_x_25x)) + '_' + str(
                        int(coor_y_25x)) + '.jpeg'
                    if not os.path.isfile(name_25x):
                        print(name_25x)
                        exit()

                    if self.config in [1, 2, 3]:

                        current_grade_label = self.tile_info_dataframe.loc[
                            (self.tile_info_dataframe['400X_coor'] == int(name_400x.split('_')[0])) & \
                            (self.tile_info_dataframe['400Y_coor'] == int(name_400x.split('_')[1])), 'Grade'].item()
                        current_til_label = self.tile_info_dataframe.loc[
                            (self.tile_info_dataframe['400X_coor'] == int(name_400x.split('_')[0])) & \
                            (self.tile_info_dataframe['400Y_coor'] == int(name_400x.split('_')[1])), 'TILs'].item()

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
                    self.images_400x.append(os.path.join(current_tile_folder_400x, name_400x))
                    self.images_100x.append(os.path.join(current_tile_folder_100x, name_100x))
                    self.images_25x.append(os.path.join(current_tile_folder_25x, name_25x))

        self.indexes = np.arange(len(self.images_400x))
        self.Y = np.array(self.Y)

        if self.preallocate:
            # Load, and normalize images
            print('[INFO]: Training on ram: Loading images')
            for i in np.arange(len(self.indexes)):
                print(str(i) + '/' + str(len(self.indexes)), end='\r')

                x400 = Image.open(self.images_400x[self.indexes[i]])
                x400 = np.asarray(x400)
                x400 = np.transpose(x400, (2, 0, 1))

                x100 = Image.open(self.images_100x[self.indexes[i]])
                x100 = np.asarray(x100)
                x100 = np.transpose(x100, (2, 0, 1))

                x25 = Image.open(self.images_25x[self.indexes[i]])
                x25 = np.asarray(x25)
                x25 = np.transpose(x25, (2, 0, 1))

                # Normalization
                self.x400[self.indexes[i], :, :, :] = x400 / 255
                self.x100[self.indexes[i], :, :, :] = x100 / 255
                self.x25[self.indexes[i], :, :, :] = x25 / 255
            print('[INFO]: Images loaded')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images_400x)

    def __getitem__(self, index):
        'Generates one sample of data'

        if self.preallocate:
            x400 = self.X400[index, :, :, :]
            x100 = self.X100[index, :, :, :]
            x25 = self.X25[index, :, :, :]
        else:

            x400 = Image.open(self.images_400x[self.indexes[index]])
            x400 = np.asarray(x400)
            x400 = np.transpose(x400, (2, 0, 1))

            x100 = Image.open(self.images_100x[self.indexes[index]])
            x100 = np.asarray(x100)
            x100 = np.transpose(x100, (2, 0, 1))

            x25 = Image.open(self.images_25x[self.indexes[index]])
            x25 = np.asarray(x25)
            x25 = np.transpose(x25, (2, 0, 1))

            # Normalization
            x400 = x400 / 255
            x100 = x100 / 255
            x25 = x25 / 255

        y = self.Y[index]

        x400 = torch.tensor(x400).float().cuda()
        x100 = torch.tensor(x100).float().cuda()
        x25 = torch.tensor(x25).float().cuda()
        y = torch.tensor(y).float().cuda()

        return x400, x100, x25, y


class Generator(object):
    def __init__(self, dataset, batch_size, shuffle=True, augmentation=False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.indexes = np.arange(0, len(self.dataset.images_400x))
        self._idx = 0

        self._reset()

    def __len__(self):
        N = len(self.dataset.images_400x)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def __next__(self):

        if self._idx == len(self.dataset.images_400x):
            self._reset()
            raise StopIteration()

        if self._idx + self.batch_size > len(self.dataset.images_400x):
            iteration_batch_size = len(self.dataset.images_400x) - self._idx
        else:
            iteration_batch_size = self.batch_size

        # Load images and include into the batch
        X400, X100, X25, Y = [], [], [], []

        # Try preallocating numpy array
        for i in range(self._idx, self._idx + iteration_batch_size):
            x400, x100, x25, y = self.dataset.__getitem__(self.indexes[i])

            X400.append(x400.unsqueeze(0))
            X100.append(x100.unsqueeze(0))
            X25.append(x25.unsqueeze(0))
            Y.append(y.unsqueeze(0))

        # Update index iterator
        self._idx += iteration_batch_size

        X400 = torch.cat(X400, 0)
        X100 = torch.cat(X100, 0)
        X25 = torch.cat(X25, 0)
        Y = torch.cat(Y, 0)

        if self.augmentation:
            X400 = self.dataset.transforms(X400)
            X100 = self.dataset.transforms(X100)
            X25 = self.dataset.transforms(X25)

        return X400, X100, X25, Y

    def _reset(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0

