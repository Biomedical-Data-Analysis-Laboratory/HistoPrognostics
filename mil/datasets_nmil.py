import os

import numpy as np
import pandas as pd
import random


# Parameters
clinicopatholigcal_dict = {
    'Gender': {
        'Male': 0,
        'Female': 1,
        'missing': 2},
    'Smoking': {
        'No': 0,
        'Yes': 1,
        'stopped': 2,
        'missing': 3},
}


class NMILDataset(object):

    def __init__(self, dir_images, dir_embeddings, list_of_wsi, clinical_dataframe,
                 data_augmentation, channel_first,
                 only_clinical, clinical_parameters):
        'Initialize internal state'

        self.dir_images = dir_images
        self.dir_embeddings = dir_embeddings
        self.list_of_wsi = list_of_wsi
        self.data_augmentation = data_augmentation
        self.channel_first = channel_first
        self.only_clinical = only_clinical
        self.clinical_parameters = clinical_parameters
        self.clinical_dataframe = pd.read_excel(clinical_dataframe)

        self.D = dict()
        self.embeddings = []
        self.bag_id = []

        # Organize bags in the form of dictionary: one key clusters indexes from all instances
        self.traceback_images = []
        self.region_id = []
        i = 0

        # Extract embeddings, bag and region belongings
        for SID in self.list_of_wsi:

            if not os.path.exists(os.path.join(self.dir_embeddings, SID) + '.npy'):
                continue

            embeddings = np.load(os.path.join(self.dir_embeddings, SID) + '.npy')
            region_info_dataframe = pd.read_csv(dir_embeddings[:-1] + '_regions/' + SID + '/region_info.csv')
            wsi_image_folder = dir_embeddings + SID

            # Multiple regions per patient, in different WSIs
            if SID in self.D:
                region_addition = max([self.region_id[instance] for instance in self.D[SID]]) + 1
            else:
                region_addition = 0

            # Append embeddings indexes to dict
            for current_wsi_index, traceback_image_filename in zip(range(len(embeddings)), os.listdir(wsi_image_folder)):

                if SID not in self.D:
                    self.D[SID] = [i]
                    self.bag_id.append(SID)
                else:
                    self.D[SID].append(i)

                x_coor, y_coor = int(traceback_image_filename.split('_')[0]), int(traceback_image_filename.split('_')[1].split('.')[0])
                region_value = region_info_dataframe.loc[(region_info_dataframe['X_coor'] == x_coor) &
                                                         (region_info_dataframe['Y_coor'] == y_coor), 'Region'].item()
                self.embeddings.append(embeddings[current_wsi_index])
                self.traceback_images.append(wsi_image_folder + traceback_image_filename)
                self.region_id.append(int(region_value + region_addition))
                i += 1

        # Generate indexes according to the number of images / embeddings
        self.indexes = np.arange(len(self.embeddings))

        # Clinicopathological data
        self.clinical_data = []
        for SID in self.D.keys():

            clinical_data_id = []
            for clinical_parameter in self.clinical_parameters:

                clinical_feature = self.clinical_dataframe.loc[
                    self.clinical_dataframe['SID'] == int(SID), clinical_parameter].item()

                if clinical_parameter == 'Yrs_age':
                    clinical_data_id.append(clinical_feature)
                else:
                    # Check for NaNs and set them to 'missing' instead
                    if isinstance(clinical_feature, float):
                        clinical_data_id.append(len(clinicopatholigcal_dict[clinical_parameter]) - 1)
                    else:
                        clinical_data_id.append(clinicopatholigcal_dict[clinical_parameter][clinical_feature])

            # Fix so instance level classification still works
            for _ in self.D[SID]:
                self.clinical_data.append(clinical_data_id)

        # If clinical only, the number of indexes corresponds to the number of patients
        if self.only_clinical:
            self.indexes = np.arange(len(self.D.keys()))

        # Preemptively load input data into memory
        else:

            # Pre-allocate embeddings
            self.X = np.zeros((len(self.indexes), 1024), dtype=np.float32)

            # Load, and normalize images
            print('[INFO]: Training on ram: Loading images')
            for i in np.arange(len(self.indexes)):
                print(str(i) + '/' + str(len(self.indexes)), end='\r')
                self.X[self.indexes[i], :] = self.images[self.indexes[i]]

        # Load instance labels
        self.y_instances = -1 * np.ones((len(self.indexes),), dtype=np.float32)
        for i in np.arange(len(self.indexes)):
            print(str(i) + '/' + str(len(self.indexes)), end='\r')
            self.y_instances[self.indexes[i]] = (1 if self.clinical_dataframe.loc[self.clinical_dataframe['SID'] == int(
                self.traceback_images[self.indexes[i]].split('/')[-1]), 'BCG_failure'].item() == 'Yes' else 0)
        self.y_instances = self.y_instances.astype(np.float32)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        if self.only_clinical:
            x = None
        else:
            x = np.squeeze(self.embeddings[self.indexes[index], :])

        return x, self.clinical_data[self.indexes[index]], self.region_id[self.indexes[index]]


class NMILDataGenerator(object):

    def __init__(self, dataset, batch_size, shuffle, max_instances):
        'Initialize internal state'

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset.bag_id))
        self.max_instances = max_instances

        self._idx = 0
        self._epoch = 1
        self._reset()

    def __len__(self):
        'Denotes the total number of samples'

        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):

        return self

    def __next__(self):
        'Generates one sample of data'

        # If dataset is completed, stop iterator
        if self._idx >= len(self.indexes):
            self._epoch += 1
            self._reset()
            raise StopIteration()

        # Select instances from bag
        SID = self.dataset.bag_id[self.indexes[self._idx]]
        embeddings_id = self.dataset.D[SID]

        # Get bag-level label
        Y = np.zeros((2,))
        if self.dataset.only_clinical:
            current_y = int(self.dataset.y_instances[self.indexes[self._idx]])
            Y[current_y] = 1
        else:
            bcg_label = (1 if self.dataset.clinical_dataframe.loc[
                                  self.dataset.clinical_dataframe['SID'] == int(SID),
                                  'BCG_failure'].item() == 'Yes' else 0)
            Y[bcg_label] = 1

        # Memory limitation of patches in one slide
        if not self.dataset.only_clinical:
            if len(embeddings_id) > self.max_instances:
                embeddings_id = random.sample(embeddings_id, self.max_instances)
            elif len(embeddings_id) < 4:
                embeddings_id.extend(embeddings_id)

        # Load images and include into the batch
        X = []
        region_info = []
        if self.dataset.only_clinical:
            for i in embeddings_id:
                x, clinical_data_id, region_info_id = self.dataset.__getitem__(i)
            X = None
        else:
            for i in embeddings_id:
                x, clinical_data_id, region_info_id = self.dataset.__getitem__(i)
                X.append(x)
                region_info.append(region_info_id)

        # Update bag index iterator
        self._idx += self.batch_size

        return np.array(X).astype('float32'), np.array(Y).astype('float32'), \
                   np.array(clinical_data_id).astype('float32'), np.array(region_info).astype('int32')

    def _reset(self):
        'Reset sampling generation'

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0
