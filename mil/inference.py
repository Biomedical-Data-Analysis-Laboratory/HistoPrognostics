import os

import pyvips
import torch
import numpy as np
from timeit import default_timer as timer

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


import skimage
from skimage.filters import rank
from skimage.morphology import disk
from skimage.transform import resize

from PIL import Image
from scipy import ndimage
from datasets_nmil import *
import pandas as pd


class MILInference():
    def __init__(self, dir_wsi, dir_out, backbone, network, dir_inference, dir_images_regions, id='',
                 virtual_batch_size=1, input_shape=(3, 224, 224), ori_patch_size=512):

        self.dir_wsi = dir_wsi
        self.dir_results = dir_out
        self.dir_inference = dir_inference
        self.dir_images_regions = dir_images_regions
        if not os.path.exists(self.dir_results + self.dir_inference):
            os.mkdir(self.dir_results + self.dir_inference)

        # Other
        self.init_time = 0
        self.backbone = backbone
        self.network = network
        self.metrics = {}
        self.id = id
        self.virtual_batch_size = virtual_batch_size
        self.input_shape = input_shape
        self.ori_patch_size = ori_patch_size
        self.current_wsi_subfolder = ''
        self.current_wsi_name = ''
        self.region_info_dataframe = ''

    def infer(self, current_wsi_subfolder):

        self.current_wsi_subfolder = current_wsi_subfolder
        self.current_wsi_name = self.current_wsi_subfolder.split('/')[-1]
        self.region_info_dataframe = pd.read_csv(self.dir_images_regions + self.current_wsi_name + '/region_info.csv')

        # Move network to gpu
        self.backbone.cuda()
        self.backbone.eval()
        self.network.cuda()
        self.network.eval()


        self.init_time = timer()

        if not os.path.exists(self.dir_results + self.dir_inference + self.current_wsi_name):
            os.mkdir(self.dir_results + self.dir_inference + self.current_wsi_name)

            wsi_thumbnail = pyvips.Image.new_from_file(
                self.dir_wsi + self.current_wsi_name + ".mrxs", autocrop=True, level=4).flatten()

            if not os.path.exists(
                    self.dir_results + self.dir_inference + self.current_wsi_name + '/patch_classification.npy'):

                # Loop over training dataset
                print('[Inference]: {}'.format(self.current_wsi_subfolder.split('/')[-1]))

                wsi_thumbnail = pyvips.Image.new_from_file(
                    self.dir_wsi + self.current_wsi_name + ".mrxs", autocrop=True, level=4).flatten()
                wsi_thumb_save = pyvips.Image.new_from_file(
                    self.dir_wsi + self.current_wsi_name + ".mrxs", autocrop=True, level=5).flatten()
                wsi_thumb_save.jpegsave(
                    self.dir_results + self.dir_inference + self.current_wsi_name + '/thumbnail.jpeg', Q=100)

                features = []
                region_info = []
                for iInstance, current_patch_filename in enumerate(os.listdir(self.current_wsi_subfolder)):
                    print(str(iInstance + 1) + '/' + str(len(os.listdir(self.current_wsi_subfolder))), end='\r')

                    # Load patch
                    x = Image.open(os.path.join(self.current_wsi_subfolder, current_patch_filename))
                    x = np.asarray(x)
                    # Normalization
                    x = self.image_normalization(x)
                    x = x[np.newaxis, ...]
                    x = torch.tensor(x).cuda().float()

                    # Transform image into low-dimensional embedding
                    features.append(torch.squeeze(self.backbone(x)).detach().cpu().numpy())

                    x_coor, y_coor = int(current_patch_filename.split('_')[0]), int(current_patch_filename.split('_')[1].split('.')[0])
                    region_value = self.region_info_dataframe.loc[(self.region_info_dataframe['X_coor'] == x_coor) &
                                                                  (self.region_info_dataframe['Y_coor'] == y_coor), 'Region'].item()
                    region_info.append(region_value)
                print('CNN Features: done')

                # Compute attention and bag prediction
                features = torch.tensor(np.array(features)).cuda().float()
                region_info = torch.tensor(np.array(region_info)).cuda().float()
                instance_classification = []
                region_embeddings_list = []
                for region_id in range(int(torch.max(region_info).item()) + 1):
                    region_features = features[torch.where(region_info == region_id, True, False)]

                    A_V = self.network.milAggregation.attentionModule.attention_V(region_features)  # Attention
                    A_U = self.network.milAggregation.attentionModule.attention_U(region_features)  # Gate
                    w_logits = self.network.milAggregation.attentionModule.attention_weights(
                        A_V * A_U)  # Probabilities - softmax over instances
                    instance_classification.append(w_logits)

                    if not len(region_features) == 1:
                        embedding_reg, _ = self.network.milAggregation(torch.squeeze(region_features))
                        region_embeddings_list.append(embedding_reg)
                    else:
                        region_embeddings_list.append(torch.squeeze(region_features, dim=0))

                if len(region_embeddings_list) == 1:
                    patch_classification = w_logits
                    embedding = embedding_reg
                    A_V = self.network.milAggregation.attentionModule.attention_V(embedding_reg)  # Attention
                    A_U = self.network.milAggregation.attentionModule.attention_U(embedding_reg)  # Gate
                    w_logits = self.network.milAggregation.attentionModule.attention_weights(
                        A_V * A_U)  # Probabilities - softmax over instances
                else:
                    embedding, _ = self.network.milAggregation(torch.squeeze(torch.stack(region_embeddings_list)))
                    A_V = self.network.milAggregation.attentionModule.attention_V(
                        torch.squeeze(torch.stack(region_embeddings_list)))  # Attention
                    A_U = self.network.milAggregation.attentionModule.attention_U(
                        torch.squeeze(torch.stack(region_embeddings_list)))  # Gate
                    w_logits = self.network.milAggregation.attentionModule.attention_weights(
                        A_V * A_U)  # Probabilities - softmax over instances
                    patch_classification = torch.cat(instance_classification, dim=0)

                # embedding, w = self.network.milAggregation.attentionModule(torch.squeeze(features))
                global_classification = self.network.classifier(embedding).detach().cpu().numpy()

                patch_classification = patch_classification.detach().cpu().numpy()

                region_classification_aux = w_logits.detach().cpu().numpy()
                region_classification = np.zeros(patch_classification.shape)

                region_info = region_info.detach().cpu().numpy()

                for i in range(len(region_info)):
                    region_classification[i] = region_classification_aux[int(region_info[i])]

                print('MIL classification: done')
                np.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/patch_classification.npy',
                        patch_classification)
                np.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/region_classification.npy',
                        region_classification)
                np.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/global_classification.npy',
                        global_classification)
            else:
                patch_classification = np.load(
                    self.dir_results + self.dir_inference + self.current_wsi_name + '/patch_classification.npy')
                region_classification = np.load(
                    self.dir_results + self.dir_inference + self.current_wsi_name + '/region_classification.npy')
                global_classification = np.load(
                    self.dir_results + self.dir_inference + self.current_wsi_name + '/global_classification.npy')

            # Normalize attention
            min_max_att = np.load(self.dir_results + self.dir_inference + '/min_max_att.npy')
            patch_classification = (patch_classification - min_max_att[0]) / (min_max_att[1] - min_max_att[0])
            region_classification = (region_classification - min_max_att[0]) / (min_max_att[1] - min_max_att[0])

            # Load thumbnail array for predictions
            wsi_20x = pyvips.Image.new_from_file(
                self.dir_wsi + self.current_wsi_name + ".mrxs", autocrop=True, level=2).flatten()
            colormap_height = wsi_20x.height // self.ori_patch_size
            colormap_width = wsi_20x.width // self.ori_patch_size

            if not os.path.exists(
                    self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction_colormap.npy') or not os.path.exists(
                    self.dir_results + self.dir_inference + self.current_wsi_name + '/roi_patch_extraction.jpeg'):
                prediction_colormap = -1 * np.ones((colormap_height, colormap_width), dtype=np.float32)
                prediction_colormap_reg = -1 * np.ones((colormap_height, colormap_width), dtype=np.float32)
                roi_mask = np.ones((colormap_height, colormap_width), dtype=np.float32)
                roi_mask_reg = np.ones((colormap_height, colormap_width), dtype=np.float32)

                # Generate WSI map prediction with attention
                for iInstance, (patch_attention_score, reg_attention_score, current_patch_filename) in enumerate(
                        zip(patch_classification, region_classification, os.listdir(self.current_wsi_subfolder))):
                    print(str(iInstance + 1) + '/' + str(patch_classification.shape[0]), end='\r')
                    x_coordinate = int(current_patch_filename.split('_')[-1].split('.')[0])
                    y_coordinate = int(current_patch_filename.split('_')[-2])

                    x_index = x_coordinate // self.ori_patch_size
                    y_index = y_coordinate // self.ori_patch_size

                    # To be overlayed using alpha channel
                    prediction_colormap[x_index, y_index] = patch_attention_score
                    prediction_colormap_reg[x_index, y_index] = reg_attention_score

                roi_mask[prediction_colormap == -1] = 0
                roi_mask = resize(roi_mask, (wsi_thumbnail.height, wsi_thumbnail.width))
                roi_image = Image.fromarray(np.uint8(roi_mask * 255))
                roi_image.save(
                    self.dir_results + self.dir_inference + self.current_wsi_name + '/roi_patch_extraction.jpeg')

                roi_mask_reg[prediction_colormap_reg == -1] = 0
                roi_mask_reg = resize(roi_mask_reg, (wsi_thumbnail.height, wsi_thumbnail.width))
                roi_image_reg = Image.fromarray(np.uint8(roi_mask_reg * 255))
                roi_image_reg.save(
                    self.dir_results + self.dir_inference + self.current_wsi_name + '/roi_reg_extraction.jpeg')

                prediction_colormap[prediction_colormap == -1] = 0.
                prediction_colormap_reg[prediction_colormap_reg == -1] = 0.

                # Save generated colormap
                np.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction_colormap.npy',
                        prediction_colormap)
                np.save(
                    self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction_colormap_region.npy',
                    prediction_colormap_reg)
                print('Prediction colormap: done')
            else:
                prediction_colormap = np.load(
                    self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction_colormap.npy')
                prediction_colormap_reg = np.load(
                    self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction_colormap_region.npy')

            # Save thumbnail image and the overlayed colormap
            wsi_alpha = np.ndarray(buffer=wsi_thumbnail.write_to_memory(),
                                   dtype=np.uint8,
                                   shape=[wsi_thumbnail.height, wsi_thumbnail.width, wsi_thumbnail.bands])

            prediction_colormap = resize(prediction_colormap, (wsi_thumbnail.height, wsi_thumbnail.width))
            prediction_colormap = prediction_colormap ** 2
            prediction_colormap = rank.mean(prediction_colormap, disk(21))
            prediction_colormap_reg = resize(prediction_colormap_reg, (wsi_thumbnail.height, wsi_thumbnail.width))
            prediction_colormap_reg = prediction_colormap_reg ** 2
            prediction_colormap_reg = rank.mean(prediction_colormap_reg, disk(21))

            alpha = 0.5;
            wsi_blended = ((plt.cm.jet(prediction_colormap)[:, :, :3] * 255) * alpha + wsi_alpha * (1 - alpha)).astype(
                np.uint8)  # [:,:,:3]
            wsi_blended_reg = (
                        (plt.cm.jet(prediction_colormap_reg)[:, :, :3] * 255) * alpha + wsi_alpha * (1 - alpha)).astype(
                np.uint8)  # [:,:,:3]

            background_mask = np.ones((wsi_thumbnail.height, wsi_thumbnail.width))
            full_slide = wsi_thumbnail.extract_area(0, 0, wsi_thumbnail.width, wsi_thumbnail.height)
            slide_numpy = full_slide.write_to_memory()
            slide_numpy = np.fromstring(slide_numpy, dtype=np.uint8).reshape(full_slide.height, full_slide.width, 3)

            background_mask[slide_numpy[:, :, 1] > 240] = 0
            background_mask = ndimage.binary_closing(background_mask, structure=np.ones((25, 25))).astype(
                background_mask.dtype)
            background_mask = ndimage.binary_opening(background_mask, structure=np.ones((25, 25))).astype(
                background_mask.dtype)

            background_mask = skimage.morphology.remove_small_objects(background_mask.astype(bool), min_size=5000)
            wsi_blended[background_mask == 0] = 255
            wsi_blended_reg[background_mask == 0] = 255

            cm = plt.get_cmap('jet')
            img = Image.fromarray(wsi_blended)
            img.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction.jpeg')
            img_reg = Image.fromarray(wsi_blended_reg)
            img_reg.save(self.dir_results + self.dir_inference + self.current_wsi_name + '/prediction_region.jpeg')
            print('Thumbnail: done')

    def image_normalization(self, x):
        # image resize
        # x = cv2.resize(x, (self.input_shape[1], self.input_shape[2]))
        # channel first
        x = np.transpose(x, (2, 0, 1))
        # intensity normalization
        x = x / 255.0
        # numeric type
        x.astype('float32')
        return x


########################################################################################################################

# Directory listing
dir_out = 'results/nmil/'
dir_inference = 'inference/'
dir_images = 'extracted_tiles_of_.../'
dir_images_regions = 'extracted_tiles_of_..._regions/'
dir_wsi = 'WSIs/'

# WSI to go through
wsi_df = pd.read_excel('patient_set_split.xlsx')
test_wsi = wsi_df.loc[wsi_df['Set'] == 'Test', 'WSI_OLD'].to_list()

# Models
backbone = torch.load('contrastive/models/backbone_....pth')
network = torch.load(dir_out + '1_network_weights_best.pth')

# Iterate over the patients
inference = MILInference(dir_wsi, dir_out, backbone, network, dir_inference, dir_images_regions)
for current_wsi_subfolder in reversed(test_wsi):
    inference.infer(current_wsi_subfolder=dir_images + current_wsi_subfolder)