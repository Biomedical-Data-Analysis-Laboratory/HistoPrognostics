import os

import pandas as pd
import numpy as np
import json
import random
import torch

from datasets_nmil import *
from model_nmil import *
from train_nmil import *

import argparse

torch.cuda.empty_cache()

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def main(args):

    # Get WSI/patients for each of the sets
    wsi_df = pd.read_excel(args.patient_set_split)
    train_wsi = wsi_df.loc[wsi_df['Set'] == 'Train', 'WSI_OLD'].to_list()
    val_wsi = wsi_df.loc[wsi_df['Set'] == 'Val', 'WSI_OLD'].to_list()
    test_wsi = wsi_df.loc[wsi_df['Set'] == 'Test', 'WSI_OLD'].to_list()

    metrics = []

    for i_iteration in np.arange(0, args.iterations):
        id = str(i_iteration) + '_'

        # Set data generators
        dataset_train = NMILDataset(args.dir_images, args.dir_embeddings, train_wsi, args.classes,
                                    clinical_dataframe=args.clinical_dataframe,
                                    data_augmentation=args.data_augmentation,
                                    channel_first=True,
                                    only_clinical=args.only_clinical,
                                    clinical_parameters=args.clinical_data)
        data_generator_train = NMILDataGenerator(dataset_train, batch_size=1, shuffle=True,
                                                 max_instances=args.max_instances)

        dataset_val = NMILDataset(args.dir_images, args.dir_embeddings, val_wsi, args.classes,
                                  clinical_dataframe=args.clinical_dataframe,
                                  data_augmentation=args.data_augmentation,
                                  channel_first=True,
                                  only_clinical=args.only_clinical,
                                  clinical_parameters=args.clinical_data)
        data_generator_val = NMILDataGenerator(dataset_val, batch_size=1, shuffle=False, max_instances=1000000)

        dataset_test = NMILDataset(args.dir_images, args.dir_embeddings, test_wsi, args.classes,
                                   clinical_dataframe=args.clinical_dataframe,
                                   data_augmentation=args.data_augmentation,
                                   channel_first=True,
                                   only_clinical=args.only_clinical,
                                   clinical_parameters=args.clinical_data)
        data_generator_test = NMILDataGenerator(dataset_test, batch_size=1, shuffle=False, max_instances=1000000)

        # Set network architecture
        network = NMILArchitecture(args.classes, aggregation=args.aggregation,
                                   only_images=args.only_images, only_clinical=args.only_clinical,
                                   clinical_classes=args.clinical_data, neurons_1=args.neurons_1,
                                   neurons_2=args.neurons_2, neurons_3=args.neurons_3,
                                   neurons_att_1=args.neurons_att_1, neurons_att_2=args.neurons_att_2,
                                   dropout_rate=args.dropout_rate)

        if args.class_weights_enable:
            class_weights = torch.mul(torch.softmax(torch.tensor(
                np.array([1, len(dataset_train.y_instances) / sum(dataset_train.y_instances)])), dim=0), 2)
        else:
            class_weights = torch.ones([2])

        # Perform training
        trainer = NMILTrainer(dir_out=args.dir_results + args.experiment_name + '/', network=network,
                              lr=args.lr, id=id, early_stopping=args.early_stopping, scheduler=args.scheduler,
                              virtual_batch_size=args.virtual_batch_size,
                              criterion=args.criterion, class_weights=class_weights,
                              loss_function=args.loss_function, tfl_alpha=args.tfl_alpha,
                              tfl_gamma=args.tfl_gamma, opt_name=args.opt_name)
        trainer.train(train_generator=data_generator_train, val_generator=data_generator_val,
                      test_generator=data_generator_test, epochs=args.epochs)

        metrics.append([list(trainer.metrics.values())[1:]])

    # Get overall metrics
    metrics = np.squeeze(np.array(metrics))

    if args.iterations > 1:
        mu = np.mean(metrics, axis=0)
        std = np.std(metrics, axis=0)

        info = "AUCtest={:.4f}({:.4f}); AUCval={:.4f}({:.4f}); acc={:.4f}({:.4f}); f1-score={:.4f}({:.4f}); k2={:.4f}({:.4f})".format(
            mu[0], std[0], mu[4], std[4], mu[5], std[5], mu[6], std[6], mu[7], std[7])
    else:
        info = "AUCtest={:.4f}; AUCval={:.4f}; acc={:.4f}; f1-score={:.4f}; k2={:.4f}".format(
            metrics[0], metrics[4], metrics[5], metrics[6], metrics[7])

    f = open(args.dir_results + args.experiment_name + '/' + 'method_metrics.txt', 'w')
    f.write(info)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument("--dir_images", default='extracted_tiles_of_.../', type=str)
    parser.add_argument("--dir_embeddings", default='../contrastive/data/embeddings_.../', type=str)
    parser.add_argument("--patient_set_split", default='patient_set_split.xlsx', type=str)
    parser.add_argument("--clinical_dataframe", default='clinical_dataframe.xlsx', type=str)
    parser.add_argument("--dir_results", default='results/', type=str)
    parser.add_argument("--experiment_name", default="nmil", type=str)

    # Dataset
    parser.add_argument("--classes", default=['Responsive', 'Failure'], type=list)
    parser.add_argument("--clinical_data", default=['Yrs_age', 'Gender', 'Smoking'], type=list)

    # Architecture
    parser.add_argument("--only_images", default=True, type=bool)
    parser.add_argument("--only_clinical", default=False, type=bool)
    parser.add_argument("--aggregation", default="attentionMIL", type=str)

    # Hyperparameters
    parser.add_argument("--iterations", default=5, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--max_instances", default=64, type=int)
    parser.add_argument("--criterion", default='auc', type=str)
    parser.add_argument("--opt_name", default="sgd", type=str)
    parser.add_argument("--lr", default=1 * 1e-2, type=float)
    parser.add_argument("--loss_function", default="tversky", type=str)
    parser.add_argument("--early_stopping", default=True, type=bool)
    parser.add_argument("--scheduler", default=True, type=bool)
    parser.add_argument("--neurons_1", default=1024, type=int)
    parser.add_argument("--neurons_2", default=4096, type=int)
    parser.add_argument("--neurons_3", default=2048, type=int)
    parser.add_argument("--neurons_att_1", default=1024, type=int)
    parser.add_argument("--neurons_att_2", default=4096, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--tfl_alpha", default=0.9, type=float)
    parser.add_argument("--tfl_gamma", default=2, type=float)
    parser.add_argument("--class_weights_enable", default=True, type=bool)
    parser.add_argument("--virtual_batch_size", default=1, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.dir_results + args.experiment_name):
        os.mkdir(args.dir_results + args.experiment_name)

    with open(args.dir_results + args.experiment_name + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)