import os

import torch
import numpy as np
import datetime
from timeit import default_timer as timer
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

from losses_class import FocalTverskyLoss


class NMILTrainer():

    def __init__(self, dir_out, network, lr, id, early_stopping, scheduler,
                 virtual_batch_size, criterion, class_weights, loss_function,
                 tfl_alpha, tfl_gamma, opt_name):

        self.dir_results = dir_out
        if not os.path.isdir(self.dir_results):
            os.mkdir(self.dir_results)

        # Other
        self.init_time = 0
        self.test_generator = None
        self.val_generator = None
        self.train_generator = None

        self.lr = lr
        self.L_epoch = 0
        self.L_lc = []
        self.Lce_lc_val = []
        self.macro_auc_lc_val = []
        self.macro_auc_lc_train = []
        self.f1_lc_val = []
        self.f1_lc_train = []
        self.k2_lc_val = []
        self.k2_lc_train = []
        self.network = network
        self.preds_train = None
        self.refs_train = None

        self.epochs = None
        self.i_epoch = None
        self.iterations = None
        self.criterion = criterion
        self.best_criterion = 0
        self.best_epoch = 0
        self.metrics = {}
        self.id = id
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.virtual_batch_size = virtual_batch_size
        self.class_weights = class_weights
        self.loss_function = loss_function
        self.tfl_alpha = tfl_alpha
        self.tfl_gamma = tfl_gamma
        self.opt_name = opt_name

        # Set optimizers
        self.params = list(self.network.parameters())

        # Set optimizers
        if self.opt_name == 'sgd':
            self.opt = torch.optim.SGD(self.params, lr=self.lr)
        else:
            self.opt = torch.optim.Adam(self.params, lr=self.lr)

        # Set losses
        if self.loss_function == 'tversky':
            self.L = FocalTverskyLoss(alpha=self.tfl_alpha, beta=1 - self.tfl_alpha, gamma=self.tfl_gamma).cuda()
        else:
            self.L = torch.nn.BCEWithLogitsLoss(weight=self.class_weights).cuda()

        if self.criterion == 'loss':
            self.best_criterion = 1000000

    def train(self, train_generator, val_generator, test_generator, epochs):
        self.epochs = epochs
        self.iterations = len(train_generator)
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator

        # Move network to gpu
        self.network.cuda()

        self.init_time = timer()
        for i_epoch in range(epochs):
            self.i_epoch = i_epoch
            # init epoch losses
            self.L_epoch = 0
            self.preds_train = []
            self.refs_train = []

            if self.scheduler:
                if self.i_epoch + 1 == (self.best_epoch + 5):
                    for g in self.opt.param_groups:
                        g['lr'] = self.lr / 2

            # Loop over training dataset
            print('[Training]: at bag level...')
            for self.i_iteration, (X, Y, clinical_data, region_info) in enumerate(self.train_generator):

                X = torch.tensor(X).cuda().float()
                Y = torch.tensor(Y).cuda().float()
                clinical_data = torch.tensor(clinical_data).cuda().float()
                region_info = torch.tensor(region_info).cuda().float()

                # Set model to training mode and clear gradients
                self.network.train()

                # Forward network
                Yhat, _ = self.network(X, clinical_data, region_info)

                # Estimate losses
                Lce = self.L(Yhat, torch.squeeze(Y))

                # Backward gradients
                L = Lce / self.virtual_batch_size
                L.backward()

                # Update weights and clear gradients
                if ((self.i_epoch + 1) % self.virtual_batch_size) == 0:
                    self.opt.step()
                    self.opt.zero_grad()

                ######################################
                ## --- Iteration/Epoch end

                # Save predictions
                self.preds_train.append(Yhat.detach().cpu().numpy())
                self.refs_train.append(Y.detach().cpu().numpy())

                # Display losses per iteration
                self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, self.iterations,
                                    Lce.cpu().detach().numpy(), 0, 0, 0, 0,
                                    end_line='\r')

                # Update epoch's losses
                self.L_epoch += Lce.cpu().detach().numpy() / len(self.train_generator)

            # Epoch-end processes
            self.on_epoch_end()

            if self.early_stopping:
                if (self.i_epoch + 1 >= (self.best_epoch + 30)) and (self.i_epoch + 1 >= 100):
                    break

    def on_epoch_end(self):

        # Obtain epoch-level metrics
        if not np.isnan(np.sum(self.preds_train)):
            macro_auc_train = roc_auc_score(np.squeeze(np.array(self.refs_train)), np.array(self.preds_train),
                                            multi_class='ovr')
        else:
            macro_auc_train = 0.5
        self.macro_auc_lc_train.append(macro_auc_train)

        # Display losses
        self.display_losses(self.i_epoch + 1, self.epochs, self.iterations, self.iterations, self.L_epoch, macro_auc_train,
                            end_line='\n')

        # Update learning curves
        self.L_lc.append(self.L_epoch)

        # Obtain results on train set
        _, _, acc_train, f1_train, k2_train = self.test_bag_level_classification(self.train_generator)
        self.f1_lc_train.append(f1_train)
        self.k2_lc_train.append(k2_train)

        # Display losses
        self.display_losses(self.i_epoch + 1, self.epochs, self.iterations, self.iterations, self.L_epoch,
                            macro_auc_train, acc_train, f1_train, k2_train, end_line='\n')

        # Obtain results on validation set
        Lce_val, macro_auc_val, acc_val, f1_val, k2_val = self.test_bag_level_classification(self.val_generator)

        # Save loss value into learning curve
        self.Lce_lc_val.append(Lce_val)
        self.macro_auc_lc_val.append(macro_auc_val)
        self.f1_lc_val.append(f1_val)
        self.k2_lc_val.append(k2_val)

        metrics = {'epoch': self.i_epoch + 1, 'AUCtrain': np.round(self.macro_auc_lc_train[-1], 4),
                   'AUCval': np.round(self.macro_auc_lc_val[-1], 4), 'F1val': np.round(self.f1_lc_val[-1], 4),
                   'K2val': np.round(self.k2_lc_val[-1], 4)}
        with open(self.dir_results + self.id + 'metrics.json', 'w') as fp:
            json.dump(metrics, fp)
        print(metrics)

        if (self.i_epoch + 1) > 5:
            if self.criterion == 'auc':
                if self.best_criterion < self.macro_auc_lc_val[-1]:
                    self.best_criterion = self.macro_auc_lc_val[-1]
                    self.best_epoch = (self.i_epoch + 1)
                    torch.save(self.network, self.dir_results + self.id + 'network_weights_best.pth')

            elif self.criterion == 'k2':
                if self.best_criterion < self.k2_lc_val[-1]:
                    self.best_criterion = self.k2_lc_val[-1]
                    self.best_epoch = (self.i_epoch + 1)
                    torch.save(self.network, self.dir_results + self.id + 'network_weights_best.pth')

            elif self.criterion == 'f1':
                if self.best_criterion < self.f1_lc_val[-1]:
                    self.best_criterion = self.f1_lc_val[-1]
                    self.best_epoch = (self.i_epoch + 1)
                    torch.save(self.network, self.dir_results + self.id + 'network_weights_best.pth')

            if self.criterion == 'loss':
                if self.best_criterion > self.L_lc[-1]:
                    self.best_criterion = self.L_lc[-1]
                    self.best_epoch = (self.i_epoch + 1)
                    torch.save(self.network, self.dir_results + self.id + 'network_weights_best.pth')

        # Each xx epochs, test models and plot learning curves
        if (self.i_epoch + 1) % 5 == 0:
            # Save weights
            torch.save(self.network, self.dir_results + self.id + 'network_weights.pth')

            # Plot learning curve
            self.plot_learning_curves()

        if (self.epochs == (self.i_epoch + 1)) or (
                self.early_stopping and (self.i_epoch + 1 >= (self.best_epoch + 30) and self.i_epoch + 1 >= 100)):
            print('-' * 20)
            print('-' * 20)

            self.network = torch.load(self.dir_results + self.id + 'network_weights_best.pth')

            # Plot learning curve
            self.plot_learning_curves()

            # Obtain results on validation set
            Lce_val, macro_auc_val, acc_val, f1_val, k2_val = self.test_bag_level_classification(self.val_generator)

            # Obtain results on test set
            Lce_test, macro_auc_test, acc_test, f1_test, k2_test = self.test_bag_level_classification(
                self.test_generator)

            if not self.test_generator.dataset.only_clinical and not (self.network.aggregation in ['mean', 'max']):
                # Test at instance level
                X = self.test_generator.dataset.X[self.test_generator.dataset.y_instances[:, ] != -1,]
                Y = self.test_generator.dataset.y_instances[self.test_generator.dataset.y_instances[:, ] != -1,]
                clinical_data = self.test_generator.dataset.clinical_data
                acc, f1, k2 = self.test_instance_level_classification(X, Y, clinical_data,
                                                                      self.test_generator.dataset.classes)

            else:
                acc, f1, k2 = 0.0, 0.0, 0.0

            metrics = {'epoch': self.best_epoch, 'AUCtest': np.round(macro_auc_test, 4),
                       'acc_test': np.round(acc_test, 4),
                       'f1_test': np.round(f1_test, 4), 'k2_test': np.round(k2_test, 4),
                       'AUCval': np.round(macro_auc_val, 4), 'acc_val': np.round(acc_val, 4),
                       'f1_val': np.round(f1_val, 4), 'k2_val': np.round(k2_val, 4), 'acc_ins': np.round(acc, 4),
                       'f1_ins': np.round(f1, 4), 'k2_ins': np.round(k2, 4),
                       }

            with open(self.dir_results + self.id + 'best_metrics.json', 'w') as fp:
                json.dump(metrics, fp)
            print(metrics)

        self.metrics = metrics
        print('-' * 20)
        print('-' * 20)

    def plot_learning_curves(self):

        def plot_subplot(axes, x, y, y_axis):
            axes.grid()
            for i in range(x.shape[0]):
                axes.plot(x[i, :], y[i, :], 'o-', label=['Train', 'Val'][i])
                axes.legend(loc="upper right")
            axes.set_ylabel(y_axis)

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        plot_subplot(axes[0, 0], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1,
                     np.array([self.L_lc, self.Lce_lc_val]), "Lce")
        plot_subplot(axes[1, 0], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1,
                     np.array([self.macro_auc_lc_train, self.macro_auc_lc_val]), "AUC")
        plot_subplot(axes[0, 1], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1,
                     np.array([self.f1_lc_train, self.f1_lc_val]), "F1")
        plot_subplot(axes[1, 1], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1,
                     np.array([self.k2_lc_train, self.k2_lc_val]), "K2")

        plt.savefig(self.dir_results + self.id + 'learning_curve.png')

    def display_losses(self, i_epoch, epochs, iteration, total_iterations, Lce, macro_auc, acc, f1, k2, end_line=''):

        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} ; AUC={:.4f} ; acc={:.4f} ; f1={:.4f} ; k2={:.4f}".format(
            i_epoch, epochs, iteration, total_iterations, Lce, macro_auc, acc, f1, k2)

        # Print losses
        et = str(datetime.timedelta(seconds=timer() - self.init_time))
        print(info + ',ET=' + et, end=end_line)

    def test_instance_level_classification(self, X, Y, clinical_data, classes):

        self.network.eval()
        print(['INFO: Testing at instance level...'])

        Yhat = []
        for iInstance in np.arange(0, Y.shape[0]):

            # Tensorize input
            if not self.test_generator.dataset.only_clinical:

                x = torch.tensor(X[iInstance]).cuda().float()
                x = x.unsqueeze(0)
            clinical_data_id = torch.tensor(np.array(clinical_data[iInstance]).astype('float32')).cuda().float()

            # Make prediction
            if not self.test_generator.dataset.only_clinical:

                if self.test_generator.dataset.only_images:
                    yhat = torch.softmax(self.network.classifier(torch.squeeze(x)), 0)

                else:
                    yhat = torch.softmax(self.network.classifier(torch.squeeze(torch.cat((x, clinical_data_id), 0))), 0)

            else:
                yhat = torch.softmax(self.network.classifier(torch.squeeze(clinical_data_id, 0)), 0)

            yhat = torch.argmax(yhat).detach().cpu().numpy()
            Yhat.append(yhat)

        Yhat = np.array(Yhat)

        cr = classification_report(Y, Yhat, target_names=classes, digits=4, zero_division=0)
        acc = accuracy_score(Y, Yhat)
        f1 = f1_score(Y, Yhat, average='macro', zero_division=0)
        cm = confusion_matrix(Y, Yhat)
        k2 = cohen_kappa_score(Y, Yhat, weights='quadratic')

        f = open(self.dir_results + self.id + 'report.txt', 'w')
        f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm, k2))
        f.close()

        print('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm, k2))

        return acc, f1, k2

    def test_bag_level_classification(self, test_generator):

        self.network.eval()
        print('[VALIDATION]: at bag level...')

        # Loop over training dataset
        Y_all = []
        Yhat_all = []
        Lce_e = 0
        for self.i_iteration, (X, Y, clinical_data, region_info) in enumerate(test_generator):
            X = torch.tensor(X).cuda().float()
            Y = torch.tensor(Y).cuda().float()
            clinical_data = torch.tensor(clinical_data).cuda().float()
            region_info = torch.tensor(region_info).cuda().float()

            # Forward network
            Yhat, _ = self.network(X, clinical_data, region_info)

            # Estimate losses
            Lce = self.L(Yhat, torch.squeeze(Y))
            Lce_e += Lce.cpu().detach().numpy() / len(test_generator)

            Y_all.append(Y.detach().cpu().numpy())
            Yhat_all.append(Yhat.detach().cpu().numpy())

            # Display losses per iteration
            self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(test_generator),
                                Lce.cpu().detach().numpy(), 0, 0, 0, 0,
                                end_line='\r')

        # Obtain overall metrics
        Yhat_all = np.array(Yhat_all)
        Y_all = np.squeeze(np.array(Y_all))

        if not np.isnan(np.sum(Yhat_all)):
            macro_auc = roc_auc_score(Y_all, Yhat_all, multi_class='ovr')

            # Metrics
            Yhat_mono = np.argmax(Yhat_all, 1)
            Y_mono = np.argmax(Y_all, 1)
            cr = classification_report(Y_mono, Yhat_mono, target_names=test_generator.dataset.classes, digits=4,
                                       zero_division=0)
            acc = accuracy_score(Y_mono, Yhat_mono)
            f1 = f1_score(Y_mono, Yhat_mono, average='macro', zero_division=0)
            cm = confusion_matrix(Y_mono, Yhat_mono)
            k2 = cohen_kappa_score(Y_mono, Yhat_mono, weights='quadratic')

            f = open(self.dir_results + self.id + 'report_bag.txt', 'w')
            f.write(
                'Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm, k2))
            f.close()

            print('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm, k2))

        else:

            macro_auc, acc, f1, k2, = 0.0, 0.0, 0.0, 0.0

        # Display losses per epoch
        self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(test_generator),
                            Lce_e, macro_auc, acc, f1, k2,
                            end_line='\n')

        return Lce_e, macro_auc, acc, f1, k2


class MontecarloInf():
    # def __init__(self, dir_out, network, lr=1*1e-4, pMIL=False, margin=0, t_ic=10,
    #              t_pc=10, alpha_ic=1, alpha_pc=1, alpha_ce=1, id='', early_stopping=False,
    #              scheduler=False, virtual_batch_size=1, criterion='auc', alpha_H=0.01,
    #              backbone_freeze=True, class_weights=torch.ones([2]), loss_function='tversky',
    #              tfl_alpha=0.7, tfl_gamma=4/3, opt_name='sgd'):
    def __init__(self, dir_out, network, lr, pMIL, margin, t_ic,
                 t_pc, alpha_ic, alpha_pc, alpha_ce, id, early_stopping,
                 scheduler, virtual_batch_size, criterion, alpha_H,
                 backbone_freeze, class_weights, loss_function,
                 tfl_alpha, tfl_gamma, opt_name, iteration):

        self.dir_results = dir_out
        if not os.path.isdir(self.dir_results):
            os.mkdir(self.dir_results)

        # Other
        self.best_auc = 0.
        self.init_time = 0
        self.lr = lr
        self.L_epoch = 0
        self.L_lc = []
        self.Lce_lc_val = []
        self.macro_auc_lc_val = []
        self.macro_auc_lc_train = []
        self.f1_lc_val = []
        self.f1_lc_train = []
        self.k2_lc_val = []
        self.k2_lc_train = []
        self.diff = []
        self.i_epoch = 0
        self.epochs = 0
        self.i_iteration = 0
        self.iterations = 0
        self.network = network
        self.test_generator = []
        self.train_generator = []
        self.preds_train = []
        self.refs_train = []
        self.pMIL = pMIL
        self.alpha_ce = alpha_ce
        self.best_criterion = 0
        self.best_epoch = 0
        self.metrics = {}
        self.id = id
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.virtual_batch_size = virtual_batch_size
        self.constrain_cumpliment_lc = []
        self.constrain_proportion_lc = []
        self.criterion = criterion
        self.alpha_H = alpha_H
        self.H_iteration = 0.
        self.H_epoch = 0.
        self.backbone_freeze = backbone_freeze
        self.class_weights = class_weights
        self.loss_function = loss_function
        self.tfl_alpha = tfl_alpha
        self.tfl_gamma = tfl_gamma
        self.opt_name = opt_name
        self.iteration = iteration

        # Set optimizers
        self.params = list(self.network.parameters())
        # Remove encoder params from optimization, requires_grad would also work
        if self.backbone_freeze and self.network.backbone != 'contrastive':
            encoder_params = list(self.network.bb.parameters())
            for encoder_layer_param in encoder_params:
                self.params.remove(encoder_layer_param)

        if self.opt_name == 'sgd':
            self.opt = torch.optim.SGD(self.params, lr=self.lr)
        else:
            self.opt = torch.optim.Adam(self.params, lr=self.lr)

        # Set losses
        if self.loss_function == 'tversky':
            # print('Tversky Focal Loss')
            self.L = FocalTverskyLoss(alpha=self.tfl_alpha, beta=1 - self.tfl_alpha, gamma=self.tfl_gamma).cuda()
        elif network.mode == 'embedding' or network.mode == 'mixed':
            self.L = torch.nn.BCEWithLogitsLoss(weight=self.class_weights).cuda()
        elif network.mode == 'instance':
            self.L = torch.nn.BCELoss(weight=self.class_weights).cuda()

        if self.criterion == 'loss' or self.diff == 'diff':
            self.best_criterion = 1000000

    def montecarlo(self, test_generator, val_generator):
        self.test_generator = test_generator
        self.val_generator = val_generator

        self.network = torch.load(self.dir_results + self.id + 'network_weights_best.pth')
        self.network.eval()

        # Move network to gpu
        self.network.cuda()

        # Obtain results on test set
        self.threshold = self.test_bag_level_classification(self.val_generator, True)
        Lce_test, macro_auc_test, acc_test, f1_test, k2_test, ba_test, ps_test, rs_test = self.test_bag_level_classification(
            self.test_generator, False)

        if not self.test_generator.dataset.only_clinical and not (self.network.aggregation in ['mean', 'max']):
            # Test at instance level
            # X = self.test_generator.dataset.X[self.test_generator.dataset.y_instances[:, 0] != -1, :, :, :]
            # Y = self.test_generator.dataset.y_instances[self.test_generator.dataset.y_instances[:, 0] != -1, :]
            if self.test_generator.dataset.backbone != 'contrastive' or not self.test_generator.dataset.images_on_ram:
                # X = self.test_generator.dataset.X[self.test_generator.dataset.y_instances[:, ] != -1, :, :, :]
                # X = self.test_generator.dataset.images[self.test_generator.dataset.y_instances[:, ] != -1]
                X = self.test_generator.dataset.images
            else:
                X = self.test_generator.dataset.X[self.test_generator.dataset.y_instances[:, ] != -1,]
            Y = self.test_generator.dataset.y_instances[self.test_generator.dataset.y_instances[:, ] != -1,]
            # clinical_data = self.test_generator.dataset.clinical_data[self.test_generator.dataset.y_instances[:, ] != -1, ]
            clinical_data = self.test_generator.dataset.clinical_data
            acc, f1, k2 = self.test_instance_level_classification(X, Y, clinical_data,
                                                                  self.test_generator.dataset.classes)

            metrics = {'epoch': self.best_epoch, 'AUCtest': np.round(macro_auc_test, 4),
                       'acc_test': np.round(acc_test, 4),
                       'f1_test': np.round(f1_test, 4), 'k2_test': np.round(k2_test, 4),
                       'ba_test': np.round(ba_test, 4), 'ps_test': np.round(ps_test, 4),
                       'rs_test': np.round(rs_test, 4),
                       'acc_ins': np.round(acc, 4),
                       'f1_ins': np.round(f1, 4), 'k2_ins': np.round(k2, 4),
                       }

        else:
            metrics = {'epoch': self.best_epoch, 'AUCtest': np.round(macro_auc_test, 4),
                       'acc_test': np.round(acc_test, 4),
                       'f1_test': np.round(f1_test, 4), 'k2_test': np.round(k2_test, 4),
                       'acc_ins': np.round(0.0, 4),
                       'f1_ins': np.round(0.0, 4), 'k2_ins': np.round(0.0, 4),
                       }

        with open(self.dir_results + self.iteration + 'mc_metrics.json', 'w') as fp:
            json.dump(metrics, fp)
        print(metrics)

        self.metrics = metrics
        print('-' * 20)
        print('-' * 20)

    def plot_learning_curves(self):
        def plot_subplot(axes, x, y, y_axis):
            axes.grid()
            for i in range(x.shape[0]):
                axes.plot(x[i, :], y[i, :], 'o-', label=['Train', 'Val'][i])
                axes.legend(loc="upper right")
            axes.set_ylabel(y_axis)

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        plot_subplot(axes[0, 0], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1,
                     np.array([self.L_lc, self.Lce_lc_val]), "Lce")
        plot_subplot(axes[1, 0], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1,
                     np.array([self.macro_auc_lc_train, self.macro_auc_lc_val]), "AUC")
        plot_subplot(axes[0, 1], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1,
                     np.array([self.f1_lc_train, self.f1_lc_val]), "F1")
        plot_subplot(axes[1, 1], np.tile(np.arange(self.i_epoch + 1), (2, 1)) + 1,
                     np.array([self.k2_lc_train, self.k2_lc_val]), "K2")

        plt.savefig(self.dir_results + self.iteration + 'learning_curve.png')

    def display_losses(self, i_epoch, epochs, iteration, total_iterations, Lce, macro_auc, acc, f1, k2, end_line=''):

        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} ; AUC={:.4f} ; acc={:.4f} ; f1={:.4f} ; k2={:.4f}".format(
            i_epoch, epochs, iteration, total_iterations, Lce, macro_auc, acc, f1, k2)

        # Print losses
        et = str(datetime.timedelta(seconds=timer() - self.init_time))
        print(info + ',ET=' + et, end=end_line)

    def test_instance_level_classification(self, X, Y, clinical_data, classes):

        self.network.eval()
        self.enable_dropout(self.network)
        print(['INFO: Testing at instance level...'])

        Yhat = []
        for iInstance in np.arange(0, Y.shape[0]):
            print(str(iInstance+1) + '/' + str(Y.shape[0]), end='\r')

            # Tensorize input
            if not self.test_generator.dataset.only_clinical:

                if self.test_generator.dataset.backbone != 'contrastive' and self.test_generator.dataset.images_on_ram:
                    x = torch.tensor(X[iInstance, :, :, :]).cuda().float()
                elif self.test_generator.dataset.backbone != 'contrastive' and not self.test_generator.dataset.images_on_ram:
                    x = Image.open(X[iInstance])
                    x = np.asarray(x)
                    # Normalization
                    x = self.test_generator.dataset.image_normalization(x)
                    x = torch.tensor(x).cuda().float()
                else:
                    # x = torch.tensor(X[iInstance,:]).cuda().float()
                    x = torch.tensor(X[iInstance]).cuda().float()
                x = x.unsqueeze(0)
            clinical_data_id = torch.tensor(np.array(clinical_data[iInstance]).astype('float32')).cuda().float()

            # Make prediction
            if not self.test_generator.dataset.only_clinical:

                if self.test_generator.dataset.backbone == 'contrastive':
                    yhat = torch.softmax(self.network.classifier(torch.squeeze(x)), 0)

                elif self.network.only_images:
                    yhat = torch.softmax(self.network.classifier(torch.squeeze(self.network.bb(x))), 0)
                    
                else:
                    yhat = torch.softmax(self.network.classifier(
                        torch.squeeze(torch.cat((torch.reshape(self.network.bb(x), (512,)), clinical_data_id), 0))), 0)
            else:
                
                yhat = torch.softmax(self.network.classifier(torch.squeeze(clinical_data_id, 0)), 0)

            yhat = torch.argmax(yhat).detach().cpu().numpy()
            Yhat.append(yhat)

        Yhat = np.array(Yhat)

        cr = classification_report(Y, Yhat, target_names=classes, digits=4, zero_division=0)
        acc = accuracy_score(Y, Yhat)
        f1 = f1_score(Y, Yhat, average='macro', zero_division=0)
        cm = confusion_matrix(Y, Yhat)
        k2 = cohen_kappa_score(Y, Yhat, weights='quadratic')

        # print('Instance Level kappa: ' + str(np.round(k2, 4)), end='\n')

        f = open(self.dir_results + self.iteration + 'report.txt', 'w')
        f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm, k2))
        f.close()

        print('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm, k2))

        return acc, f1, k2

    def test_bag_level_classification(self, test_generator, binary):
        self.network.eval()
        self.enable_dropout(self.network)
        print('[VALIDATION]: at bag level...')

        # Loop over training dataset
        Y_all = []
        Yhat_all = []
        Lce_e = 0
        for self.i_iteration, (X, Y, clinical_data, region_info) in enumerate(test_generator):
            X = torch.tensor(X).cuda().float()
            Y = torch.tensor(Y).cuda().float()
            clinical_data = torch.tensor(clinical_data).cuda().float()
            region_info = torch.tensor(region_info).cuda().float()

            # Forward network
            if self.test_generator.dataset.only_clinical:
                Yhat = self.network(X, clinical_data)
            elif X is not None:
                Yhat, _ = self.network(X, clinical_data, region_info)
            # Estimate losses
            Lce = self.L(Yhat, torch.squeeze(Y))
            Lce_e += Lce.cpu().detach().numpy() / len(test_generator)

            Y_all.append(Y.detach().cpu().numpy())
            Yhat_all.append(Yhat.detach().cpu().numpy())

        # Obtain overall metrics
        Yhat_all = np.array(Yhat_all)
        Y_all = np.squeeze(np.array(Y_all))

        if not binary:
            #     Yhat_all = np.max(Yhat_all, 1)
            #     Y_all = np.max(Y_all, 1)

            if not np.isnan(np.sum(Yhat_all)):
                macro_auc = roc_auc_score(Y_all, Yhat_all, multi_class='ovr')

                # Metrics
                Yhat_mono = np.zeros((len(Yhat_all),)).astype(np.int32)
                for index, pred in enumerate(Yhat_all):
                    if pred[1] >= self.threshold:
                        Yhat_mono[index] = 1
                Y_mono = np.argmax(Y_all, 1)  # softmax(Y_all, 1)[:,1]
                cr = classification_report(Y_mono, Yhat_mono, target_names=test_generator.dataset.classes, digits=4,
                                           zero_division=0)
                # print(Y_mono)
                # print(Yhat_mono)
                acc = accuracy_score(Y_mono, Yhat_mono)
                f1 = f1_score(Y_mono, Yhat_mono, average='macro', zero_division=0)
                cm = confusion_matrix(Y_mono, Yhat_mono)
                k2 = cohen_kappa_score(Y_mono, Yhat_mono, weights='quadratic')
                ba = balanced_accuracy_score(Y_mono, Yhat_mono)
                ps = precision_score(Y_mono, Yhat_mono)
                rs = recall_score(Y_mono, Yhat_mono)

                f = open(self.dir_results + self.iteration + 'report_bag.txt', 'w')
                f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm,
                                                                                                                 k2))
                f.close()

                print('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nKappa\n\n{}\n'.format(cr, cm,
                                                                                                               k2))
            else:

                macro_auc, acc, f1, k2, = 0.0, 0.0, 0.0, 0.0

            return Lce_e, macro_auc, acc, f1, k2, ba, ps, rs

        else:
            X_scan = Yhat_all[:, 1]
            Y_mono = np.argmax(Y_all, 1)  # softmax(Y_all, 1)[:,1]

            scan_range = range(0, 105, 1)

            results_dict = {
                'threshold': 0,
                'acc': 0,
                'f1': 0,
                'k2': 0,
                'cm': 0,
                'cr': 0,
                'ba': 0,
                'ps': 0,
                'rs': 0,
            }

            for threshold in scan_range:
                threshold /= 100

                Y_scan = np.zeros((len(Yhat_all),))
                for index, pred in enumerate(X_scan):
                    if pred >= threshold:
                        Y_scan[index] = 1

                cr = classification_report(Y_mono, Y_scan, target_names=test_generator.dataset.classes, digits=4,
                                           zero_division=0)
                acc = accuracy_score(Y_mono, Y_scan)
                f1 = f1_score(Y_mono, Y_scan, average='macro', zero_division=0)
                cm = confusion_matrix(Y_mono, Y_scan)
                k2 = cohen_kappa_score(Y_mono, Y_scan, weights='quadratic')
                ba = balanced_accuracy_score(Y_mono, Y_scan)
                ps = precision_score(Y_mono, Y_scan)
                rs = recall_score(Y_mono, Y_scan)

                if results_dict['ba'] < ba:
                    results_dict = {
                        'threshold': threshold,
                        'acc': acc,
                        'f1': f1,
                        'k2': k2,
                        'cm': cm,
                        'cr': cr,
                        'ba': ba,
                        'ps': ps,
                        'rs': rs,
                    }

            for key in results_dict.keys():
                print('')
                print(key)
                print(results_dict[key])

            return results_dict['threshold']

    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                m.p = 0.05
