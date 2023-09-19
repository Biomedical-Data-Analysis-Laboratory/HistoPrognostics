import os

import pandas as pd
import torch
import torchvision
import kornia
from losses_con import SupConLoss
from dataset_multiscale import *

torch.cuda.empty_cache()
train_gpu = torch.cuda.is_available()
torch.cuda.manual_seed(42)


# Train variables
roi_choice = 'front' # ROI choice
config_run_list = [0, 1, 2, 3, 4, 5]
# 0: ('ImageNet backbone weights')
# 1: ('Multi-task learning: Unsupervised Contrastive Loss + Binary CrossEntropy')
# 2: ('Supervised Classification with Binary CrossEntropy')
# 3: ('Supervised Contrastive Loss')
# 4: ('Unsupervised Contrastive Loss')
# 5: ('Supervised Contrastive Loss: BCG weak labels')

generate_embedding = True # Generate embeddings after training
supervised = True # Supervised modality
multi_task_learning = False # Multi-task modality
retrain_backbone = True # Train backbone anew or load previously trained backbone

# Hyperparameters
input_shape=(3, 512, 512)
bs = 16
lr = 0.0001
epochs = 10
alpha_ce = 0.5

# Extraction repository where tiles are stored
dir_images = 'extracted_tiles_of_' + roi_choice + '_multiscale/'
clinical_dataframe = pd.read_excel('clinical_data.xlsx')

# Obtain the list of valid WSI filenames
wsi_list = os.listdir(dir_images)

wsi_df = pd.read_excel('patient_set_split.xlsx')
train_list = wsi_df.loc[wsi_df['Set'] == 'Train', 'WSI_OLD'].to_list()
val_list = wsi_df.loc[wsi_df['Set'] == 'Val', 'WSI_OLD'].to_list()
test_list = wsi_df.loc[wsi_df['Set'] == 'Test', 'WSI_OLD'].to_list()

# Not all slides are annotated
if roi_choice == "anno":
    for train_wsi in (x for x in train_list if x not in wsi_list):
        train_list.remove(train_wsi)
    for val_wsi in (x for x in val_list if x not in wsi_list):
        val_list.remove(val_wsi)
    for test_wsi in (x for x in test_list if x not in wsi_list):
        test_list.remove(test_wsi)

# Chose which configuration, type of contrastive learning
print('')
for config in config_run_list:
    print('*' * 100)
    if config == 0:
        print('ImageNet backbone weights')
        retrain_backbone = False
    elif config == 1:
        print('Multi-task learning: Unsupervised Contrastive Loss + Binary CrossEntropy')
    elif config == 2:
        print('Supervised Classification with Binary CrossEntropy')
    elif config == 3:
        print('Supervised Contrastive Loss')
    elif config == 4:
        print('Unsupervised Contrastive Loss')
    elif config == 5:
        print('Supervised Contrastive Loss: BCG weak labels')
    print('*' * 100)

    weights_directory = 'models/'
    if config == 0:
        output_dataset_directory = 'data/embeddings_' + roi_choice + '_multi_imagenet/'
        info_filename = 'report_' + roi_choice + '_multi_imagenet.txt'
    elif config == 1:
        output_dataset_directory = 'data/embeddings_' + roi_choice + '_multi_multi/'
        info_filename = 'report_' + roi_choice + '_multi_multi.txt'
    elif config == 2:
        output_dataset_directory = 'data/embeddings_ce/'
        info_filename = 'report_ce.txt'
    elif config == 3:
        output_dataset_directory = 'data/embeddings_supervised/'
        info_filename = 'report_supervised.txt'
    elif config == 5:
        output_dataset_directory = 'data/embeddings_' + roi_choice + '_multi_bcgweak/'
        info_filename = 'report_' + roi_choice + '_multi_bcgweak.txt'
    else:
        output_dataset_directory = 'data/embeddings_' + roi_choice + '_multi_unsupervised/'
        info_filename = 'report_' + roi_choice + '_multi_unsupervised.txt'

    if retrain_backbone:
        dataset_train = Dataset(dir_images, clinical_dataframe, input_shape=input_shape, augmentation=False,
                                preallocate=False, inference=False, wsi_list=train_list, config=config)
        train_generator = Generator(dataset_train, bs, shuffle=True, augmentation=False)

    # Prepare model backbone
    model = torchvision.models.densenet121(pretrained=True)
    modules = list(model.children())[:-1]

    backbone_400x = torch.nn.Sequential(*modules, torch.nn.AdaptiveMaxPool2d(1))
    backbone_100x = torch.nn.Sequential(*modules, torch.nn.AdaptiveMaxPool2d(1))
    backbone_25x = torch.nn.Sequential(*modules, torch.nn.AdaptiveMaxPool2d(1))

    # Prepare augmentations module
    transforms = torch.nn.Sequential(
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.RandomRotation(degrees=45, p=0.5),
        kornia.augmentation.RandomAffine(degrees=0, scale=(0.95, 1.20), p=0.5),
        kornia.augmentation.RandomAffine(degrees=0, translate=(0.05, 0), p=0.5),
        kornia.augmentation.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0., p=0.5),
        # kornia.augmentation.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 2.3), value=1.0, p=0.2),
        # kornia.augmentation.RandomElasticTransform(kernel_size=(63, 63), sigma=(32.0, 32.0), alpha=(1.0, 1.0), p=0.25)
    )

    # Prepare projection head
    projection = torch.nn.Sequential(torch.nn.Linear(3072, 128),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(128, 128))

    # Prepare crossentropy classifier
    classifier = torch.nn.Sequential(torch.nn.Linear(3072, 128),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(128, 8))

    # Set loss function
    Lsimclr = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)
    Lce = torch.nn.CrossEntropyLoss().cuda()

    # Prepare optimizer
    if config not in [1, 2]:
        trainable_parameters = list(backbone_400x.parameters()) + list(backbone_100x.parameters()) + list(backbone_25x.parameters()) + list(projection.parameters())
    else:
        trainable_parameters = list(backbone_400x.parameters()) + list(backbone_100x.parameters()) + list(backbone_25x.parameters()) + list(projection.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(lr=lr, params=trainable_parameters)

    ########################################################################################################################
    # TRAIN BACKBONE
    ########################################################################################################################

    if train_gpu:
        backbone_400x.cuda()
        backbone_100x.cuda()
        backbone_25x.cuda()
        projection.cuda()
        transforms.cuda()
        classifier.cuda()
        Lsimclr.cuda()
    torch.save(backbone_400x, weights_directory + 'backbone_400x_imagenet.pth')
    torch.save(backbone_100x, weights_directory + 'backbone_100x_imagenet.pth')
    torch.save(backbone_25x, weights_directory + 'backbone_25x_imagenet.pth')

    if retrain_backbone and (config != 0):

        for i_epoch in range(epochs):
            L_epoch = 0.0

            for i_iteration, (X400, X100, X25, Y) in enumerate(train_generator):
                ####################
                # --- Training epoch
                model.train()

                X400 = torch.tensor(X400).cuda().float()
                X100 = torch.tensor(X100).cuda().float()
                X25 = torch.tensor(X25).cuda().float()
                Y = torch.tensor(Y).cuda().float()

                # Move to cuda
                if train_gpu:
                    X400.cuda()
                    X100.cuda()
                    X25.cuda()
                    Y.cuda()

                # Forward
                F_o_400x = torch.squeeze(backbone_400x(X400))
                F_o_100x = torch.squeeze(backbone_100x(X100))
                F_o_25x = torch.squeeze(backbone_25x(X25))
                F_o = torch.cat((F_o_400x, F_o_100x, F_o_25x), 1)

                if (config != 2) or (config == 1):

                    # Augmentation
                    Xa_400x = transforms(X400.clone())
                    Xa_100x = transforms(X100.clone())
                    Xa_25x = transforms(X25.clone())

                    # Forward augmentation
                    F_a_400x = torch.squeeze(backbone_400x(Xa_400x))
                    F_a_100x = torch.squeeze(backbone_100x(Xa_100x))
                    F_a_25x = torch.squeeze(backbone_25x(Xa_25x))
                    F_a = torch.cat((F_a_400x, F_a_100x, F_a_25x), 1)

                    # Projection
                    F_o_proj = projection(F_o).unsqueeze(1)
                    F_a_proj = projection(F_a).unsqueeze(1)

                    # Normalization
                    F_o_proj = torch.nn.functional.normalize(F_o_proj, dim=-1)
                    F_a_proj = torch.nn.functional.normalize(F_a_proj, dim=-1)

                    # Loss
                    if ((config == 3) or (config == 5)) and (config != 1):
                        L_iteration = Lsimclr(torch.cat([F_o_proj, F_a_proj], 1), labels=Y, mask=None)
                    else:
                        L_iteration = Lsimclr(torch.cat([F_o_proj, F_a_proj], 1), labels=None, mask=None)

                # Multi-task learning for tissue type classification
                if (config == 1) or (config == 2):

                    if sum(i!=-1 for i in Y).bool():

                        Xtt = F_o[Y != -1].cuda()
                        Ytt = Y[Y != -1].cuda()

                        # Constrain the impact of the classification based in the proportion of labeled samples
                        label_prop = len(Ytt) / len(Y)

                        y_hat = torch.squeeze(classifier(Xtt))

                        if len(Ytt) == 1:
                            y_hat = torch.unsqueeze(y_hat, 0)

                        if not multi_task_learning:
                            L_iteration = alpha_ce * label_prop * Lce(y_hat, Ytt.type(torch.LongTensor).cuda())
                        else:
                            L_iteration += alpha_ce * label_prop * Lce(y_hat, Ytt.type(torch.LongTensor).cuda())

                # Backward and weights update
                L_iteration.backward()  # Backward
                opt.step()              # Update weights
                opt.zero_grad()         # Clear gradients

                L_epoch += L_iteration.cpu().detach().numpy() / len(train_generator)

                # Display training information per iteration
                info = "[INFO] Epoch {}/{}  -- Step {}/{}: L_contrastive={:.6f}".format(
                    i_epoch + 1, epochs, i_iteration + 1, len(train_generator), L_iteration.cpu().detach().numpy())
                print(info, end='\r')

            # Display training information per epoch
            info = "[INFO] Epoch {}/{}  -- Step {}/{}: L_contrastive={:.6f}".format(
                i_epoch + 1, epochs, i_iteration + 1, len(train_generator), L_epoch)
            print(info, end='\n')

            f = open(info_filename, 'w')
            f.write(info)
            f.close()

            # Save last model
            if config == 1:
                torch.save(backbone_400x, weights_directory + 'backbone_400x_contrastive_' + roi_choice + '_multi_multi.pth')
                torch.save(backbone_100x, weights_directory + 'backbone_100x_contrastive_' + roi_choice + '_multi_multi.pth')
                torch.save(backbone_25x, weights_directory + 'backbone_25x_contrastive_' + roi_choice + '_multi_multi.pth')
                torch.save(projection, weights_directory + 'projection_contrastive_' + roi_choice + '_multi_multi.pth')
                torch.save(classifier, weights_directory + 'classifier_contrastive_' + roi_choice + '_multi_multi.pth')
            elif config == 2:
                torch.save(backbone_400x, weights_directory + 'backbone_400x_contrastive_' + roi_choice + '_multi_ce.pth')
                torch.save(backbone_100x, weights_directory + 'backbone_100x_contrastive_' + roi_choice + '_multi_ce.pth')
                torch.save(backbone_25x, weights_directory + 'backbone_25x_contrastive_' + roi_choice + '_multi_ce.pth')
                torch.save(classifier, weights_directory + 'classifier_contrastive_' + roi_choice + '_multi_ce.pth')
            elif config == 3:
                torch.save(backbone_400x, weights_directory + 'backbone_400x_contrastive_' + roi_choice + '_multi_supervised.pth')
                torch.save(backbone_100x, weights_directory + 'backbone_100x_contrastive_' + roi_choice + '_multi_supervised.pth')
                torch.save(backbone_25x, weights_directory + 'backbone_25x_contrastive_' + roi_choice + '_multi_supervised.pth')
                torch.save(projection, weights_directory + 'projection_contrastive_' + roi_choice + '_multi_supervised.pth')
            elif config == 5:
                torch.save(backbone_400x, weights_directory + 'backbone_400x_contrastive_' + roi_choice + '_multi_bcgweak.pth')
                torch.save(backbone_100x, weights_directory + 'backbone_100x_contrastive_' + roi_choice + '_multi_bcgweak.pth')
                torch.save(backbone_25x, weights_directory + 'backbone_25x_contrastive_' + roi_choice + '_multi_bcgweak.pth')
                torch.save(projection, weights_directory + 'projection_contrastive_' + roi_choice + '_multi_bcgweak.pth')
            else:
                torch.save(backbone_400x, weights_directory + 'backbone_400x_contrastive_' + roi_choice + '_multi_unsupervised.pth')
                torch.save(backbone_100x, weights_directory + 'backbone_100x_contrastive_' + roi_choice + '_multi_unsupervised.pth')
                torch.save(backbone_25x, weights_directory + 'backbone_25x_contrastive_' + roi_choice + '_multi_unsupervised.pth')
                torch.save(projection, weights_directory + 'projection_contrastive_' + roi_choice + '_multi_unsupervised.pth')

    ########################################################################################################################
    # INFERENCE EMBEDDING GENERATION
    ########################################################################################################################

    if generate_embedding:

        if not os.path.exists(output_dataset_directory):
            os.mkdir(output_dataset_directory)

        # Load last model
        if config == 0:
            backbone_inference_400x = torch.load(weights_directory + 'backbone_400x_imagenet.pth')
            backbone_inference_100x = torch.load(weights_directory + 'backbone_100x_imagenet.pth')
            backbone_inference_25x = torch.load(weights_directory + 'backbone_25x_imagenet.pth')
        elif config == 1:
            backbone_inference_400x = torch.load(weights_directory + 'backbone_400x_contrastive_' + roi_choice + '_multi_multi.pth')
            backbone_inference_100x = torch.load(weights_directory + 'backbone_100x_contrastive_' + roi_choice + '_multi_multi.pth')
            backbone_inference_25x = torch.load(weights_directory + 'backbone_25x_contrastive_' + roi_choice + '_multi_multi.pth')
        elif config == 2:
            backbone_inference_400x = torch.load(weights_directory + 'backbone_400x_contrastive_' + roi_choice + '_multi_ce.pth')
            backbone_inference_100x = torch.load(weights_directory + 'backbone_100x_contrastive_' + roi_choice + '_multi_ce.pth')
            backbone_inference_25x = torch.load(weights_directory + 'backbone_25x_contrastive_' + roi_choice + '_multi_ce.pth')
        elif config == 3:
            backbone_inference_400x = torch.load(weights_directory + 'backbone_400x_contrastive_' + roi_choice + '_multi_supervised.pth')
            backbone_inference_100x = torch.load(weights_directory + 'backbone_100x_contrastive_' + roi_choice + '_multi_supervised.pth')
            backbone_inference_25x = torch.load(weights_directory + 'backbone_25x_contrastive_' + roi_choice + '_multi_supervised.pth')
        elif config == 5:
            backbone_inference_400x = torch.load(weights_directory + 'backbone_400x_contrastive_' + roi_choice + '_multi_bcgweak.pth')
            backbone_inference_100x = torch.load(weights_directory + 'backbone_100x_contrastive_' + roi_choice + '_multi_bcgweak.pth')
            backbone_inference_25x = torch.load(weights_directory + 'backbone_25x_contrastive_' + roi_choice + '_multi_bcgweak.pth')
        else:
            backbone_inference_400x = torch.load(weights_directory + 'backbone_400x_contrastive_' + roi_choice + '_multi_unsupervised.pth')
            backbone_inference_100x = torch.load(weights_directory + 'backbone_100x_contrastive_' + roi_choice + '_multi_unsupervised.pth')
            backbone_inference_25x = torch.load(weights_directory + 'backbone_25x_contrastive_' + roi_choice + '_multi_unsupervised.pth')
            

        # List all files to be processed
        subfolder_list = []
        filename_list = []

        # for subfolder in os.listdir(dir_images):
        embedding_extraction_list = train_list + val_list + test_list
        for subfolder in embedding_extraction_list:
            subfolder_list.append(os.path.join(dir_images,subfolder))
            filename_list.append(subfolder)
        
        for wsi_folder, filename in zip(subfolder_list, filename_list):

            print(filename)
            print(wsi_folder)
            if os.path.exists(output_dataset_directory + filename + '.npy'):
                continue

            # If there are no tiles, skip to the next one
            if roi_choice == 'anno' and not os.path.isdir(wsi_folder): continue
            if len(os.listdir(os.path.join(wsi_folder, '400x'))) == 0: continue

            # Just use this to know we are working on it
            np.save(output_dataset_directory + filename + '.npy', np.zeros((1,1)))

            # Set up current WSI data
            dataset_wsi = Dataset(wsi_folder, clinical_dataframe, input_shape=input_shape, augmentation=False,
                                    preallocate=False, inference=True)
            wsi_generator = Generator(dataset_wsi, bs, shuffle=False, augmentation=False)

            # Store outputs in a list
            wsi_embeddings_list = []
            wsi_norm_list = []

            for (X400, X100, X25, Y) in wsi_generator:
                ####################
                # --- Inference
                backbone_inference_400x.eval()
                backbone_inference_100x.eval()
                backbone_inference_25x.eval()

                X400 = torch.tensor(X400).cuda().float()
                X100 = torch.tensor(X100).cuda().float()
                X25 = torch.tensor(X25).cuda().float()

                # Move to cuda
                if train_gpu:
                    X400.cuda()
                    X100.cuda()
                    X25.cuda()

                # Forward
                feature_embeddings_400x = torch.squeeze(backbone_inference_400x(X400))
                feature_embeddings_100x = torch.squeeze(backbone_inference_100x(X100))
                feature_embeddings_25x = torch.squeeze(backbone_inference_25x(X25))
                if len(feature_embeddings_400x.shape) == 1:
                    feature_embeddings = torch.cat((feature_embeddings_400x, feature_embeddings_100x, feature_embeddings_25x), 0)
                    feature_embeddings = torch.unsqueeze(feature_embeddings, dim=0)
                else:
                    feature_embeddings = torch.cat((feature_embeddings_400x, feature_embeddings_100x, feature_embeddings_25x), 1)

                # Save embeddings
                wsi_embeddings_list.append(feature_embeddings.cpu().detach().numpy())

            # Convert list to numpy and save it in output dataset directory
            if len(wsi_embeddings_list) == 0: continue
            np.save(output_dataset_directory + filename + '.npy', np.vstack(wsi_embeddings_list))

