import torch


class NMILArchitecture(torch.nn.Module):

    def __init__(self, classes, aggregation, only_images, only_clinical, clinical_classes, neurons_1, neurons_2,
                 neurons_3, neurons_att_1, neurons_att_2, dropout_rate):
        super(NMILArchitecture, self).__init__()

        self.classes = classes
        self.only_images = only_images
        self.only_clinical = only_clinical
        self.nClasses = len(classes)

        self.neurons_1 = neurons_1
        self.neurons_2 = neurons_2
        self.neurons_3 = neurons_3
        self.neurons_att_1 = neurons_att_1
        self.neurons_att_2 = neurons_att_2
        self.dropout_rate = dropout_rate
        self.aggregation = aggregation

        # Classifiers
        if self.only_images and (self.only_images != self.only_clinical):
            self.classifier = torch.nn.Sequential(
                                    torch.nn.Linear(self.neurons_1, self.neurons_2),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(self.dropout_rate),
                                    torch.nn.Linear(self.neurons_2, self.neurons_3),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(self.dropout_rate),
                                    torch.nn.Linear(self.neurons_3, self.nClasses),
                                )
        elif self.only_clinical:
            self.classifier = torch.nn.Sequential(
                                    torch.nn.Linear(len(clinical_classes), self.neurons_2),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(self.dropout_rate),
                                    torch.nn.Linear(self.neurons_2, self.neurons_3),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(self.dropout_rate),
                                    torch.nn.Linear(self.neurons_3, self.nClasses),
                                )
        else:
            self.neurons_1 = self.neurons_1 + len(clinical_classes)
            self.classifier = torch.nn.Sequential(
                                    torch.nn.Linear(self.neurons_1, self.neurons_2),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(self.dropout_rate),
                                    torch.nn.Linear(self.neurons_2, self.neurons_3),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(self.dropout_rate),
                                    torch.nn.Linear(self.neurons_3, self.nClasses),
                                )

        # MIL aggregation
        self.milAggregation = MILAggregation(aggregation=aggregation, L=self.neurons_att_1, D=self.neurons_att_2)

    def forward(self, features, clinical_data, region_info):

        if self.only_clinical:

            global_classification = self.classifier(clinical_data)

            return global_classification, None

        else:

            if self.aggregation == 'attentionMIL':

                instance_classification = []
                region_embeddings_list = []
                for region_id in range(int(torch.max(region_info).item())+1):
                    region_features = features[torch.where(region_info == region_id, True, False)]
                    if len(region_features) == 1:
                        region_embeddings_list.append(torch.squeeze(region_features, dim=0))
                        A_V = self.milAggregation.attentionModule.attention_V(region_features)  # Attention
                        A_U = self.milAggregation.attentionModule.attention_U(region_features)  # Gate
                        w_logits = self.milAggregation.attentionModule.attention_weights(A_V * A_U)  # Probabilities - softmax over instances
                        instance_classification.append(w_logits)
                        continue
                    embedding_reg, w_reg = self.milAggregation(torch.squeeze(region_features))
                    instance_classification.append(w_reg)
                    region_embeddings_list.append(embedding_reg)

                if len(region_embeddings_list) == 1:
                    embedding = embedding_reg
                    patch_classification = w_reg
                else:
                    embedding, w = self.milAggregation(torch.squeeze(torch.stack(region_embeddings_list)))
                    patch_classification = torch.cat(instance_classification, dim=0)

                if self.only_images:
                    global_classification = self.classifier(embedding)
                else:
                    global_classification = self.classifier(torch.cat((embedding, clinical_data), 0))

            elif self.aggregation in ['mean', 'max']:
                embedding = self.milAggregation(torch.squeeze(features))
                if self.only_images:
                    global_classification = self.classifier(embedding)
                else:
                    global_classification = self.classifier(torch.cat((embedding,clinical_data), 0))
                patch_classification = torch.softmax(self.classifier(torch.squeeze(features)), 1)

            return global_classification, patch_classification


class MILAggregation(torch.nn.Module):
    def __init__(self, aggregation, L, D):
        super(MILAggregation, self).__init__()

        self.aggregation = aggregation
        self.L = L
        self.D = D

        if self.aggregation == 'attentionMIL':
            self.attentionModule = attentionMIL(self.L, self.D, 1)

    def forward(self, feats):

        if self.aggregation == 'max':
            embedding = torch.max(feats, dim=0)[0]
            return embedding
        elif self.aggregation == 'mean':
            embedding = torch.mean(feats, dim=0)
            return embedding
        elif self.aggregation == 'attentionMIL':
            # Attention embedding from Ilse et al. (2018) for MIL. It only works at the binary scenario at instance-level
            embedding, w_logits = self.attentionModule(feats)
            return embedding, torch.softmax(w_logits, dim=0)


class attentionMIL(torch.nn.Module):
    def __init__(self, L, D, K):
        super(attentionMIL, self).__init__()

        # Attention embedding from Ilse et al. (2018) for MIL. It only works at the binary scenario.

        self.L = L
        self.D = D
        self.K = K
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Tanh()
        )
        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Sigmoid()
        )
        self.attention_weights = torch.nn.Linear(self.D, self.K)

    def forward(self, feats):
        # Attention weights computation
        A_V = self.attention_V(feats)  # Attention
        A_U = self.attention_U(feats)  # Gate
        w_logits = self.attention_weights(A_V * A_U)  # Probabilities - softmax over instances

        # Weighted average computation per class
        feats = torch.transpose(feats, 1, 0)
        embedding = torch.squeeze(torch.mm(feats, torch.softmax(w_logits, dim=0)))  # KxL

        return embedding, w_logits