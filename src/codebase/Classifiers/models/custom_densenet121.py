import warnings;
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
import pickle


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=True)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Modified from torchvision to have a variable number of input channels

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4,
                 drop_rate=0, num_classes=18, in_channels=1, weights=None, op_threshs=None, progress=True,
                 apply_sigmoid=False, return_logit=False):

        super(DenseNet, self).__init__()
        self.drop_rate = drop_rate
        self.apply_sigmoid = apply_sigmoid
        self.return_logit = return_logit
        self.weights = weights
        self.gradients = None  # required for gradcam_forward

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        # needs to be register_buffer here so it will go to cuda/cpu easily
        self.register_buffer('op_threshs', op_threshs)

        if self.weights != None:
            self.weights_filename_local = self.weights
            print("weights_filename_local: ", self.weights_filename_local)

            try:
                savedmodel = torch.load(self.weights_filename_local)
                print("model loaded", self.weights_filename_local)
                # patch to load old models https://github.com/pytorch/pytorch/issues/42242
                for mod in savedmodel.modules():
                    if not hasattr(mod, "_non_persistent_buffers_set"):
                        mod._non_persistent_buffers_set = set()

                self.load_state_dict(savedmodel.state_dict())
            except Exception as e:
                print("Loading failure. Check weights file:", self.weights_filename_local)
                raise (e)

            self.eval()

    def __repr__(self):
        if self.weights != None:
            return "XRV-DenseNet121-{}".format(self.weights)
        else:
            return "XRV-DenseNet"

    def features2(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        if self.return_logit:
            return features, out
        else:
            return out

    # ==================================GRADCAM on Intermediate Layers using KNN output======================================
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    def l1_norm(self, X_i, X_j):
        return (abs(X_i - X_j)).sum(-1)

    def l2_norm(self, X_i, X_j):
        return ((X_i - X_j) ** 2).sum(-1)

    def gradcam_forward(self, x, nbr_feats, nbr_labels, s):
        x = self.features[:4](x)
        x_feat = self.features[4].denselayer1(x)
        _ = x_feat.register_hook(self.activations_hook)
        # self.features.denseblock1.denselayer1.register_forward_hook(self.hook_feat_map)
        # self.feat_maps = []
        # _ = self.features(x)
        x_feat_1d = torch.reshape(x_feat, (x_feat.shape[0], -1))
        nr = 0
        dr = 0
        for nbr in nbr_feats[nbr_labels == 1]:
            nr = nr + torch.exp(-self.l1_norm(x_feat_1d, nbr) / s)
        for nbr in nbr_feats:
            dr = dr + torch.exp(-self.l1_norm(x_feat_1d, nbr) / s)
        out = nr / dr

        return out, x_feat  # probability value

    # ==================================end of GRADCAM code======================================

    def forward(self, x):
        if self.return_logit:
            raw_features, pooled_features = self.features2(x)
            dropout_features = F.dropout(pooled_features, p=self.drop_rate,
                                         training=True)  # trainging is always true as drop_rate!=0 means MCDropout
            out = self.classifier(dropout_features)
            # print('logits')
            return out, raw_features, pooled_features

        if hasattr(self, 'apply_sigmoid') and self.apply_sigmoid:  # False
            features = self.features2(x)
            features = F.dropout(features, p=self.drop_rate,
                                 training=True)  # trainging is always true as drop_rate!=0 means MCDropout
            out = self.classifier(features)
            out = torch.sigmoid(out)
            # print("Only sigmoid")
            return out

        elif hasattr(self, "op_threshs"):
            try:
                if (self.op_threshs == None):
                    features = self.features2(x)
                    features = F.dropout(features, p=self.drop_rate,
                                         training=True)  # trainging is always true as drop_rate!=0 means MCDropout
                    out = self.classifier(features)
                    # print('logits')
                    return out
                else:
                    features = self.features2(x)
                    features = F.dropout(features, p=self.drop_rate,
                                         training=True)  # trainging is always true as drop_rate!=0 means MCDropout
                    out = self.classifier(features)
                    out = torch.sigmoid(out)
                    out = op_norm(out, self.op_threshs)
                    # print("Sigmoid + op_threshs")
                    return out

            except:
                import pdb
                pdb.set_trace()
                out = torch.sigmoid(out)
                out = op_norm(out, self.op_threshs)
                # print("Sigmoid + op_threshs")
                return out
        else:
            # print('logits')
            features = self.features2(x)
            features = F.dropout(features, p=self.drop_rate,
                                 training=True)  # trainging is always true as drop_rate!=0 means MCDropout
            out = self.classifier(features)
            return out

    def intermediate_forward(self, x, layer_index):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        return out

    def feature_list(self, x):
        out_list = []
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out_list.append(out)
        features = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        features = F.dropout(features, p=self.drop_rate,
                             training=True)  # trainging is always true as drop_rate!=0 means MCDropout
        out = self.classifier(features)

        return out, out_list


def op_norm(outputs, op_threshs):
    """normalize outputs according to operating points for a given model.
    Args:
        outputs: outputs of self.classifier(). torch.Size(batch_size, num_tasks)
        op_threshs_arr: torch.Size(batch_size, num_tasks) with self.op_threshs expanded.
    Returns:
        outputs_new: normalized outputs, torch.Size(batch_size, num_tasks)
    """
    # expand to batch size so we can do parallel comp
    op_threshs = op_threshs.expand(outputs.shape[0], -1)

    # initial values will be 0
    outputs_new = torch.zeros(outputs.shape, device=outputs.device)

    # only select non-nan elements otherwise the gradient breaks
    mask_leq = (outputs < op_threshs) & ~torch.isnan(op_threshs)
    mask_gt = ~(outputs < op_threshs) & ~torch.isnan(op_threshs)

    # scale outputs less than thresh
    outputs_new[mask_leq] = outputs[mask_leq] / (op_threshs[mask_leq] * 2)
    # scale outputs greater than thresh
    outputs_new[mask_gt] = 1.0 - ((1.0 - outputs[mask_gt]) / ((1 - op_threshs[mask_gt]) * 2))

    return outputs_new


def get_densenet_params(arch):
    assert 'dense' in arch
    if arch == 'densenet161':
        ret = dict(growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96)
    elif arch == 'densenet169':
        ret = dict(growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64)
    elif arch == 'densenet201':
        ret = dict(growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64)
    else:
        # default configuration: densenet121
        ret = dict(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64)
    return ret


class customDenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Modified from torchvision to have a variable number of input channels

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, train_embs_path, pd, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4,
                 drop_rate=0, num_classes=18, in_channels=1, weights=None, op_threshs=None, progress=True,
                 apply_sigmoid=False, return_logit=False):

        super(customDenseNet, self).__init__()
        self.drop_rate = drop_rate
        self.apply_sigmoid = apply_sigmoid
        self.return_logit = return_logit
        self.weights = weights

        # required for custom forward pass:
        self.train_embs_path = train_embs_path
        self.pd = pd
        self.train_emb_idx = int(pd / 4 - 1)
        self.test_feat_1d = None
        self.nbr_feats = None
        self.nbr_labels = None
        self.s = None

        # load train embeddings for soft-KNN
        with open(self.train_embs_path, 'rb') as handle:
            self.info_dict = pickle.load(handle)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        # needs to be register_buffer here so it will go to cuda/cpu easily
        self.register_buffer('op_threshs', op_threshs)

        if self.weights != None:
            if '.pt' in self.weights:
                self.weights_filename_local = self.weights
                print("weights_filename_local: ", self.weights_filename_local)
                print("........")

            try:
                savedmodel = torch.load(self.weights_filename_local, map_location='cpu')
                print("model loaded", self.weights_filename_local)
                # patch to load old models https://github.com/pytorch/pytorch/issues/42242
                for mod in savedmodel.modules():
                    if not hasattr(mod, "_non_persistent_buffers_set"):
                        mod._non_persistent_buffers_set = set()

                self.load_state_dict(savedmodel.state_dict())
            except Exception as e:
                print("Loading failure. Check weights file:", weights_filename_local)
                raise (e)

            self.eval()

    def __repr__(self):
        if self.weights != None:
            return "XRV-DenseNet121-{}".format(self.weights)
        else:
            return "XRV-DenseNet"

    def l1_norm(self, X_i, X_j):
        return (abs(X_i - X_j)).sum(-1)

    def l2_norm(self, X_i, X_j):
        return ((X_i - X_j) ** 2).sum(-1)

    def get_nbr_info(self, x):
        # get the nearest K neighbours for test img
        K = 29

        if self.pd / 4 >= 1 and self.pd / 4 <= 3:
            x = self.features[:4](x)
            denselayer = int(1 + 2 * (self.pd / 4 - 1))
            for dl_id in range(denselayer):
                x = self.features.denseblock1[dl_id](x)

        elif self.pd / 4 >= 4 and self.pd / 4 <= 9:
            x = self.features[:6](x)
            denselayer = int(1 + 2 * (self.pd / 4 - 4))
            for dl_id in range(denselayer):
                x = self.features.denseblock2[dl_id](x)

        elif self.pd / 4 >= 10 and self.pd / 4 <= 21:
            x = self.features[:8](x)
            denselayer = int(1 + 2 * (self.pd / 4 - 10))
            for dl_id in range(denselayer):
                x = self.features.denseblock3[dl_id](x)

        else:
            x = self.features[:10](x)
            denselayer = int(1 + 2 * (self.pd / 4 - 22))
            for dl_id in range(denselayer):
                x = self.features.denseblock4[dl_id](x)

        test_feat = x
        self.test_feat_1d = torch.reshape(test_feat, (test_feat.shape[0], -1))

        info_dict = self.info_dict
        X_i = self.test_feat_1d.unsqueeze(1)  # (10000, 1, 784) test set
        X_j = info_dict['feats'][self.train_emb_idx].unsqueeze(0)  # (1, 60000, 784) train set
        D_ij = (abs(X_i - X_j)).sum(-1)
        ind_knn = torch.topk(-D_ij, K, dim=1)  # Samples <-> Dataset, (N_test, K)
        lab_knn = info_dict['labels'][ind_knn[1]]  # (N_test, K) array of integers in [0,9]

        # free GPU memory
        del info_dict
        torch.cuda.empty_cache()

        # feed KNN nbr info of the test img to model
        nbr_inds = np.array(ind_knn[1][0].detach().cpu())
        self.nbr_feats = X_j[0, nbr_inds, :].to('cuda')
        self.nbr_labels = lab_knn.squeeze()
        self.s = float(torch.median(-ind_knn[0]).detach().cpu())  # for the median trick

    def forward(self, imgs):
        out = torch.empty((0, 2)).to('cuda')
        for x in imgs:
            x = x.unsqueeze(0)
            self.get_nbr_info(x)
            x_feat_1d = self.test_feat_1d
            nr = 0
            dr = 0
            pred_cls = int(self.nbr_labels.mode()[0])  # predicted class based, argmax of the KNN
            for nbr in self.nbr_feats[self.nbr_labels == pred_cls]:
                nr = nr + torch.exp(-self.l1_norm(x_feat_1d, nbr) / self.s)
            for nbr in self.nbr_feats:
                dr = dr + torch.exp(-self.l1_norm(x_feat_1d, nbr) / self.s)
            frac = torch.cat((nr / dr, 1 - nr / dr)).unsqueeze(0)
            out = torch.cat((out, frac))

        return out
