import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
        # init.constant_(m.bias.data, 0.0)


class ResNet(nn.Module):
    def __init__(
            self, pre_trained=True, n_class=200, model_choice="resnet50", weights=None, return_logit=False,
            layer="layer4"
    ):
        super(ResNet, self).__init__()
        self.feature_store = {}
        self.n_class = n_class
        self.base_model = self._model_choice(pre_trained, model_choice)
        feat_dim = self.base_model.fc.weight.shape[1]

        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base_model.fc = nn.Linear(in_features=feat_dim, out_features=n_class)
        self.base_model.fc.apply(weight_init_kaiming)
        self.return_logit = return_logit
        self.feature_store = {}
        self.weights = weights
        if self.weights is not None:
            self.weights_filename_local = self.weights
            print("weights_filename_local: ", self.weights_filename_local)

            try:
                print(self.weights_filename_local)
                savedmodel = torch.load(self.weights_filename_local, map_location='cpu')
                # self.base_model.load_state_dict(torch.load(weights))
                ret = self.load_state_dict(savedmodel.state_dict())
                print(ret)
            except Exception as e:
                print("Loading failure. Check weights file:", self.weights_filename_local)
                raise (e)

        self.base_model.layer4.register_forward_hook(self.save_activation("layer4"))
        self.base_model.avgpool.register_forward_hook(self.save_activation("adaptive_avg_pool"))

    def save_activation(self, layer):
        def hook(module, input, output):
            self.feature_store[layer] = output

        return hook

    def forward(self, x):
        N = x.size(0)
        # assert x.size() == (N, 3, 448, 448)
        x = self.base_model(x)
        assert x.size() == (N, self.n_class)

        if self.return_logit:
            raw_features = self.feature_store["layer4"]
            pooled_features = self.feature_store["adaptive_avg_pool"]
            pooled_features = torch.squeeze(pooled_features)
            return x, raw_features, pooled_features
        else:
            return x

    @staticmethod
    def _model_choice(pre_trained, model_choice):
        if model_choice.lower() == "resnet50":
            return models.resnet50(pretrained=pre_trained)
        elif model_choice.lower() == "resnet101":
            return models.resnet101(pretrained=pre_trained)
        elif model_choice.lower() == "resnet152":
            return models.resnet152(pretrained=pre_trained)