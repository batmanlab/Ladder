import torch
from torch import nn

from .image_classifier import LinearClassifier
from .. import build_model


class MammoClassifier(nn.Module):
    def __init__(self, cfg, clf_checkpoint, n_class):
        super(MammoClassifier, self).__init__()
        self.clf = build_model(model_config=cfg["classifier"], loss_config=cfg["loss"], tokenizer=None)
        self.ckpt = torch.load(clf_checkpoint, map_location="cpu")
        image_encoder_weights = {}
        for k in self.ckpt["model"].keys():
            image_encoder_weights[k] = self.ckpt["model"][k]

        image_encoder_weights.pop('_fc.weight')
        image_encoder_weights.pop('_fc.bias')
        ret = self.clf.load_state_dict(image_encoder_weights, strict=True)
        print(ret)

        clf_ft_dim = 0
        if cfg["classifier"]["clf_arch"] == "tf_efficientnet_b5_ns-detect":
            clf_ft_dim = 2048

        self.classifier = LinearClassifier(feature_dim=clf_ft_dim, num_class=n_class)
        image_clf_weights = {}
        for k in self.ckpt["model"].keys():
            if k == "_fc.weight":
                image_clf_weights["classification_head.weight"] = self.ckpt["model"][k]
            elif k == "_fc.bias":
                image_clf_weights["classification_head.bias"] = self.ckpt["model"][k]
        ret = self.classifier.load_state_dict(image_clf_weights, strict=True)
        print(ret)

    def get_predictions_from_chkpt(self):
        return self.ckpt["predictions"]

    def forward(self, images):
        features = self.clf(images)
        logits = self.classifier(features)
        return features, logits
