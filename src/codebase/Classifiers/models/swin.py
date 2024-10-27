import torch
from torch import nn

from breastclip.model.modules import load_image_encoder, LinearClassifier


class SwinClassifier(nn.Module):
    def __init__(self, args, ckpt, n_class):
        super(SwinClassifier, self).__init__()

        print(ckpt["config"]["model"]["image_encoder"])
        self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])
        image_encoder_weights = {}
        for k in ckpt["model"].keys():
            if k.startswith("image_encoder."):
                image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
        self.image_encoder.load_state_dict(image_encoder_weights, strict=True)
        self.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]

        if (
                args.arch.lower() == "upmc_breast_clip_b5_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_b2_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_det_b2_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_resnet101_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_resnet152_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_swin_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_swin_tiny_512_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_swin_base_512_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_swin_large_512_period_n_lp" or
                args.arch.lower() == "upmc_rsna_breast_clip_b5_period_n_lp" or
                args.arch.lower() == "upmc_rsna_breast_clip_swin_period_n_lp" or
                args.arch.lower() == "upmc_rsna_breast_clip_swin_tiny_512_period_n_lp" or
                args.arch.lower() == "upmc_rsna_breast_clip_swin_base_512_period_n_lp" or
                args.arch.lower() == "upmc_rsna_breast_clip_swin_large_512_period_n_lp"):
            print("freezing image encoder to not be trained")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.classifier = LinearClassifier(feature_dim=self.image_encoder.out_dim, num_class=n_class)
        self.raw_features = None
        self.pool_features = None

    def get_image_encoder_type(self):
        return self.image_encoder_type

    def encode_image(self, image):
        image_features, raw_features = self.image_encoder(image)
        self.raw_features = raw_features
        self.pool_features = image_features

        if self.image_encoder_type == "cnn":
            return image_features
        else:
            # get [CLS] token for global representation (only for vision transformer)
            global_features = image_features[:, 0]
            return global_features

    def forward(self, images):
        if self.image_encoder_type.lower() == "swin":
            images = images.squeeze(1).permute(0, 3, 1, 2)
        # get image features and predict
        image_feature = self.encode_image(images)
        logits = self.classifier(image_feature)
        return logits
