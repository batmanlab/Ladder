import numpy as np
import pandas as pd
import pickle
import torch
import torchvision
import torchvision.transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
# from med_img_datasets_clf.custom_datasets import SubsetDataset
# from albumentations import *
# from imgaug import augmenters as iaa
# from med_img_datasets_clf.dataset_NIH import Dataset_NIH, collate_NIH
# from med_img_datasets_clf.dataset_concepts import MammoDataset_concept, MammoDataset_concept_detection, \
#     collater_for_concept_detection, MammoDataset, collator_mammo_dataset_w_concepts, \
#     MammoDataset_concept_set, MammoDataset_Mapper
# from med_img_datasets_clf.dataset_domain_classification import Dataset_domain_classifer, collator_domain_classifier


class center_crop(object):
    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img):
        return self.crop_center(img)


class normalize(object):
    def normalize_(self, img, maxval=255):
        img = (img) / (maxval)
        return img

    def __call__(self, img):
        return self.normalize_(img)


def get_transforms(args):
    if args.dataset == "NIH" and args.arch == "densenet121":
        return torchvision.transforms.Compose([
            # torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((args.img_size, args.img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(center_crop()),
            torchvision.transforms.Lambda(normalize())
        ])
    elif args.dataset == "NIH" and (
            args.arch.lower() == "resnet50" or args.arch.lower() == "resnet101" or args.arch.lower() == "resnet152"
    ):
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(args.img_size),
            torchvision.transforms.CenterCrop(args.img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif (
            args.dataset.lower() == "upmc-rsna" or args.dataset.lower() == "upmc"
            or args.dataset.lower() == "rsna" or args.dataset.lower() == "vindr"
    ) and (
            args.model_type.lower() == "domain-classifier" or args.model_type.lower() == "concept-classifier"
            or args.model_type.lower() == "classifier"
    ):
        if args.img_size[0] == 1520 and args.img_size[1] == 912:
            return Compose([
                HorizontalFlip(),
                VerticalFlip(),
                # RandomRotate90(),
                #         ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.8,
                #                          border_mode=cv2.BORDER_REFLECT)
                # OneOf([Affine(rotate=20, translate_percent=0.1, scale=[0.8,1.2], shear=20)])
                Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
                ElasticTransform(alpha=args.alpha, sigma=args.sigma)
            ], p=args.p)
        else:
            return Compose([
                Resize(width=int(args.img_size[0]), height=int(args.img_size[1])),
                HorizontalFlip(),
                VerticalFlip(),
                Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
                ElasticTransform(alpha=args.alpha, sigma=args.sigma)
            ], p=args.p
            )
    elif (args.dataset.lower() == "vindr" or args.dataset.lower() == "upmc") and (
            args.model_type.lower() == "concept-detector"
    ):
        if args.image_net_transform:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])

        train_affine_trans = iaa.Sequential([
            iaa.Resize({'height': args.resize, 'width': args.resize}),
            iaa.Fliplr(0.5),  # HorizontalFlip
            iaa.Flipud(0.5),  # VerticalFlip
            iaa.Affine(rotate=(-20, 20), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, scale=(0.8, 1.2),
                       shear=(-20, 20)),
            iaa.ElasticTransformation(alpha=args.alpha, sigma=args.sigma)
        ])

        test_affine_trans = iaa.Sequential([
            iaa.Resize({'height': args.resize, 'width': args.resize}),
            iaa.CropToFixedSize(width=args.resize, height=args.resize)  # Adjust width and height as needed
        ])

        return transform, train_affine_trans, test_affine_trans


# def get_dataloader_NIH(args, is_train_mode, is_classifier):
#     transforms = get_transforms(args)
#     is_rgb = False if args.arch == "densenet121" else True
#     dataset = Dataset_NIH(
#         csvpath=args.data_file, class_names=args.class_names, transform=transforms, seed=args.seed,
#         is_train_mode=is_train_mode, is_rgb=is_rgb, is_classifier=is_classifier
#     )
#     df = pd.read_csv(args.data_file)
#     try:
#         df_train = df.loc[(df[args.column_name_split] == 1)]
#         train_inds = np.asarray(df_train.index)
#         df_test = df.loc[(df[args.column_name_split] == 0)]
#         test_inds = np.asarray(df_test.index)
#         print("train: ", train_inds.shape, "test: ", test_inds.shape)
#     except:
#         print(
#             "The data_file doesn't have a train column, "
#             "hence we will randomly split the entire dataset to have 15% samples as validation set.")
#         train_inds = np.empty([])
#         test_inds = np.empty([])
#     train_dataset = SubsetDataset(dataset, train_inds)
#     valid_dataset = SubsetDataset(dataset, test_inds)
#     # Dataloader
#     if is_classifier:
#         train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=args.batch_size, shuffle=True,
#             num_workers=args.num_workers, pin_memory=True
#         )
#         valid_loader = torch.utils.data.DataLoader(
#             valid_dataset, batch_size=args.batch_size, shuffle=False,
#             num_workers=args.num_workers, pin_memory=True,
#         )
#     else:
#         train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=args.batch_size, shuffle=False,
#             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_NIH
#         )
#         valid_loader = torch.utils.data.DataLoader(
#             valid_dataset, batch_size=args.batch_size, shuffle=False,
#             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_NIH
#         )

#     return train_loader, valid_loader


# def get_dataloader_detection_based_domain_classifier(args):
#     transform = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#     ])

#     train_affine_trans = iaa.Sequential([
#         iaa.Resize({'height': args.resize, 'width': args.resize}),
#         iaa.Fliplr(0.5),  # HorizontalFlip
#         iaa.Flipud(0.5),  # VerticalFlip
#         iaa.Affine(rotate=(-20, 20), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, scale=(0.8, 1.2),
#                    shear=(-20, 20)),
#         iaa.ElasticTransformation(alpha=args.alpha, sigma=args.sigma)
#     ])

#     test_affine_trans = iaa.Sequential([
#         iaa.Resize({'height': args.resize, 'width': args.resize}),
#         iaa.CropToFixedSize(width=args.resize, height=args.resize)  # Adjust width and height as needed
#     ])
#     valid_dataset = Dataset_domain_classifer(
#         args=args, df=args.valid_folds, iaa_transform=test_affine_trans, transform=transform
#     )
#     valid_loader = DataLoader(
#         valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
#         drop_last=False, collate_fn=collator_domain_classifier
#     )

#     train_dataset = Dataset_domain_classifer(
#         args=args, df=args.train_folds, iaa_transform=train_affine_trans, transform=transform
#     )
#     train_loader = DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
#         drop_last=True, collate_fn=collator_domain_classifier
#     )

#     return train_loader, valid_loader


# def get_dataloader_classification_based_domain_classifier(args):
#     train_dataset = Dataset_domain_classifer(args=args, df=args.train_folds, transform=get_transforms(args))
#     valid_dataset = Dataset_domain_classifer(args=args, df=args.valid_folds)
#     train_loader = DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
#         drop_last=True, collate_fn=collator_domain_classifier
#     )
#     valid_loader = DataLoader(
#         valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
#         drop_last=False, collate_fn=collator_domain_classifier
#     )

#     return train_loader, valid_loader


# def get_dataloader_concept_classifier_set(args, train=True):
#     valid_dataset = MammoDataset_concept_set(args=args, df=args.valid_folds)
#     valid_loader = DataLoader(
#         valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
#         drop_last=False
#     )
#     if train:
#         train_dataset = MammoDataset_concept_set(args=args, df=args.train_folds, tfms=get_transforms(args))
#         train_loader = DataLoader(
#             train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
#             drop_last=True
#         )
#         return train_loader, valid_loader
#     else:
#         return valid_loader


# def get_dataloader_concept_classifier(args, train=True):
#     valid_dataset = MammoDataset_concept(args=args, df=args.valid_folds, dataset=args.dataset)
#     valid_loader = DataLoader(
#         valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
#         drop_last=False,
#     )
#     if train:
#         train_dataset = MammoDataset_concept(
#             args=args, df=args.train_folds, dataset=args.dataset, transform=get_transforms(args)
#         )
#         train_loader = DataLoader(
#             train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
#             drop_last=True,
#         )
#         return train_loader, valid_loader
#     else:
#         return valid_loader


# def get_dataloader_concept_detector(args, train=True):
#     transform, train_affine_trans, test_affine_trans = get_transforms(args)
#     valid_dataset = MammoDataset_concept_detection(
#         args=args, df=args.valid_folds, iaa_transform=test_affine_trans, transform=transform
#     )
#     valid_loader = DataLoader(
#         valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
#         drop_last=False, collate_fn=collater_for_concept_detection
#     )
#     if train:
#         train_dataset = MammoDataset_concept_detection(
#             args=args, df=args.train_folds, iaa_transform=train_affine_trans, transform=transform
#         )
#         train_loader = DataLoader(
#             train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
#             drop_last=True, collate_fn=collater_for_concept_detection
#         )
#         return train_loader, valid_loader, valid_dataset
#     else:
#         return valid_loader, valid_dataset


# def get_dataloader_Vindr(args):
#     train_dataset = MammoDataset_Mapper(args=args, df=args.train_folds, transform=get_transforms(args))
#     valid_dataset = MammoDataset_Mapper(args=args, df=args.valid_folds)

#     if args.balanced_dataloader == "y":
#         weight_path = args.output_path / f"random_sampler_weights_fold{str(args.cur_fold)}.pkl"
#         if weight_path.exists():
#             weights = pickle.load(open(weight_path, "rb"))
#         else:
#             weight_for_positive_class = args.sampler_weights[f"fold{str(args.cur_fold)}"]["pos_wt"]
#             weight_for_negative_class = args.sampler_weights[f"fold{str(args.cur_fold)}"]["neg_wt"]
#             args.train_folds["weights_random_sampler"] = args.train_folds.apply(
#                 lambda row: weight_for_positive_class if row["cancer"] == 1 else weight_for_negative_class, axis=1
#             )
#             weights = args.train_folds["weights_random_sampler"].values
#             pickle.dump(weights, open(args.output_path / f"random_sampler_weights_fold{args.cur_fold}.pkl", "wb"))

#         weights = weights.tolist()
#         sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
#         train_loader = DataLoader(
#             train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
#             drop_last=True, sampler=sampler
#         )
#     else:
#         train_loader = DataLoader(
#             train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
#             drop_last=True,
#         )

#     valid_loader = DataLoader(
#         valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
#         drop_last=False
#     )

#     return train_loader, valid_loader


# def get_dataloader_RSNA(args):
#     train_tfm = None
#     val_tfm = None
#     if args.arch.lower() == "swin_tiny_custom_norm" or args.arch.lower() == "swin_base_custom_norm":
#         color_jitter_transform = torchvision.transforms.ColorJitter(
#             brightness=0.1,
#             contrast=0.2,
#             saturation=0.2,
#             hue=0.1
#         )
#         normalize_transform = torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
#         train_tfm = torchvision.transforms.Compose([
#             color_jitter_transform,
#             torchvision.transforms.ToTensor(),
#             normalize_transform
#         ])
#         val_tfm = torchvision.transforms.Compose([
#             torchvision.transforms.ToTensor(),
#             normalize_transform
#         ])
#     elif args.arch.lower() == "swin_tiny_custom" or args.arch.lower() == "swin_base_custom":
#         train_tfm = Compose([
#             ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1, p=1),
#         ])
#     else:
#         train_tfm = get_transforms(args)

#     train_dataset = MammoDataset(args=args, df=args.train_folds, transform=train_tfm)
#     valid_dataset = MammoDataset(args=args, df=args.valid_folds, transform=val_tfm)

#     if args.balanced_dataloader == "y":
#         weight_path = args.output_path / f"random_sampler_weights_fold{str(args.cur_fold)}.pkl"
#         if weight_path.exists():
#             weights = pickle.load(open(weight_path, "rb"))
#         else:
#             weight_for_positive_class = args.sampler_weights[f"fold{str(args.cur_fold)}"]["pos_wt"]
#             weight_for_negative_class = args.sampler_weights[f"fold{str(args.cur_fold)}"]["neg_wt"]
#             args.train_folds["weights_random_sampler"] = args.train_folds.apply(
#                 lambda row: weight_for_positive_class if row["cancer"] == 1 else weight_for_negative_class, axis=1
#             )
#             weights = args.train_folds["weights_random_sampler"].values
#             pickle.dump(weights, open(args.output_path / f"random_sampler_weights_fold{args.cur_fold}.pkl", "wb"))

#         weights = weights.tolist()
#         sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
#         train_loader = DataLoader(
#             train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
#             drop_last=True, collate_fn=collator_mammo_dataset_w_concepts, sampler=sampler
#         )
#     else:
#         train_loader = DataLoader(
#             train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
#             drop_last=True, collate_fn=collator_mammo_dataset_w_concepts
#         )

#     valid_loader = DataLoader(
#         valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
#         drop_last=False, collate_fn=collator_mammo_dataset_w_concepts
#     )

#     return train_loader, valid_loader


# def get_dataset(args, is_train_mode=True, is_classifier=True, train=True):
#     if args.dataset == "NIH":
#         return get_dataloader_NIH(args, is_train_mode=is_train_mode, is_classifier=is_classifier)
#     elif args.dataset.lower() == "rsna" and args.model_type.lower() == "classifier":
#         return get_dataloader_RSNA(args)
#     elif (
#             args.dataset.lower() == "upmc-rsna" or args.dataset.lower() == "vindr-rsna"
#     ) and args.model_type.lower() == "domain-classifier" and args.concept_model_type.lower() == "classification":
#         return get_dataloader_classification_based_domain_classifier(args)
#     elif (
#             args.dataset.lower() == "upmc-rsna" or args.dataset.lower() == "vindr-rsna"
#     ) and args.model_type.lower() == "domain-classifier" and args.concept_model_type.lower() == "detection":
#         return get_dataloader_detection_based_domain_classifier(args)
#     elif (
#             args.dataset.lower() == "upmc" or args.dataset.lower() == "vindr" or args.dataset.lower() == "rsna"
#     ) and args.model_type.lower() == 'concept-classifier':
#         return get_dataloader_concept_classifier(args, train=train)
#     elif (
#             args.dataset.lower() == "upmc" or args.dataset.lower() == "vindr" or args.dataset.lower() == "rsna"
#     ) and args.model_type.lower() == 'concept-classifier-set':
#         return get_dataloader_concept_classifier_set(args, train=train)
#     elif (
#             args.dataset.lower() == "upmc" or args.dataset.lower() == "vindr"
#     ) and args.model_type.lower() == 'concept-detector':
#         return get_dataloader_concept_detector(args, train=train)
