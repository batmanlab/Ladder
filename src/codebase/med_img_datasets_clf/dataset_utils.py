import numpy as np
import pandas as pd
import pickle
import torch
import torchvision
import torchvision.transforms
from albumentations import *
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset_CXR import Dataset_NIH, collate_NIH
from .dataset_mammo import MammoDataset, collator_mammo_dataset_w_concepts

from .custom_datasets import SubsetDataset


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
    if (args.dataset == "NIH" or args.dataset.lower() == "mimic") and (
            args.arch.lower() == "resnet50" or args.arch.lower() == "resnet101" or args.arch.lower() == "resnet152"
    ):
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(args.img_size),
            torchvision.transforms.CenterCrop(args.img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif (args.dataset.lower() == "rsna" or args.dataset.lower() == "vindr") and (
            args.model_type.lower() == "classifier"
    ):
        return Compose([
            HorizontalFlip(),
            VerticalFlip(),
            Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
            ElasticTransform(alpha=args.alpha, sigma=args.sigma)
        ], p=args.p)


def get_dataloader_CXR(args, is_train_mode, is_classifier):
    transforms = get_transforms(args)
    is_rgb = False if args.arch == "densenet121" else True
    dataset = Dataset_NIH(
        csvpath=args.data_file, class_names=args.class_names, transform=transforms, seed=args.seed,
        is_train_mode=is_train_mode, is_rgb=is_rgb, is_classifier=is_classifier, dataset_name=args.dataset
    )
    df = pd.read_csv(args.data_file)
    if args.dataset.lower() == "nih":
        try:
            df_train = df.loc[(df[args.column_name_split] == 1)]
            train_inds = np.asarray(df_train.index)
            df_test = df.loc[(df[args.column_name_split] == 2)]
            test_inds = np.asarray(df_test.index)
            print("train: ", train_inds.shape, "test: ", test_inds.shape)
        except:
            print(
                "The data_file doesn't have a train column, "
                "hence we will randomly split the entire dataset to have 15% samples as validation set.")
            train_inds = np.empty([])
            test_inds = np.empty([])
    else:  # MIMIC
        try:
            train_inds = np.asarray(df.index)
            test_inds = np.asarray(df.index)
            print("train: ", train_inds.shape, "test: ", test_inds.shape)
        except:
            print(
                "The data_file doesn't have a train column, "
                "hence we will randomly split the entire dataset to have 15% samples as validation set.")
            train_inds = np.empty([])
            test_inds = np.empty([])

    train_dataset = SubsetDataset(dataset, train_inds)
    valid_dataset = SubsetDataset(dataset, test_inds)
    # Dataloader
    if is_classifier:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_NIH
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_NIH
        )

    return train_loader, valid_loader


def get_dataloader_mammo(args):
    train_tfm = None
    val_tfm = None
    if args.arch.lower() == "swin_tiny_custom_norm" or args.arch.lower() == "swin_base_custom_norm":
        color_jitter_transform = torchvision.transforms.ColorJitter(
            brightness=0.1,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        normalize_transform = torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        train_tfm = torchvision.transforms.Compose([
            color_jitter_transform,
            torchvision.transforms.ToTensor(),
            normalize_transform
        ])
        val_tfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize_transform
        ])
    elif args.arch.lower() == "swin_tiny_custom" or args.arch.lower() == "swin_base_custom":
        train_tfm = Compose([
            ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1, p=1),
        ])
    else:
        train_tfm = get_transforms(args)

    train_dataset = MammoDataset(args=args, df=args.train_folds, transform=train_tfm)
    valid_dataset = MammoDataset(args=args, df=args.valid_folds, transform=val_tfm)

    if args.balanced_dataloader == "y":
        weight_path = args.output_path / f"random_sampler_weights_fold{str(args.cur_fold)}.pkl"
        if weight_path.exists():
            weights = pickle.load(open(weight_path, "rb"))
        else:
            weight_for_positive_class = args.sampler_weights[f"fold{str(args.cur_fold)}"]["pos_wt"]
            weight_for_negative_class = args.sampler_weights[f"fold{str(args.cur_fold)}"]["neg_wt"]
            args.train_folds["weights_random_sampler"] = args.train_folds.apply(
                lambda row: weight_for_positive_class if row["cancer"] == 1 else weight_for_negative_class, axis=1
            )
            weights = args.train_folds["weights_random_sampler"].values
            pickle.dump(weights, open(args.output_path / f"random_sampler_weights_fold{args.cur_fold}.pkl", "wb"))

        weights = weights.tolist()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
            drop_last=True, collate_fn=collator_mammo_dataset_w_concepts, sampler=sampler
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
            drop_last=True, collate_fn=collator_mammo_dataset_w_concepts
        )

    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        drop_last=False, collate_fn=collator_mammo_dataset_w_concepts
    )
    print("---------------------------------------")
    print("Train Loader: ", len(train_loader))
    print("Valid Loader: ", len(valid_loader))
    print("Train dataset", len(train_dataset))
    print("Valid dataset", len(valid_dataset))
    print("---------------------------------------")
    return train_loader, valid_loader


def get_dataset(args, is_train_mode=True, is_classifier=True, train=True):
    if args.dataset == "NIH" or args.dataset.lower() == "mimic":
        return get_dataloader_CXR(args, is_train_mode=is_train_mode, is_classifier=is_classifier)
    elif args.dataset.lower() == "rsna" and args.model_type.lower() == "classifier":
        return get_dataloader_mammo(args)
