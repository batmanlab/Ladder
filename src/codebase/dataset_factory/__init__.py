from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from albumentations import *
from torch.utils.data import DataLoader

from dataset_factory import datasets
from dataset_factory.UrbanCars import UrbanCars
from dataset_factory.custom_datasets import SubsetDataset
from dataset_factory.datasets import RSNADataset, Dataset_NIH, collate_NIH, EmbedDataset
from utils import get_hparams


def get_natural_images_dataloaders(args):
    hparams = get_hparams(args.dataset, args.classifier)
    if args.dataset in vars(datasets):
        train_dataset = vars(datasets)[args.dataset](args.data_dir, 'tr', hparams, train_attr="yes")
        val_dataset = vars(datasets)[args.dataset](args.data_dir, 'va', hparams)
        test_dataset = vars(datasets)[args.dataset](args.data_dir, 'te', hparams)
        print(f"Dataset sizes => train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
        num_workers = train_dataset.N_WORKERS

        if hparams['group_balanced']:
            train_weights = np.asarray(train_dataset.weights_g)
            train_weights /= np.sum(train_weights)
        else:
            train_weights = None

        train_loader = DataLoader(
            train_dataset, batch_size=hparams['batch_size'], shuffle=args.shuffle, num_workers=num_workers,
            pin_memory=True, drop_last=True
        )

        valid_loader = DataLoader(
            val_dataset, batch_size=max(128, hparams['batch_size'] * 2), shuffle=args.shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=max(128, hparams['batch_size'] * 2), shuffle=args.shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
        return {
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "test_loader": test_loader
        }


def get_rsna_dataloaders(args):
    train_tfms = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
        ElasticTransform(alpha=10, sigma=15)
    ], p=1.0)
    val_tfms = None

    data_dir = Path(args.data_dir)
    train_df = None
    valid_df = None
    test_df = None

    if args.dataset.lower() == "rsna":
        df = pd.read_csv(data_dir / "rsna_w_upmc_concepts_breast_clip.csv")
        mapping = {'L': 0, 'R': 1}
        df['laterality'] = df['laterality'].map(mapping)
        train_df = df[(df['fold'] == 1) | (df['fold'] == 2)].reset_index(drop=True)
        valid_df = df[df['fold'] == 3].reset_index(drop=True)
        test_df = df[df['fold'] == 0].reset_index(drop=True)
    elif args.dataset.lower() == "vindr":
        df = pd.read_csv(data_dir / "vindr_detection_v1_folds_cancer.csv")
        mapping = {'L': 0, 'R': 1}
        df['laterality'] = df['laterality'].map(mapping)
        train_df = df[df["split_new"] == "train"].reset_index(drop=True)
        valid_df = df[df["split_new"] == "val"].reset_index(drop=True)
        test_df = df[df["split_new"] == "test"].reset_index(drop=True)

    train_dataset = RSNADataset(
        train_df, data_dir, train_tfms, mean=0.3089279, std=0.25053555408335154, dataset=args.dataset.lower())
    valid_dataset = RSNADataset(
        valid_df, data_dir, val_tfms, mean=0.3089279, std=0.25053555408335154, dataset=args.dataset.lower())
    test_dataset = RSNADataset(
        test_df, data_dir, val_tfms, mean=0.3089279, std=0.25053555408335154, dataset=args.dataset.lower())
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
        drop_last=False
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )
    print("\n ==========================> [Shapes] Dataset <==========================")
    print("train: ", len(train_dataset), "val: ", len(valid_dataset), "test: ", len(test_dataset))
    print("\n ==========================> [Shapes] Dataloaders <==========================")
    print("train: ", len(train_loader), "val: ", len(valid_loader), "test: ", len(test_loader))
    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader
    }

def get_nih_dataloaders(args):
    column_name_split = "val_train_split"
    tfms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_file = Path(args.data_dir) / "nih_processed_v2.csv"
    df = pd.read_csv(data_file)
    dataset = Dataset_NIH(df=df, class_names=["Pneumothorax"], transform=tfms, seed=args.seed)
    try:
        df_train = df.loc[(df[column_name_split] == 1)]
        train_inds = np.asarray(df_train.index)
        df_test = df.loc[(df[column_name_split] == 0)]
        test_inds = np.asarray(df_test.index)
        df_val = df.loc[(df[column_name_split] == 2)]
        val_inds = np.asarray(df_val.index)
        print("train: ", train_inds.shape, "test: ", test_inds.shape, "val: ", val_inds.shape)
    except:
        print(
            "The data_file doesn't have a train column, "
            "hence we will randomly split the entire dataset to have 15% samples as validation set.")
        train_inds = np.empty([])
        test_inds = np.empty([])
        val_inds = np.empty([])

    train_dataset = SubsetDataset(dataset, train_inds)
    valid_dataset = SubsetDataset(dataset, val_inds)
    test_dataset = SubsetDataset(dataset, test_inds)
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=args.shuffle, num_workers=4, pin_memory=True, collate_fn=collate_NIH
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=128, shuffle=args.shuffle, num_workers=4, pin_memory=True, collate_fn=collate_NIH
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=args.shuffle, num_workers=4, pin_memory=True, collate_fn=collate_NIH
    )

    print("\n ==========================> [Shapes] NIH Dataset <==========================")
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    print("\n ==========================> [Shapes] NIH Dataloaders <==========================")
    print(len(train_loader), len(valid_loader), len(test_loader))

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader
    }


def create_dataloaders(args):
    if args.dataset.lower() == "waterbirds" or args.dataset.lower() == "celeba" or args.dataset.lower() == "metashift":
        return get_natural_images_dataloaders(args)
    elif args.dataset.lower() == "nih":
        return get_nih_dataloaders(args)
    elif args.dataset.lower() == "rsna" or args.dataset.lower() == "vindr":
        return get_rsna_dataloaders(args)