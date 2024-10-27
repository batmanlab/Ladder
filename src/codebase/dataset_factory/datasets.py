import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from transformers import BertTokenizer, AutoTokenizer, DistilBertTokenizer, GPT2Tokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Synthetic dataset
    "CMNIST",
    # Current subpop datasets
    "CUB_200_2011_3D_toy_shape",
    "Waterbirds",
    "CelebA",
    "CivilCommentsFine",  # "CivilComments"
    "MultiNLI",
    "MetaShift",
    "ImagenetBG",
    "NICOpp",
    "MIMICNoFinding",
    "MIMICNotes",
    "CXRMultisite",
    "CheXpertNoFinding",
    "Living17",
    "Entity13",
    "Entity30",
    "Nonliving26"
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class SubpopDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    INPUT_SHAPE = None  # Subclasses should override
    SPLITS = {  # Default, subclasses may override
        'tr': 0,
        'va': 1,
        'te': 2
    }
    EVAL_SPLITS = ['te']  # Default, subclasses may override

    def __init__(self, root, split, metadata, transform, train_attr='yes', subsample_type=None, duplicates=None):
        df = pd.read_csv(metadata)
        df = df[df["split"] == (self.SPLITS[split])]

        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = df["y"].tolist()
        self.a = df["a"].tolist() if train_attr == 'yes' else [0] * len(df["a"].tolist())
        self.transform_ = transform
        self._count_groups()

        if subsample_type is not None:
            self.subsample(subsample_type)

        if duplicates is not None:
            self.duplicate(duplicates)

    def _count_groups(self):
        self.weights_g, self.weights_y = [], []
        self.num_attributes = len(set(self.a))
        self.num_labels = len(set(self.y))
        self.group_sizes = [0] * self.num_attributes * self.num_labels
        self.class_sizes = [0] * self.num_labels

        for i in self.idx:
            self.group_sizes[self.num_attributes * self.y[i] + self.a[i]] += 1
            self.class_sizes[self.y[i]] += 1

        for i in self.idx:
            self.weights_g.append(len(self) / self.group_sizes[self.num_attributes * self.y[i] + self.a[i]])
            self.weights_y.append(len(self) / self.class_sizes[self.y[i]])

    def subsample(self, subsample_type):
        assert subsample_type in {"group", "class"}
        perm = torch.randperm(len(self)).tolist()
        min_size = min(list(self.group_sizes)) if subsample_type == "group" else min(list(self.class_sizes))

        counts_g = [0] * self.num_attributes * self.num_labels
        counts_y = [0] * self.num_labels
        new_idx = []
        for p in perm:
            y, a = self.y[self.idx[p]], self.a[self.idx[p]]
            if (subsample_type == "group" and counts_g[self.num_attributes * int(y) + int(a)] < min_size) or (
                    subsample_type == "class" and counts_y[int(y)] < min_size):
                counts_g[self.num_attributes * int(y) + int(a)] += 1
                counts_y[int(y)] += 1
                new_idx.append(self.idx[p])

        self.idx = new_idx
        self._count_groups()

    def duplicate(self, duplicates):
        new_idx = []
        for i, duplicate in zip(self.idx, duplicates):
            new_idx += [i] * duplicate
        self.idx = new_idx
        self._count_groups()

    def __getitem__(self, index):
        i = self.idx[index]
        x = self.transform(self.x[i])
        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.a[i], dtype=torch.long)
        return i, x, y, a

    def __len__(self):
        return len(self.idx)


class CMNIST(SubpopDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 250
    INPUT_SHAPE = (3, 224, 224,)
    data_type = "images"

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        root = Path(data_path) / 'cmnist'
        mnist = datasets.MNIST(root, train=True)
        X, y = mnist.data, mnist.targets

        if split == 'tr':
            X, y = X[:30000], y[:30000]
        elif split == 'va':
            X, y = X[30000:40000], y[30000:40000]
        elif split == 'te':
            X, y = X[40000:], y[40000:]
        else:
            raise NotImplementedError

        rng = np.random.default_rng(666)

        self.binary_label = np.bitwise_xor(y >= 5, (rng.random(len(y)) < hparams['cmnist_flip_prob'])).numpy()
        self.color = np.bitwise_xor(self.binary_label, (rng.random(len(y)) < hparams['cmnist_spur_prob']))
        self.imgs = torch.stack([X, X, torch.zeros_like(X)], dim=1).numpy()
        self.imgs[list(range(len(self.imgs))), (1 - self.color), :, :] *= 0

        # subsample color = 0
        if hparams['cmnist_attr_prob'] > 0.5:
            n_samples_0 = int((self.color == 1).sum() * (1 - hparams['cmnist_attr_prob']) / hparams['cmnist_attr_prob'])
            self._subsample(self.color == 0, n_samples_0, rng)
        # subsample color = 1
        elif hparams['cmnist_attr_prob'] < 0.5:
            n_samples_1 = int((self.color == 0).sum() * hparams['cmnist_attr_prob'] / (1 - hparams['cmnist_attr_prob']))
            self._subsample(self.color == 1, n_samples_1, rng)

        # subsample y = 0
        if hparams['cmnist_label_prob'] > 0.5:
            n_samples_0 = int(
                (self.binary_label == 1).sum() * (1 - hparams['cmnist_label_prob']) / hparams['cmnist_label_prob'])
            self._subsample(self.binary_label == 0, n_samples_0, rng)
        # subsample y = 1
        elif hparams['cmnist_label_prob'] < 0.5:
            n_samples_1 = int(
                (self.binary_label == 0).sum() * hparams['cmnist_label_prob'] / (1 - hparams['cmnist_label_prob']))
            self._subsample(self.binary_label == 1, n_samples_1, rng)

        self.idx = list(range(len(self.color)))
        self.x = torch.from_numpy(self.imgs).float() / 255.
        self.y = self.binary_label
        self.a = self.color

        self.transform_ = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
        ])
        self._count_groups()

        if subsample_type is not None:
            self.subsample(subsample_type)

        if duplicates is not None:
            self.duplicate(duplicates)

    def _subsample(self, mask, n_samples, rng):
        assert n_samples <= mask.sum()
        idxs = np.concatenate((np.nonzero(~mask)[0], rng.choice(np.nonzero(mask)[0], size=n_samples, replace=False)))
        rng.shuffle(idxs)
        self.imgs = self.imgs[idxs]
        self.color = self.color[idxs]
        self.binary_label = self.binary_label[idxs]

    def transform(self, x):
        return self.transform_(x)


class Waterbirds(SubpopDataset):
    CHECKPOINT_FREQ = 300
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        root = os.path.join(data_path, "waterbirds", "waterbird_complete95_forest2water2")
        metadata = os.path.join(data_path, "waterbirds", "metadata_waterbirds.csv")
        transform = transforms.Compose([
            transforms.Resize((int(224 * (256 / 224)), int(224 * (256 / 224)),)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data_type = "images"
        super().__init__(root, split, metadata, transform, train_attr, subsample_type, duplicates)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))

class CUB_200_2011_3D_toy_shape(SubpopDataset):
    CHECKPOINT_FREQ = 300
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        root = os.path.join(data_path, "CUB_200_2011_3D_toy_shape")
        metadata = os.path.join(data_path, "CUB_200_2011_3D_toy_shape", "metadata.csv")
        transform = transforms.Compose([
            transforms.Resize((int(224 * (256 / 224)), int(224 * (256 / 224)),)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data_type = "images"
        super().__init__(root, split, metadata, transform, train_attr, subsample_type, duplicates)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class CelebA(SubpopDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        root = os.path.join(data_path, "celeba", "img_align_celeba")
        metadata = os.path.join(data_path, "celeba", "metadata_celeba.csv")
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data_type = "images"
        super().__init__(root, split, metadata, transform, train_attr, subsample_type, duplicates)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class MultiNLI(SubpopDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        root = os.path.join(data_path, "multinli", "glue_data", "MNLI")
        metadata = os.path.join(data_path, "multinli", "metadata_multinli.csv")

        self.features_array = []
        assert hparams['text_arch'] == 'bert-base-uncased'
        for feature_file in [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]:
            features = torch.load(os.path.join(root, feature_file))
            self.features_array += features

        self.all_input_ids = torch.tensor(
            [f.input_ids for f in self.features_array], dtype=torch.long)
        self.all_input_masks = torch.tensor(
            [f.input_mask for f in self.features_array], dtype=torch.long)
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in self.features_array], dtype=torch.long)
        self.all_label_ids = torch.tensor(
            [f.label_id for f in self.features_array], dtype=torch.long)
        self.x_array = torch.stack(
            (self.all_input_ids, self.all_input_masks, self.all_segment_ids), dim=2)
        self.data_type = "text"
        super().__init__("", split, metadata, self.transform, train_attr, subsample_type, duplicates)

    def transform(self, i):
        return self.x_array[int(i)]


class CivilComments(SubpopDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None,
                 granularity="coarse"):
        text = pd.read_csv(os.path.join(
            data_path, "civilcomments/civilcomments_{}.csv".format(granularity))
        )
        metadata = os.path.join(data_path, "civilcomments", "metadata_civilcomments_{}.csv".format(granularity))

        self.text_array = list(text["comment_text"])
        if hparams['text_arch'] == 'bert-base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif hparams['text_arch'] in ['xlm-roberta-base', 'allenai/scibert_scivocab_uncased']:
            self.tokenizer = AutoTokenizer.from_pretrained(hparams['text_arch'])
        elif hparams['text_arch'] == 'distilbert-base-uncased':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        elif hparams['text_arch'] == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise NotImplementedError
        self.data_type = "text"
        super().__init__("", split, metadata, self.transform, train_attr, subsample_type, duplicates)

    def transform(self, i):
        text = self.text_array[int(i)]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
        )

        if len(tokens) == 3:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"]
                ), dim=2
                ), dim=0
            )
        else:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"]
                ), dim=2
                ), dim=0
            )


class CivilCommentsFine(CivilComments):

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        super().__init__(data_path, split, hparams, train_attr, subsample_type, duplicates, "fine")


class MetaShift(SubpopDataset):
    CHECKPOINT_FREQ = 300
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "metashift", "metadata_metashift.csv")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data_type = "images"
        super().__init__('/', split, metadata, transform, train_attr, subsample_type, duplicates)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class ImagenetBG(SubpopDataset):
    INPUT_SHAPE = (3, 224, 224,)
    SPLITS = {
        'tr': 'train',
        'va': 'val',
        'te': 'test',
        'mixed_rand': 'mixed_rand',
        'no_fg': 'no_fg',
        'only_fg': 'only_fg'
    }
    EVAL_SPLITS = ['te', 'mixed_rand', 'no_fg', 'only_fg']
    N_STEPS = 10001
    CHECKPOINT_FREQ = 500

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "backgrounds_challenge", "metadata.csv")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data_type = "images"
        super().__init__('/', split, metadata, transform, train_attr, subsample_type, duplicates)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class BaseImageDataset(SubpopDataset):

    def __init__(self, metadata, split, train_attr='yes', subsample_type=None, duplicates=None):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data_type = "images"
        super().__init__('/', split, metadata, transform, train_attr, subsample_type, duplicates)

    def transform(self, x):
        if self.__class__.__name__ in ['MIMICNoFinding', 'CXRMultisite'] and 'MIMIC-CXR-JPG' in x:
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-5] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')

            if reduced_img_path.is_file():
                x = str(reduced_img_path.resolve())

        return self.transform_(Image.open(x).convert("RGB"))


class NICOpp(BaseImageDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "nicopp", "metadata.csv")
        super().__init__(metadata, split, train_attr, subsample_type, duplicates)


class MIMICNoFinding(BaseImageDataset):
    N_STEPS = 20001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "MIMIC-CXR-JPG", 'subpop_bench_meta', "metadata_no_finding.csv")
        super().__init__(metadata, split, train_attr, subsample_type, duplicates)


class CheXpertNoFinding(BaseImageDataset):
    N_STEPS = 20001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "metadata_no_finding.csv")
        super().__init__(metadata, split, train_attr, subsample_type, duplicates)


class CXRMultisite(BaseImageDataset):
    N_STEPS = 20001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)
    SPLITS = {
        'tr': 0,
        'va': 1,
        'te': 2,
        'deploy': 3
    }
    EVAL_SPLITS = ['te', 'deploy']

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "MIMIC-CXR-JPG", 'subpop_bench_meta', "metadata_multisite.csv")
        super().__init__(metadata, split, train_attr, subsample_type, duplicates)


class MIMICNotes(SubpopDataset):
    N_STEPS = 10001
    CHECKPOINT_FREQ = 200
    INPUT_SHAPE = (10000,)

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        assert hparams['text_arch'] == 'bert-base-uncased'
        metadata = os.path.join(data_path, "mimic_notes", 'subpop_bench_meta', "metadata.csv")
        self.x_array = np.load(os.path.join(data_path, "mimic_notes", 'features.npy'))
        self.data_type = "tabular"
        super().__init__("", split, metadata, self.transform, train_attr, subsample_type, duplicates)

    def transform(self, x):
        return self.x_array[int(x), :].astype('float32')


class BREEDSBase(BaseImageDataset):
    N_STEPS = 60_001
    CHECKPOINT_FREQ = 2000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)
    SPLITS = {
        'tr': 0,
        'va': 1,
        'te': 2,
        'zs': 3
    }
    EVAL_SPLITS = ['te', 'zs']


class Living17(BREEDSBase):
    def __init__(self, data_path, split, hparams, train_attr='no', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "breeds", "metadata_living17.csv")
        super().__init__(metadata, split, train_attr, subsample_type, duplicates)


class Entity13(BREEDSBase):
    def __init__(self, data_path, split, hparams, train_attr='no', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "breeds", "metadata_entity13.csv")
        super().__init__(metadata, split, train_attr, subsample_type, duplicates)


class Entity30(BREEDSBase):
    def __init__(self, data_path, split, hparams, train_attr='no', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "breeds", "metadata_entity30.csv")
        super().__init__(metadata, split, train_attr, subsample_type, duplicates)


class Nonliving26(BREEDSBase):
    def __init__(self, data_path, split, hparams, train_attr='no', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "breeds", "metadata_nonliving26.csv")
        super().__init__(metadata, split, train_attr, subsample_type, duplicates)


class RSNADataset(Dataset):
    def __init__(self, df, data_dir, tfms, mean=0, std=0, label_col='cancer', dataset='rsna'):
        self.df = df
        self.data_dir = Path(data_dir)
        print(tfms)
        self.transform = tfms

        self.mean = mean
        self.std = std
        self.label_col = label_col
        self.dataset = dataset

        if self.dataset.lower() == "vindr":
            unique_patient_ids = {id: idx for idx, id in enumerate(df['patient_id'].unique())}
            self.df['patient_id_idx'] = self.df['patient_id'].map(unique_patient_ids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        if self.dataset.lower() == 'rsna':
            img_path = str(
                self.data_dir / "train_images_png" / str(self.df.iloc[idx]['patient_id']) / str(
                    self.df.iloc[idx]['image_id']))
        elif self.dataset.lower() == 'vindr':
            img_path = str(
                self.data_dir / "images_png" / str(self.df.iloc[idx]['patient_id']) / str(
                    self.df.iloc[idx]['image_id']))

        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = img.astype('float32')
        img -= img.min()
        img /= img.max()
        img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32)

        label = torch.tensor(data[self.label_col], dtype=torch.long)
        clip = torch.tensor(data["CLIP_V1_bin"], dtype=torch.long)
        scar = torch.tensor(data["SCAR_V1_bin"], dtype=torch.long)
        mark = torch.tensor(data["MARK_V1_bin"], dtype=torch.long)
        mole = torch.tensor(data["MOLE_V1_bin"], dtype=torch.long)
        laterality_tensor = torch.tensor(data["laterality"], dtype=torch.long)

        mass = None
        calc = None
        if self.dataset.lower() == 'rsna':
            calc = torch.tensor(data["Suspicious_Calcification_th_0.25"], dtype=torch.long)
            mass = torch.tensor(data["Mass_th_0.15"], dtype=torch.long)
            fold = torch.tensor(data["fold"], dtype=torch.long)
            patient_id = torch.tensor(data["patient_id"], dtype=torch.long)
        elif self.dataset.lower() == 'vindr':
            calc = torch.tensor(data["Suspicious_Calcification"], dtype=torch.long)
            mass = torch.tensor(data["Mass"], dtype=torch.long)
            fold = torch.tensor(1, dtype=torch.long)
            patient_id = torch.tensor(data["patient_id_idx"], dtype=torch.long)
        return {
            'img': img,
            'label': label,
            'mass': mass,
            'calc': calc,
            'clip': clip,
            'scar': scar,
            'mark': mark,
            'mole': mole,
            'fold': fold,
            'patient_id': patient_id,
            'laterality': laterality_tensor
        }


class EmbedDataset(Dataset):
    def __init__(self, df, data_dir, tfms, mean=0, std=0, label_col='cancer'):
        self.df = df
        self.data_dir = Path(data_dir)
        print(tfms)
        self.transform = tfms

        self.mean = mean
        self.std = std
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img_path = str(self.data_dir / "train_images_png" / str(self.df.iloc[idx]['anon_dicom_path']))
        img_path = str(img_path).replace('.dcm', '.png')

        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = img.astype('float32')
        img -= img.min()
        img /= img.max()
        img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32)

        label = torch.tensor(data[self.label_col], dtype=torch.long)
        calc = torch.tensor(data["calc"], dtype=torch.long)
        fold = torch.tensor(data["fold"], dtype=torch.long)
        patient_id = torch.tensor(data["patient_id"], dtype=torch.long)
        laterality_tensor = torch.tensor(data["laterality"], dtype=torch.long)
        return {
            'img': img,
            'label': label,
            'calc': calc,
            'fold': fold,
            'patient_id': patient_id,
            'laterality': laterality_tensor
        }


class VindrDataset(Dataset):
    def __init__(self, df, data_dir, tfms, mean=0, std=0, label_col='Mass'):
        self.df = df
        self.data_dir = Path(data_dir)
        print(tfms)
        self.transform = tfms

        self.mean = mean
        self.std = std
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img_path = str(
            self.data_dir / "train_images_png" / str(self.df.iloc[idx]['patient_id']) / str(
                self.df.iloc[idx]['image_id']))

        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = img.astype('float32')
        img -= img.min()
        img /= img.max()
        img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32)

        label = torch.tensor(data[self.label_col], dtype=torch.long)
        calc = torch.tensor(data["Suspicious_Calcification"], dtype=torch.long)
        return {
            'img': img,
            'label': label,
            'calc': calc
        }


class Dataset_NIH(Dataset):
    def __init__(self, df, class_names, transform, seed=0):

        super(Dataset_NIH, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.class_names = class_names
        self.transform = transform
        self.df = df
        self.df = self.df.fillna(0)
        self.labels = []
        for name in self.class_names:

            if name in self.df.columns:
                # self.csv.loc[healthy, name] = 0
                mask = self.df[name]
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        self.labels[self.labels == -1] = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = str(self.df.iloc[idx]["path"])
        img_path = img_path.replace(
            "/ocean/projects/asc170022p/shared/Data/chestXRayDatasets/NIH_ChestXRay/images/images",
            "/restricted/projectnb/batmanlab/shared/Data/chestXRayDatasets/NIH_ChestXRay/images/images",
        )
        raw_img = Image.open(img_path)
        img = raw_img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        tube_label = self.df.iloc[idx]["tube_geq_0p5"]
        effusion = self.df.iloc[idx]["Effusion"]
        return {
            "raw_img": raw_img,
            "img": img,
            "label": self.labels[idx],
            "text": "dummy",
            "tube": tube_label,
            "img_path": img_path,
            "effusion": effusion
        }


def collate_NIH(batch):
    raw_img = [item["raw_img"] for item in batch]
    img = [item["img"] for item in batch]
    label = [torch.tensor(item["label"], dtype=torch.float32) for item in batch]
    text = [item["text"] for item in batch]
    tube = [torch.tensor(item["tube"], dtype=torch.float32) for item in batch]
    effusion = [torch.tensor(item["effusion"], dtype=torch.float32) for item in batch]
    img_path = [item["img_path"] for item in batch]

    return {
        "raw_img": raw_img,
        "img": torch.stack(img),
        "label": torch.stack(label),
        "tube": torch.stack(tube),
        "effusion": torch.stack(effusion),
        "text": text,
        "img_path": img_path
    }

