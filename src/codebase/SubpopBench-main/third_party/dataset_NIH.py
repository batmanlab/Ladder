import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class Dataset_NIH(Dataset):
    """
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY, Mark RG, Horng S.
    MIMIC-CXR: A large publicly available database of labeled chest radiographs.
    arXiv preprint arXiv:1901.07042. 2019 Jan 21.

    https://arxiv.org/abs/1901.07042

    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(
            self, csvpath, class_names, transform, data_aug=None, seed=0, unique_patients=True,
            is_rgb=False, is_train_mode=True, is_classifier=True
    ):

        # super(Dataset_NIH, self).__init__()
        super().__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.class_names = class_names

        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.csv = self.csv.fillna(0)
        self.is_train_mode = is_train_mode
        self.is_rgb = is_rgb
        self.is_classifier = is_classifier
        # self.views = ["PA"]
        # self.csv = self.csv[self.csv["ViewPosition"].isin(self.views)]

        # if unique_patients:
        #     self.csv = self.csv.groupby("patient_id").first().reset_index()
        # Get our classes.
        # healthy = self.csv["No Finding"] == 1
        self.labels = []
        # for name in self.class_names:
        #     if name in self.csv.columns:
        #         # self.csv.loc[healthy, name] = 0
        #         mask = self.csv[name]
            # self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        # make all the -1 values into 0 to keep things simple
        self.labels[self.labels == -1] = 0

    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self), self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = str(self.csv.iloc[idx]["new_path"])
        raw_img = Image.open(img_path)
        if self.is_rgb:
            img = raw_img.convert("RGB")
        else:
            img = raw_img
            if len(img.getbands()) > 1:  # needed for NIH dataset as some images are RGBA 4 channel
                img = img.convert('L')

        if self.transform is not None:
            img = self.transform(img)
        if self.data_aug is not None:
            img = self.data_aug(img)

        if not self.is_classifier:  # for the aligner
            tube_label = self.csv.iloc[idx]["tube_geq_0p5"]
            tube_prob = self.csv.iloc[idx]["tube_prob"]
            return {
                "raw_img": raw_img, "clf_img": img, "lab": self.labels[idx], "file_name": img_path, "text": "dummy",
                "tube_label": tube_label, "tube_prob": tube_prob
            }
        elif self.is_classifier and self.is_train_mode:  # for the biased classifier training
            return {
                "img": img, "lab": self.csv.iloc[idx]["Pneumothorax"], "idx": idx, "file_name": img_path
            }
        elif self.is_classifier and not self.is_train_mode:  # for the biased classifier testing
            tube_label = self.csv.iloc[idx]["tube_geq_0p5"]
            tube_prob = self.csv.iloc[idx]["tube_prob"]
            return {
                "img": img, "lab": self.labels[idx], "idx": idx, "file_name": img_path,
                "tube_label": tube_label, "tube_prob": tube_prob
            }


def collate_NIH(batch):
    raw_img = [item["raw_img"] for item in batch]
    clf_img = [item["clf_img"] for item in batch]
    lab = [torch.tensor(item["lab"], dtype=torch.float32) for item in batch]
    tube_label = [torch.tensor(item["tube_label"], dtype=torch.float32) for item in batch]
    tube_prob = [torch.tensor(item["tube_prob"], dtype=torch.float32) for item in batch]
    file_name = [item["file_name"] for item in batch]
    text = [item["text"] for item in batch]

    return {
        "raw_img": raw_img,
        "clf_img": torch.stack(clf_img),
        "tube_label": torch.stack(tube_label),
        "tube_prob": torch.stack(tube_prob),
        "lab": torch.stack(lab),
        "file_name": file_name,
        "text": text
    }
