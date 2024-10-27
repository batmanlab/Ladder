import numpy as np
import os
import os.path
import pprint

import numpy as np
import collections

class Dataset():
    def __init__(self):
        pass

    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.class_names, counts))

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.string()

    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")


class SubsetDataset(Dataset):
    def __init__(self, dataset, idxs=None):
        super(SubsetDataset, self).__init__()
        self.dataset = dataset
        self.class_names = dataset.class_names

        self.idxs = idxs
        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]

        self.csv = self.csv.reset_index(drop=True)

        if hasattr(self.dataset, 'which_dataset'):
            self.which_dataset = self.dataset.which_dataset[self.idxs]

    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(
            len(self)) + "└ of " + self.dataset.string().replace("\n", "\n  ")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]
