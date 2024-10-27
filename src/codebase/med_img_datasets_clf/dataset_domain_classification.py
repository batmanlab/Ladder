import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


class Dataset_domain_classifer(torch.utils.data.Dataset):
    def __init__(self, args, df, iaa_transform=None, transform=None):
        self.transform = transform
        self.iaa_transform = iaa_transform
        self.df = df
        self.concept_model_type = args.concept_model_type
        self.root_dir = args.data_dir
        self.img_dir_rsna = args.img_dir_rsna
        self.img_dir_vindr = args.img_dir_vindr
        self.img_dir_upmc = args.img_dir_upmc
        self.mean = args.mean
        self.std = args.std
        self.df = self.df.fillna(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = None
        dataset = self.df.iloc[idx]['DATASET']
        if dataset.lower() == 'upmc':
            study_id = str(self.df.iloc[idx]['STUDY_ID'])
            img_path = self.root_dir / self.img_dir_upmc / f'Patient_{study_id}' / self.df.iloc[idx]['IMAGE_ID']
        elif dataset.lower() == 'rsna':
            img_path = self.root_dir / self.img_dir_rsna / str(self.df.iloc[idx]['STUDY_ID']) / self.df.iloc[idx][
                'IMAGE_ID']
            img_path = f'{img_path}.png'
        elif dataset.lower() == 'vindr':
            study_id = str(self.df.iloc[idx]['STUDY_ID'])
            img_path = self.root_dir / self.img_dir_vindr / f'{study_id}' / self.df.iloc[idx]['IMAGE_ID']
            img_path = f'{img_path}.png'

        if self.concept_model_type.lower() == "classification":
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if self.transform is not None:
                augmented = self.transform(image=img)
                img = augmented['image']
            img = img.astype('float32')
            img -= img.min()
            img /= img.max()
            img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32)
            img = img.unsqueeze(0)
        elif self.concept_model_type.lower() == "detection":
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = Image.fromarray(img).convert('RGB')
            img = np.array(img)
            if self.iaa_transform:
                img = self.iaa_transform(image=img)
            if self.transform:
                img = self.transform(img)
            img = img.to(torch.float32)
            img -= img.min()
            img /= img.max()
            img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32)

        y = self.df.iloc[idx]['TARGET']
        return {'x': img, 'y': y.astype(np.float32), 'img_path': str(img_path)}


def collator_domain_classifier(batch):
    return {
        'x': torch.stack([item['x'] for item in batch]),
        'y': torch.from_numpy(np.array([item["y"] for item in batch], dtype=np.float32)),
        'img_path': [item['img_path'] for item in batch]
    }
