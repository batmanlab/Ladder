import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionTextDualEncoderProcessor, VisionEncoderDecoderModel
from PIL import Image
import argparse
import os
from tqdm import tqdm

from utils import seed_all


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CelebA", type=str)
    parser.add_argument("--img-path", metavar="DIR", default="../data/celeba/img_align_celeba", help="" )
    parser.add_argument("--csv", metavar="DIR", default="../data/celeba/metadata_celeba.csv", help="")
    parser.add_argument("--save_csv", metavar="DIR", default="../data/celeba/metadata_celeba_captions.csv", help="")
    parser.add_argument("--split", default="tr", type=str)
    parser.add_argument("--captioner", default="blip", type=str)
    parser.add_argument("--seed", default="0", type=int)
    return parser.parse_args()


class ImageDataset(Dataset):
    def __init__(self, dataframe, img_path, transform=None):
        self.dataframe = dataframe
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_filename = self.dataframe.iloc[index]['filename']
        img_full_path = os.path.join(self.img_path, img_filename)
        image = Image.open(img_full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_filename


def custom_collate(batch):
    # Unzip the batch into images and filenames
    images, filenames = zip(*batch)
    return list(images), list(filenames)


import time
import transformers


def load_model_with_retries(model_name, retries=3, delay=5):
    for attempt in range(retries):
        try:
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            processor = BlipProcessor.from_pretrained(model_name)
            return model, processor
        except transformers.utils.logging.HttpRequestError as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            else:
                raise


def main(args):
    seed_all(args.seed)
    print(args.save_csv)
    SPLITS = {
        'tr': 0,
        'va': 1,
        'te': 2
    }
    df = pd.read_csv(args.csv)
    print(f"Original shape: {df.shape}")
    if args.dataset.lower() != "urbancars":
        df = df[df["split"] == SPLITS[args.split]]

    print(f"{args.split} shape: {df.shape}")
    print(df.columns)
    print(df.head())

    # iterate df
    captions = []
    processor = None
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.captioner == "blip":
        model_path = "./blip_model"
        processor_path = "./blip_processor"
        if os.path.exists(model_path):
            model = BlipForConditionalGeneration.from_pretrained(model_path)
            processor = BlipProcessor.from_pretrained(processor_path)
        else:
            model, processor = load_model_with_retries("Salesforce/blip-image-captioning-base")
            model.save_pretrained(model_path)
            processor.save_pretrained(processor_path)

        # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.to(device)
    elif args.captioner == "google":
        loc = "ydshieh/vit-gpt2-coco-en"
        model = VisionEncoderDecoderModel.from_pretrained(loc)
        model.to(device)
        processor = VisionTextDualEncoderProcessor.from_pretrained(loc)

    with tqdm(total=df.shape[0], desc=f"Processing {args.split} data") as t:
        for idx, row in df.iterrows():
            img_path = os.path.join(args.img_path, row["filename"])
            image = Image.open(img_path)
            inputs = processor(image, return_tensors="pt").to(device)
            if args.captioner == "blip":
                outputs = model.generate(**inputs, max_length=50, num_beams=3)
            elif args.captioner == "google":
                outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            captions.append(caption)
            t.set_postfix(batch_id='{0}'.format(idx))
            t.update()

    df[f"{args.captioner}_caption"] = captions
    df.to_csv(args.save_csv, index=False)
    print(args.save_csv)


if __name__ == "__main__":
    _args = config()
    main(_args)
