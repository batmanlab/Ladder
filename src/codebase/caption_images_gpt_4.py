import argparse
from utils import seed_all
import pandas as pd
import requests
import base64
from mimetypes import guess_type
from PIL import Image
import io
from pathlib import Path


def caption_image(encoded_image, text, model="gpt-4-turbo", api_key="xxxx"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{encoded_image}"
                        }
                    },
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CelebA", type=str)
    parser.add_argument("--img-path", metavar="DIR", default="", help="")
    parser.add_argument("--csv", metavar="DIR", default="", help="")
    parser.add_argument("--save_csv", metavar="DIR", default="", help="")
    parser.add_argument("--split", default="va", type=str)
    parser.add_argument("--model", default="gpt-4o", type=str)
    parser.add_argument("--api_key", default="xxxx", type=str)
    parser.add_argument("--seed", default="0", type=int)
    return parser.parse_args()


def resize_and_encode_image(image_path, size=(150, 150)):
    with Image.open(image_path) as img:
        img = img.resize(size, Image.Resampling.LANCZOS)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    base64_encoded_data = base64.b64encode(img_byte_arr).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


def main(args):
    seed_all(args.seed)
    print(args.save_csv)
    print(args.csv)
    df = pd.read_csv(args.csv)
    print(df.shape)
    SPLITS = {
        'tr': 0,
        'va': 1,
        'te': 2
    }
    print(f"Original shape: {df.shape}")
    df = df[df["split"] == SPLITS[args.split]]
    print(f"{args.split} shape: {df.shape}")
    print(df.columns)
    print(df.head())

    print(f"Total images: {len(df)}")
    caption_arr = []
    for idx, row in df.iterrows():
        encoded_image = resize_and_encode_image(Path(args.img_path) / row["filename"])
        text = "Describe the image here based on manual observation or another model's output in maximum 2 lines."
        response = caption_image(encoded_image, text, model=args.model, api_key=args.api_key)
        message_content = response.json()['choices'][0]['message']['content']
        print(f"{idx}: {message_content}")
        if "id" in row:
            caption_arr.append({
                "id": row["id"],
                "filename": row["filename"],
                "split": row["split"],
                "y": row["y"],
                "a": row["a"],
                f"{args.model}_caption": message_content
            })
        else:
            caption_arr.append({
                "id": idx,
                "filename": row["filename"],
                "split": row["split"],
                "y": row["y"],
                "a": row["a"],
                f"{args.model}_caption": message_content
            })

    caption_df = pd.DataFrame(caption_arr)
    caption_df.to_csv(args.save_csv, index=False)


if __name__ == "__main__":
    _args = config()
    main(_args)
