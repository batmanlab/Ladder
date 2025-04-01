import random
from PIL import Image, ImageDraw
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


# Train Set Statistics:
# Waterbirds are 0.230 of the examples.
# Class 0:
#   a = 0 (yellow left of red): 0.950, n = 3507
#   a = 1 (random positioning): 0.050, n = 185
# Class 1:
#   a = 0 (yellow left of red): 0.050, n = 55
#   a = 1 (random positioning): 0.950, n = 1048
#
# Val Set Statistics:
# Waterbirds are 0.230 of the examples.
# Class 0:
#   a = 0 (yellow left of red): 0.501, n = 462
#   a = 1 (random positioning): 0.499, n = 461
# Class 1:
#   a = 0 (yellow left of red): 0.500, n = 138
#   a = 1 (random positioning): 0.500, n = 138
#
# Test Set Statistics:
# Waterbirds are 0.222 of the examples.
# Class 0:
#   a = 0 (yellow left of red): 0.500, n = 2255
#   a = 1 (random positioning): 0.500, n = 2255
# Class 1:
#   a = 0 (yellow left of red): 0.500, n = 642
#   a = 1 (random positioning): 0.500, n = 642

import argparse
import random
from PIL import Image, ImageDraw
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def draw_3d_box(draw, x, y, box_size, color):
    """
    Draws a 3D box at the specified position.
    :param draw: ImageDraw object.
    :param x: Top-left x coordinate of the box.
    :param y: Top-left y coordinate of the box.
    :param box_size: Size of the box.
    :param color: Base color of the box (R, G, B).
    """
    # Define the 3D shift to create depth (top face and side face)
    depth_shift = int(box_size * 0.2)

    # Front face (main color)
    draw.rectangle([x, y, x + box_size, y + box_size], fill=color)

    # Top face (lighter shade)
    draw.polygon([(x, y), (x + depth_shift, y - depth_shift),
                  (x + box_size + depth_shift, y - depth_shift),
                  (x + box_size, y)],
                 fill=(int(color[0] * 1.2), int(color[1] * 1.2), int(color[2] * 1.2)))

    # Side face (darker shade)
    draw.polygon([(x + box_size, y), (x + box_size + depth_shift, y - depth_shift),
                  (x + box_size + depth_shift, y + box_size - depth_shift),
                  (x + box_size, y + box_size)],
                 fill=(int(color[0] * 0.8), int(color[1] * 0.8), int(color[2] * 0.8)))


def overlay_3d_boxes(image, class_label, bias, color1=(255, 255, 0), color2=(255, 0, 0), size_fraction=0.2):
    """
    Overlays two 3D rectangles (boxes) on the image. The yellow (color1) and red (color2) boxes.
    :param image: PIL Image
    :param class_label: Class label (0 or 1) to determine box positioning.
    :param bias: a = 0 for correlated positioning (yellow to left of red), a = 1 for random positioning.
    :param color1: Tuple for first rectangle color (Yellow).
    :param color2: Tuple for second rectangle color (Red).
    :param size_fraction: Fraction of image size each box occupies.
    :return: Image with 3D rectangle overlays.
    """
    img = image.copy().convert('RGBA')
    width, height = img.size
    box_size = int(min(width, height) * size_fraction)

    # Create drawing context
    draw = ImageDraw.Draw(img)

    if bias == 0:
        # Correlated positioning: Yellow to the left of Red
        x1_pos = int(width * 0.1)  # Yellow box at 10% of the width
        y1_pos = int(height * 0.4)
        x2_pos = x1_pos + int(width * 0.15)  # Reduce gap between yellow and red boxes
        y2_pos = y1_pos  # Both on the same height level
    else:
        # Random positioning, ensure boxes don't overlap
        x1_pos, y1_pos = random.randint(0, width - box_size), random.randint(0, height - box_size)
        while True:
            x2_pos, y2_pos = random.randint(0, width - box_size), random.randint(0, height - box_size)
            if abs(x1_pos - x2_pos) > box_size or abs(y1_pos - y2_pos) > box_size:
                break

    draw_3d_box(draw, x1_pos, y1_pos, box_size, color1)  # Yellow box
    draw_3d_box(draw, x2_pos, y2_pos, box_size, color2)  # Red box

    return img.convert('RGB')


def main():
    parser = argparse.ArgumentParser(description="Create a 3D toy-shape version of the CUB dataset.")
    parser.add_argument('--cub_dir', type=str, default='/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/data/CUB_200_2011',
                        help='Path to the original CUB_200_2011 directory.')
    parser.add_argument('--output_dir', type=str, default='/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/data/',
                        help='Directory where the new dataset will be saved.')
    parser.add_argument('--dataset_name', type=str, default='CUB_200_2011_3D_toy_shape',
                        help='Name of the output dataset folder.')
    parser.add_argument('--val_frac', type=float, default=0.2,
                        help='Fraction of training data to allocate for validation.')
    parser.add_argument('--confounder_strength', type=float, default=0.95,
                        help='Confounder correlation strength for the training set.')
    args = parser.parse_args()

    cub_dir = args.cub_dir
    output_dir = args.output_dir
    dataset_name = args.dataset_name
    val_frac = args.val_frac
    confounder_strength = args.confounder_strength

    images_path = os.path.join(cub_dir, 'images.txt')

    df = pd.read_csv(
        images_path,
        sep=" ",
        header=None,
        names=['id', 'filename'],
        index_col='id')

    # Set up labels of waterbirds vs. landbirds
    species = np.unique([img_filename.split('/')[0].split('.')[1].lower() for img_filename in df['filename']])
    water_birds_list = [
        'Albatross',  # Seabirds
        'Auklet',
        'Cormorant',
        'Frigatebird',
        'Fulmar',
        'Gull',
        'Jaeger',
        'Kittiwake',
        'Pelican',
        'Puffin',
        'Tern',
        'Gadwall',  # Waterfowl
        'Grebe',
        'Mallard',
        'Merganser',
        'Guillemot',
        'Pacific_Loon'
    ]

    water_birds = {}
    for species_name in species:
        water_birds[species_name] = 0
        for water_bird in water_birds_list:
            if water_bird.lower() in species_name:
                water_birds[species_name] = 1
    species_list = [img_filename.split('/')[0].split('.')[1].lower() for img_filename in df['filename']]
    df['y'] = [water_birds[species] for species in species_list]

    # Assign train/val/test splits
    train_test_df = pd.read_csv(
        os.path.join(cub_dir, 'train_test_split.txt'),
        sep=" ",
        header=None,
        names=['id', 'split'],
        index_col='id')

    df = df.join(train_test_df, on='id')
    test_ids = df.loc[df['split'] == 0].index
    train_ids = np.array(df.loc[df['split'] == 1].index)
    val_ids = np.random.choice(
        train_ids,
        size=int(np.round(val_frac * len(train_ids))),
        replace=False)

    df.loc[train_ids, 'split'] = 0
    df.loc[val_ids, 'split'] = 1
    df.loc[test_ids, 'split'] = 2

    # Assign confounders (rectangle positions)
    df['a'] = 1  # Default: no correlation (random positioning)

    train_ids = np.array(df.loc[df['split'] == 0].index)
    val_ids = np.array(df.loc[df['split'] == 1].index)
    test_ids = np.array(df.loc[df['split'] == 2].index)

    for split_idx, ids in enumerate([train_ids, val_ids, test_ids]):
        for y in (0, 1):
            if split_idx == 0:  # Training set
                pos_fraction = confounder_strength if y == 0 else 1 - confounder_strength
            else:  # Validation and test sets: use a 50-50 split
                pos_fraction = 0.5

            subset_df = df.loc[ids, :]
            y_ids = np.array((subset_df.loc[subset_df['y'] == y]).index)
            num_pos = int(np.round(pos_fraction * len(y_ids)))

            if num_pos > 0:
                pos_ids = np.random.choice(y_ids, size=num_pos, replace=False)
                df.loc[pos_ids, 'a'] = 0
                df.loc[list(set(y_ids) - set(pos_ids)), 'a'] = 1

    # Loop through images and apply the modifications
    output_subfolder = os.path.join(output_dir, dataset_name)
    os.makedirs(output_subfolder, exist_ok=True)

    for i in tqdm(df.index):
        img_path = os.path.join(cub_dir, 'images', df.loc[i, 'filename'])
        img = Image.open(img_path).convert('RGBA')

        class_label = df.loc[i, 'y']
        bias = df.loc[i, 'a']

        img_with_boxes = overlay_3d_boxes(
            img,
            class_label=class_label,
            bias=bias,
            color1=(255, 255, 0),
            color2=(255, 0, 0),
            size_fraction=0.2
        )

        output_path = os.path.join(output_subfolder, df.loc[i, 'filename'])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_with_boxes.save(output_path)

    df.to_csv(os.path.join(output_subfolder, 'rsna_metadata.csv'))

    # Print dataset statistics
    for split, split_label in [(0, 'train'), (1, 'val'), (2, 'test')]:
        print(f"\n{split_label.capitalize()} Set Statistics:")
        split_df = df.loc[df['split'] == split, :]
        print(f"Waterbirds are {np.mean(split_df['y']):.3f} of the examples.")
        for y in (0, 1):
            print(f"Class {y}:")
            print(f"  a = 0 (yellow left of red): {np.mean(split_df.loc[split_df['y'] == y, 'a'] == 0):.3f}, "
                  f"n = {np.sum((split_df['y'] == y) & (df['a'] == 0))}")
            print(f"  a = 1 (random positioning): {np.mean(split_df.loc[split_df['y'] == y, 'a'] == 1):.3f}, "
                  f"n = {np.sum((split_df['y'] == y) & (df['a'] == 1))}")


if __name__ == "__main__":
    main()
