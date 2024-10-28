# Ladder
[![Project](https://img.shields.io/badge/Project%20page-lightgreen)]()
[![Paper](https://img.shields.io/badge/Paper-9cf)](https://arxiv.org/abs/2408.07832)
[![Poster](https://img.shields.io/badge/Poster-orange)]()
![](https://visitor-badge.laobi.icu/badge?page_id=batmanlab.Ladder&right_color=%23FFA500)

Official repository of [Ladder](https://arxiv.org/abs/2408.07832). Code to be uploaded soon.


## Datasets
 - Waterbirds
 - CelebA
 - MetaShift
 - NIH Chest X-ray
 - RSNA-Mammo
 - VinDr-Mammo

## Data download
- Waterbids and Metashift
```bash
python ./src/codebase/SubpopBench-main/subpopbench/scripts/download.py \
--datasets "waterbirds" "metashift" \
--data_path "Ladder/data/new" \
--download True
```

- CelebA - Download from URL
- Same for NIH, RSNA-Mammo, VinDr-Mammo
- Mention metadata files for them

## Generating captions
Using blip
```bash
python ./src/codebase/caption_images.py \
  --seed=0 \
  --dataset="Waterbirds" \
  --img-path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/waterbird_complete95_forest2water2" \
  --csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/metadata_waterbirds.csv" \
  --save_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/va_metadata_waterbirds_captioning_blip.csv" \
  --split="va" \
  --captioner="blip" 
```

Using gpt4
```bash
```

## Vision language models
 - CLIP
 - CXR-CLIP
 - Mammo-CLIP

### [Warning] edit TRANSFORMERS_CACHE and TORCH_HOME in save_img_reps.py

### Mention the sentences of NIH-CXR

### Upload tokenizers to huggingface