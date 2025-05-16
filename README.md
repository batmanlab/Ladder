# <div align="center"> LADDER: Language-Driven Slice Discovery and Error Rectification in Vision Classifiers <div>

[![Project](https://img.shields.io/badge/Project%20page-lightgreen)](https://shantanu-ai.github.io/projects/ACL-2025-Ladder/index.html)
[![Paper](https://img.shields.io/badge/Paper-9cf)](https://arxiv.org/abs/2408.07832)
[![Poster](https://img.shields.io/badge/Poster-orange)]()
![](https://visitor-badge.laobi.icu/badge?page_id=batmanlab.Ladder&right_color=%23FFA500)
---
> [Shantanu Ghosh<sup>1</sup>](https://shantanu-ai.github.io/),
> [Rayan Syed<sup>1</sup>](https://heartyhaven.github.io/),
> [Chenyu Wang<sup>1</sup>](https://chyuwang.com/),
> [Vaibhav Choudhary<sup>1</sup>](https://vaibhavchoudhary.com/),
> [Binxu Li<sup>4</sup>](https://www.linkedin.com/in/binxu-li-595b64245/),
> [Clare B. Poynton<sup>2</sup>](https://www.bumc.bu.edu/camed/profile/clare-poynton/),
> [Shyam Visweswaran<sup>3</sup>](https://www.thevislab.com/lab/doku.php)
> [Kayhan Batmanghelich<sup>1</sup>](https://www.batman-lab.com/)
<br/>
<sup>1</sup>BU ECE, <sup>2</sup> Stanford University, <sup>3</sup> BUMC, <sup>4</sup> Pitt DBMI <br/>

---

## 🚨 Highlights

- 📊 **6 Datasets Evaluated**
  - 🐦 **Natural Images**: Waterbirds, CelebA, MetaShift  
  - 🏥 **Medical Imaging**: NIH ChestX-ray, RSNA-Mammo, VinDr-Mammo  

- 🧪 **~20 Bias Mitigation Algorithms Benchmarked**
  - 💡 ERM, GroupDRO, CVaR-DRO, JTT, LfF, DFR  
  - 🧬 CORAL, IRM, V-REx, IB-IRM, Reweighting, Mixup, AugMix

- 🧠 **11 Architectures Across 5 Pretraining Strategies**
  - 🧱 **CNNs**: ResNet-50, EfficientNet-B5  
  - 🔲 **ViTs**: ViT-B/16, ViT-S/16  
  - 🧪 **Pretrained With**: SimCLR, Barlow Twins, DINO, CLIP (OpenAI),  
    IN1K, IN21K, SWAG, LAION-2B  
  
- 💬 **4 LLMs for Hypothesis Generation**
  - 🧠 GPT-4o, Gemini, LLaMA, Claude  

  Star 🌟 us if you think it is helpful!!
---

> ⚠️ **Caveats**  
> - All LLMs were evaluated using checkpoints available **before Jan 11, 2025**.  
>   With the release of newer versions, we **cannot guarantee** the same hypotheses will be generated as reported in the paper.  
> - By default, we demonstrate results using **ResNet-50** (classifier), **ViT-B/32** (as the vision-language representation space), and **GPT-4o** (for hypothesis generation).  
>   The code is modular and can be **easily extended** to other models and LLMs.


## Data zoo
Please refer to the [dataset_zoo.md](dataset_zoo.md) for the details of the datasets used in this project.


Official repository of [Ladder](https://arxiv.org/abs/2408.07832). Code to be uploaded soon.


## Datasets download
We rely heavily on the :
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

### While discussing the Mammo-results, discuss the preprocessing of the report.
discuss about this file: /restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset/External/UPMC/upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv

### Discuss about the sentence path for RSNA and VinDr
--language_emb_path="<save_path for report-embeddings>/sent_emb_word_ge_3.npy" \
--sent_path="<save_path for reports>/sentences_word_ge_3.pkl" \
  
### Use vertex cloud for NIH-CXR:
https://console.cloud.google.com/vertex-ai/studio/chat?project=gen-lang-client-0586677956