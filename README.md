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
> - Look for `/restricted/projectnb/batmanlab/shawn24/PhD` and replace it with your own path
> - All LLMs were evaluated using checkpoints available **before Jan 11, 2025**.  
    > With the release of newer versions, we **cannot guarantee** the same hypotheses will be generated as reported in
    the paper.
> - By default, we demonstrate results using **ResNet-50** (classifier), **ViT-B/32** (as the vision-language
    representation space), and **GPT-4o** (for hypothesis generation).  
    > The code is modular and can be **easily extended** to other models and LLMs.
---
### 🙏 Acknowledgements

We rely heavily on the [Subpopulation Shift Benchmark (SubpopBench)](https://github.com/YyzHarry/SubpopBench) codebase
for:

- 📥 Downloading and processing datasets
- 🧠 Classifier training on natural image benchmarks

Note: **SubpopBench does not support medical imaging datasets**.  
To address this, our codebase includes extended experiments for **NIH ChestX-ray (NIH-CXR)**, which are discussed in
subsequent sections.
---

## Environment Setup

Use [environment.yaml](https://github.com/batmanlab/Ladder/blob/main/environment.yaml) to setup the environment.

## Dataset zoo

Please refer to the [dataset_zoo.md](dataset_zoo.md) for the details of the datasets used in this project.

## Classifier zoo
For the details of the classifiers used in this project, please refer to the [classifier_zoo.md](classifier_zoo.md).
### 💾 Downloading Classifier Checkpoints Used in the Paper

We provide the pretrained **ResNet-50 (`resnet_sup_in1k`)** and **EfficientNet-B5 (`tf_efficientnet_b5_ns-detect`)** classifier checkpoints used in our experiments via [Hugging Face Hub](https://huggingface.co/shawn24/Ladder/tree/main/out).

#### 📦 Available Checkpoints by Dataset:

- 🐦 **[Waterbirds (ResNet-50)](https://huggingface.co/shawn24/Ladder/blob/main/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/model.pkl)**
- 👤 **[CelebA (ResNet-50)](https://huggingface.co/shawn24/Ladder/blob/main/out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed0/model.pkl)**
- 🐶 **[MetaShift (ResNet-50)](https://huggingface.co/shawn24/Ladder/tree/main/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed0/model.pkl)**
- 🫁 **[NIH ChestX-ray (ResNet-50)](https://huggingface.co/shawn24/Ladder/blob/main/out/NIH/resnet50/seed0/chk_pt-best-auc0.8674.pt)**
- 🧪 **[RSNA-Mammo (EfficientNet-B5)](https://huggingface.co/shawn24/Ladder/blob/main/out/RSNA/fold0/b5-model-best-epoch-7.tar)**
- 🏥 **[VinDr-Mammo (EfficientNet-B5)](https://huggingface.co/shawn24/Ladder/blob/main/out/ViNDr/fold0/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth)**


### Model Architectures & Pretraining Methods

The [supported image architectures](./src/codebase/SubpopBench-main/subpopbench/models/networks.py) are:

* ResNet-50 on ImageNet-1K using supervised pretraining (`resnet_sup_in1k`)
* ResNet-50 on ImageNet-21K using supervised pretraining (`resnet_sup_in21k`, [Ridnik et al., 2021](https://arxiv.org/pdf/2104.10972v4.pdf))
* ResNet-50 on ImageNet-1K using SimCLR (`resnet_simclr_in1k`, [Chen et al., 2020](https://arxiv.org/abs/2002.05709))
* ResNet-50 on ImageNet-1K using Barlow Twins (`resnet_barlow_in1k`, [Zbontar et al., 2021](https://arxiv.org/abs/2103.03230))
* ResNet-50 on ImageNet-1K using DINO (`resnet_dino_in1k`, [Caron et al., 2021](https://arxiv.org/abs/2104.14294))
* ViT-B on ImageNet-1K using supervised pretraining (`vit_sup_in1k`, [Steiner et al., 2021](https://arxiv.org/abs/2106.10270))
* ViT-B on ImageNet-21K using supervised pretraining (`vit_sup_in21k`, [Steiner et al., 2021](https://arxiv.org/abs/2106.10270))
* ViT-B from OpenAI CLIP (`vit_clip_oai`, [Radford et al., 2021](https://arxiv.org/abs/2103.00020))
* ViT-B pretrained using CLIP on LAION-2B (`vit_clip_laion`, [OpenCLIP](https://github.com/mlfoundations/open_clip))
* ViT-B on SWAG using weakly supervised pretraining (`vit_sup_swag`, [Singh et al., 2022](https://arxiv.org/abs/2201.08371))
* ViT-B on ImageNet-1K using DINO (`vit_dino_in1k`, [Caron et al., 2021](https://arxiv.org/abs/2104.14294))



Official repository of [Ladder](https://arxiv.org/abs/2408.07832). Code to be uploaded soon.

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

discuss about this file:
/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset/External/UPMC/upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv

### Discuss about the sentence path for RSNA and VinDr

--language_emb_path="<save_path for report-embeddings>/sent_emb_word_ge_3.npy" \
--sent_path="<save_path for reports>/sentences_word_ge_3.pkl" \

### Use vertex cloud for NIH-CXR:

https://console.cloud.google.com/vertex-ai/studio/chat?project=gen-lang-client-0586677956