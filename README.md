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

The [currently available algorithms](./subpopbench/learning/algorithms.py) are:

* Empirical Risk Minimization (**ERM**, [Vapnik, 1998](https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034))
* Invariant Risk Minimization (**IRM**, [Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893))
* Group Distributionally Robust Optimization (**GroupDRO**, [Sagawa et al., 2020](https://arxiv.org/abs/1911.08731))
* Conditional Value-at-Risk Distributionally Robust Optimization (**CVaRDRO**, [Duchi and Namkoong, 2018](https://arxiv.org/abs/1810.08750))
* Mixup (**Mixup**, [Zhang et al., 2018](https://arxiv.org/abs/1710.09412))
* Just Train Twice (**JTT**, [Liu et al., 2021](http://proceedings.mlr.press/v139/liu21f.html))
* Learning from Failure (**LfF**, [Nam et al., 2020](https://proceedings.neurips.cc/paper/2020/file/eddc3427c5d77843c2253f1e799fe933-Paper.pdf))
* Learning Invariant Predictors with Selective Augmentation (**LISA**, [Yao et al., 2022](https://arxiv.org/abs/2201.00299))
* Deep Feature Reweighting (**DFR**, [Kirichenko et al., 2022](https://arxiv.org/abs/2204.02937))
* Maximum Mean Discrepancy (**MMD**, [Li et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf))
* Deep Correlation Alignment (**CORAL**, [Sun and Saenko, 2016](https://arxiv.org/abs/1607.01719))
* Data Re-Sampling (**ReSample**, [Japkowicz, 2000](https://site.uottawa.ca/~nat/Papers/ic-ai-2000.ps))
* Cost-Sensitive Re-Weighting (**ReWeight**, [Japkowicz, 2000](https://site.uottawa.ca/~nat/Papers/ic-ai-2000.ps))
* Square-Root Re-Weighting (**SqrtReWeight**, [Japkowicz, 2000](https://site.uottawa.ca/~nat/Papers/ic-ai-2000.ps))
* Focal Loss (**Focal**, [Lin et al., 2017](https://arxiv.org/abs/1708.02002))
* Class-Balanced Loss (**CBLoss**, [Cui et al., 2019](https://arxiv.org/abs/1901.05555))
* Label-Distribution-Aware Margin Loss (**LDAM**, [Cao et al., 2019](https://arxiv.org/abs/1906.07413))
* Balanced Softmax (**BSoftmax**, [Ren et al., 2020](https://arxiv.org/abs/2007.10740))
* Classifier Re-Training (**CRT**, [Kang et al., 2020](https://arxiv.org/abs/1910.09217))


### Model Architectures & Pretraining Methods

The [supported image architectures](./subpopbench/models/networks.py) are:

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