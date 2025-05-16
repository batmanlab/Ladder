## 📚 Table of Contents (Classifiers Section)

- [⚙️ Training Algorithms](#️-training-algorithms)
- [🧠 Model Architectures & Pretraining](#-model-architectures--pretraining)
- [🧪 Training Scripts for Different Classifiers](#-training-scripts-for-different-classifiers)
  - [CelebA](#-celebA)
  - [MetaShift](#-metashift)
  - [NIH ChestX-ray (via SubpopBench)](#-nih-chestx-ray-using-subpopshift)
  - [Waterbirds](#-waterbirds)
  - [🏥 Training ResNet50 on NIH-CXR](#-training-resnet50-using-erm-resnet_sup_in1k-as-classifier-for-nih-cxr)
  - [🏥 Training EN-B5 on RSNA-Mammo](#-training-efficientnet-b5-en-b5-using-erm-as-classifier--for-rsna-mammo)
  - [🏥 Training EN-B5 on VinDr-Mammo](#-training-efficientnet-b5-en-b5--as-classifier--for-vindr-mammo)

### ⚙️ Training Algorithms

We support a wide range of algorithms for bias mitigation and robust training.  
[Click here to view the implementation](./src/codebase/SubpopBench-main/subpopbench/learning/algorithms.py).

- **ERM** — Empirical Risk Minimization ([Vapnik, 1998](https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034))
- **IRM** — Invariant Risk Minimization ([Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893))
- **GroupDRO** — Group Distributionally Robust Optimization ([Sagawa et al., 2020](https://arxiv.org/abs/1911.08731))
- **CVaRDRO** — Conditional Value-at-Risk DRO ([Duchi & Namkoong, 2018](https://arxiv.org/abs/1810.08750))
- **Mixup** ([Zhang et al., 2018](https://arxiv.org/abs/1710.09412))
- **JTT** — Just Train Twice ([Liu et al., 2021](http://proceedings.mlr.press/v139/liu21f.html))
- **LfF** — Learning from Failure ([Nam et al., 2020](https://proceedings.neurips.cc/paper/2020/file/eddc3427c5d77843c2253f1e799fe933-Paper.pdf))
- **LISA** — Learning Invariant Selective Augmentation ([Yao et al., 2022](https://arxiv.org/abs/2201.00299))
- **DFR** — Deep Feature Reweighting ([Kirichenko et al., 2022](https://arxiv.org/abs/2204.02937))
- **MMD** — Maximum Mean Discrepancy ([Li et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf))
- **CORAL** — Deep Correlation Alignment ([Sun & Saenko, 2016](https://arxiv.org/abs/1607.01719))
- **ReSample** — Data Re-Sampling ([Japkowicz, 2000](https://site.uottawa.ca/~nat/Papers/ic-ai-2000.ps))
- **ReWeight** — Cost-Sensitive Re-Weighting ([Japkowicz, 2000](https://site.uottawa.ca/~nat/Papers/ic-ai-2000.ps))
- **SqrtReWeight** — Square-Root Reweighting ([Japkowicz, 2000](https://site.uottawa.ca/~nat/Papers/ic-ai-2000.ps))
- **Focal** — Focal Loss ([Lin et al., 2017](https://arxiv.org/abs/1708.02002))
- **CBLoss** — Class-Balanced Loss ([Cui et al., 2019](https://arxiv.org/abs/1901.05555))
- **LDAM** — Label-Distribution-Aware Margin Loss ([Cao et al., 2019](https://arxiv.org/abs/1906.07413))
- **BSoftmax** — Balanced Softmax ([Ren et al., 2020](https://arxiv.org/abs/2007.10740))
- **CRT** — Classifier Re-Training ([Kang et al., 2020](https://arxiv.org/abs/1910.09217))

---

### 🧠 Model Architectures & Pretraining

[Supported image model architectures are implemented here](./src/codebase/SubpopBench-main/subpopbench/models/networks.py).

**ResNet-50 Variants**
- `resnet_sup_in1k` — Supervised on ImageNet-1K
- `resnet_sup_in21k` — Supervised on ImageNet-21K ([Ridnik et al., 2021](https://arxiv.org/pdf/2104.10972v4.pdf))
- `resnet_simclr_in1k` — SimCLR ([Chen et al., 2020](https://arxiv.org/abs/2002.05709))
- `resnet_barlow_in1k` — Barlow Twins ([Zbontar et al., 2021](https://arxiv.org/abs/2103.03230))
- `resnet_dino_in1k` — DINO ([Caron et al., 2021](https://arxiv.org/abs/2104.14294))

**ViT-B Variants**
- `vit_sup_in1k` — Supervised on ImageNet-1K ([Steiner et al., 2021](https://arxiv.org/abs/2106.10270))
- `vit_sup_in21k` — Supervised on ImageNet-21K ([Steiner et al., 2021](https://arxiv.org/abs/2106.10270))
- `vit_clip_oai` — CLIP (OpenAI) ([Radford et al., 2021](https://arxiv.org/abs/2103.00020))
- `vit_clip_laion` — OpenCLIP on LAION-2B ([OpenCLIP](https://github.com/mlfoundations/open_clip))
- `vit_sup_swag` — Supervised on SWAG ([Singh et al., 2022](https://arxiv.org/abs/2201.08371))
- `vit_dino_in1k` — DINO ([Caron et al., 2021](https://arxiv.org/abs/2104.14294))

---

### 🧪 Training scripts for different classifiers

Training scripts for different datasets are available in the repo:

- 📷 [CelebA](./src/codebase/SubpopBench-main/scripts_bash_celebA.sh)
- 🐶 [MetaShift](./src/codebase/SubpopBench-main/scripts_bash_metashift.sh)
- 🫁 [NIH ChestX-ray using subpopshift](./src/codebase/SubpopBench-main/scripts_bash_nih.sh)
- 🐦 [Waterbirds](./src/codebase/SubpopBench-main/scripts_bash_waterbirds.sh)

### 🏥 Training ResNet50 using ERM (`resnet_sup_in1k`) as classifier for NIH-CXR 
We train the ERM variant for NIH-CXR as follows:
```bash
python ./src/codebase/train_classifier_CXR.py --img-size 224 --arch ResNet50 --lr 1e-5
```

### 🏥 Training EfficientNet-B5 (`EN-B5`) using ERM as classifier  for RSNA-Mammo
We train the ERM variant for RSNA-Mammo as follows:
```bash
python ./src/codebase/train_classifier_Mammo.py \
  --data-dir '/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset/' \
  --img-dir 'RSNA_Cancer_Detection/train_images_png' \
  --csv-file 'RSNA_Cancer_Detection/rsna_w_upmc_concepts_breast_clip.csv' --start-fold 0 --n_folds 1 \
  --dataset 'RSNA' --arch 'tf_efficientnet_b5_ns-detect' --epochs 9 --batch-size 6 --num-workers 0 \
  --print-freq 10000 --log-freq 500 --running-interactive 'n' \
  --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n' \
  --tensorboard-path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0" \
  --checkpoints="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0" \
  --output_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0" \
  --label "cancer"
```

### 🏥 Training EfficientNet-B5 (`EN-B5`)  as classifier  for VinDr-Mammo
We train the ERM variant for VinDr-Mammo as follows:
```bash
python ./src/codebase/train_classifier_Mammo.py \
  --data-dir '/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset' \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds_abnormal.csv' \
  --dataset 'ViNDr' --arch 'tf_efficientnet_b5_ns-detect' --epochs 20 --batch-size 8 --num-workers 0 \
  --print-freq 10000 --log-freq 500 --running-interactive 'n' \
  --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n'  --n_folds 1  --label "abnormal" \
  --tensorboard-path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold0" \
  --checkpoints="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold0" \
  --output_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold0"

```