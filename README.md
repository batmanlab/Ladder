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

## 📌 TL;DR
**LADDER** is a modular framework that uses large language models (LLMs) to discover, explain, and mitigate hidden biases in vision classifiers—**without requiring prior knowledge of the biases or attribute labels**.

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
> ⚠️ **Warnings**
>
> - 🔧 Replace all hardcoded paths like `/restricted/projectnb/batmanlab/shawn24/PhD` with your **own directory**.
>
> - Following guidelines of [MIMIC-CXR] (https://physionet.org/news/post/gpt-responsible-use), we setup [google vertex ai](https://console.cloud.google.com/vertex-ai/studio/chat?project=gen-lang-client-0586677956) for setting GEMINI as LLM for hypothesis
> generation for medical images in this codebase. 
> - 📅 All LLMs were evaluated using checkpoints available **before Jan 11, 2025**.  
>   Newer versions may produce **different hypotheses** than those reported in the paper.
>
> - 🧠 Default setup uses:
>   -  **GPT-4o** as captioner for the natural images
>   - **ResNet-50** for the classifier
>   - **ViT-B/32** for the vision-language representation space
>   - **GPT-4o** for hypothesis generation  
>   The code is modular and can be **easily extended** to other models and LLMs.
>
> - 🗂️ Update cache directory locations in `save_img_reps.py` to your own:
>   ```python
>   os.environ['TRANSFORMERS_CACHE'] = '/your/custom/.cache/huggingface/transformers'
>   os.environ['TORCH_HOME'] = '/your/custom/.cache/torch'
>   ```

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

## 🛠️ Environment Setup

Use [environment.yaml](https://github.com/batmanlab/Ladder/blob/main/environment.yaml) to setup the environment.

## 📚 Dataset zoo

Please refer to the [dataset_zoo.md](dataset_zoo.md) for the details of the datasets used in this project.
**For toy dataset in Fig. 1, run the python [script](./src/codebase/toy_dataset/dataset_creation_new.py)**.

## 🧠 Model zoo

For the details of the classifiers, pretraining methods and algorithms supported by this codebase, refer to
the [classifier_zoo.md](classifier_zoo.md).

### 💾 Downloading Classifier Checkpoints Used in the Paper

We provide the pretrained **ResNet-50 (`resnet_sup_in1k`)** and **EfficientNet-B5 (`tf_efficientnet_b5_ns-detect`)**
classifier checkpoints used in our experiments
via [Hugging Face Hub](https://huggingface.co/shawn24/Ladder/tree/main/out).

#### 📦 Available Checkpoints by Dataset:

- 🐦 [Waterbirds (ResNet-50)](https://huggingface.co/shawn24/Ladder/blob/main/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/model.pkl)
- 👤 [CelebA (ResNet-50)](https://huggingface.co/shawn24/Ladder/blob/main/out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed0/model.pkl)
- 🐶 [MetaShift (ResNet-50)](https://huggingface.co/shawn24/Ladder/tree/main/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed0/model.pkl)
- 🫁 [NIH ChestX-ray (ResNet-50)](https://huggingface.co/shawn24/Ladder/blob/main/out/NIH/resnet50/seed0/chk_pt-best-auc0.8674.pt)
- 🧪 [RSNA-Mammo (EfficientNet-B5)](https://huggingface.co/shawn24/Ladder/blob/main/out/RSNA/fold0/b5-model-best-epoch-7.tar)
- 🏥 [VinDr-Mammo (EfficientNet-B5)](https://huggingface.co/shawn24/Ladder/blob/main/out/ViNDr/fold0/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth)

## 🤖 Vision-language representation space

We use the following vision-language representation space for our experiments:

- Natual images: [CLIP](https://github.com/openai/CLIP)
- Mammograms: [Mammo-CLIP](https://github.com/batmanlab/Mammo-CLIP)
- Chest-X-Rays: [CXR-CLIP](https://github.com/Soombit-ai/cxr-clip)

Download the latest checkpoints from the respective repositories.

## 💬 Generating captions

### 🌄 For Natural Images

Ladder requires captions for the images in the dataset. We provide a script to generate captions for the images using
BLIP and GPT-4o.

### Using BLIP

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

### Using GPT-4o

```bash
python ./src/codebase/caption_images_gpt_4.py \
  --seed=0 \
  --dataset="Waterbirds" \
  --img-path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/waterbird_complete95_forest2water2" \
  --csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/metadata_waterbirds.csv" \
  --save_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/va_metadata_waterbirds_captioning_GPT.csv" \
  --split="va" \
  --model="gpt-4o" \
  --api_key="<open-ai key>"
```

### 🫁 For Medical Images

- For NIH-CXR, we use the radiology report from [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) dataset.
  Download the metadata csv containing impression and
  findings from [here](https://huggingface.co/shawn24/Ladder/blob/main/Data/NIH/mimic-cxr-chexpert.csv).
- For RSNA-Mammo and VinDr-Mammo, we use the radiology text from [Mammo-FActOR](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/breastclip/data/datasets/prompts.json) codebase.
## 🪜 LADDER Pipeline

Ladder pipeline consists of 6 steps. We uploaded the outputs of every step in
the [huggingface](https://huggingface.co/shawn24/Ladder/tree/main/out). The steps are as follows:

### 🔁 Pipeline Overview

### Step1: Save image representations of the image classifier and vision encoder from vision language representation space

```bash
python ./src/codebase/save_img_reps.py \
  --seed=0 \
  --dataset="Waterbirds" \
  --classifier="resnet_sup_in1k" \
  --classifier_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/model.pkl" \
  --flattening-type="adaptive" \
  --clip_vision_encoder="ViT-B/32" \
  --data_dir="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}"
```

### Step2: Save text representations text encoder from vision language representation space

```bash
python ./src/codebase/save_text_reps.py \
  --seed=0 \
  --dataset="Waterbirds" \
  --clip_vision_encoder="ViT-B/32" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}" \
  --prompt_sent_type="captioning" \
  --captioning_type="gpt-4o" \
  --prompt_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/va_metadata_waterbirds_captioning_GPT.csv"
```

### Step3: Train aligner to align the classifier and vision language image representations

```bash
python ./src/codebase/learn_aligner.py \
  --seed=0 \
  --epochs=30 \
  --dataset="Waterbirds" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32" \
  --clf_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32/{1}_classifier_embeddings.npy" \
  --clip_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32/{1}_clip_embeddings.npy"
```

### Step4: Retrieving sentences indicative of biases

```bash
python ./src/codebase/discover_error_slices.py \
  --seed=0 \
  --topKsent=200 \
  --dataset="Waterbirds" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/test_additional_info.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy" \
  --language_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/sent_emb_captions_gpt-4o.npy" \
  --sent_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/sentences_captions_gpt-4o.pkl" \
  --aligner_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/aligner_30.pth"
```

### Step5: Discovering error slices via LLM-driven hypothesis generation

```bash
python ./src/codebase/validate_error_slices_w_LLM.py \
  --seed=0 \
  --LLM="gpt-4o" \
  --dataset="Waterbirds" \
  --class_label="landbirds" \
  --clip_vision_encoder="ViT-B/32" \
  --key="<open-ai key>" \
  --top50-err-text="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/landbirds_error_top_200_sent_diff_emb.txt" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_additional_info.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_classifier_embeddings.npy" \
  --aligner_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/aligner_30.pth"
```

### Step6: Mitigate multi-bias w/o annotation

```bash
python ./src/codebase/mitigate_error_slices.py \
  --seed=0 \
  --epochs=9 \
  --lr=0.001 \
  --weight_decay=0.0001 \
  --n=600 \
  --mode="last_layer_finetune" \
  --dataset="Waterbirds" \
  --classifier="resnet_sup_in1k" \
  --slice_names="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_prompt_dict.pkl" \
  --classifier_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/model.pkl" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_{}_dataframe_mitigation.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_classifier_embeddings.npy"

```

## ➡️ Demo notebooks with qualitative results

Refer to [notebook1](./src/codebase/demo/Demo-NIH.ipynb) and [notebook2](./src/codebase/demo/Demo-Waterbirds.ipynb) for
qualitative results of the error slices discovered by LADDER.

## 📜 Scripts to replicate the experiments of Ladder pipeline
We provide runnable shell scripts to replicate the full LADDER pipeline across all datasets:

 - [Waterbirds](./src/scripts/waterbirds/resnet_sup_in1k.sh)
 - [CelebA](./src/scripts/waterbirds/resnet_sup_in1k.sh)
 - [Metashift](./src/scripts/waterbirds/resnet_sup_in1k.sh)
 - [NIH-CXR](./src/scripts/waterbirds/resnet_sup_in1k.sh)
 - [RSNA-Mammo](./src/scripts/waterbirds/resnet_sup_in1k.sh)
 - [VinDr-Mammo](./src/scripts/waterbirds/resnet_sup_in1k.sh)

## 📖 Citation
If you find this work useful, please cite our paper:
```bibtex
@article{ghosh2024ladder,
  title={LADDER: Language Driven Slice Discovery and Error Rectification},
  author={Ghosh, Shantanu and Syed, Rayan and Wang, Chenyu and Poynton, Clare B and Visweswaran, Shyam and Batmanghelich, Kayhan},
  journal={arXiv preprint arXiv:2408.07832},
  year={2024}
}
```

## License and copyright

Licensed under the Creative Commons Attribution 4.0 International

Copyright © [Batman Lab](https://www.batman-lab.com/), 2025

## Contact

For any queries, contact [Shantanu Ghosh](https://shantanu-ai.github.io/) (email: **shawn24@bu.edu**)


## Contributing

Did you try some other classifier on a new dataset and want to report the results? Feel free to send
a [pull request](https://github.com/batmanlab/Ladder/pulls).