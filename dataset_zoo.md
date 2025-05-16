### 📁 Dataset Directory Structure
We follow the directory structure below for the datasets used in this project. The datasets are organized into subdirectories, each containing the necessary files for training and evaluation.
```bash
data/
├── celeba/
│   ├── img_align_celeba/
│   ├── list_attr_celeba.csv
│   ├── list_bbox_celeba.csv
│   ├── list_eval_partition.csv
│   ├── list_landmarks_align_celeba.csv
│   ├── metadata_celeba.csv
│   ├── va_metadata_celeba_captioning_blip.csv
│   └── va_metadata_celeba_captioning_GPT.csv
├── metashift/
│   ├── metadata_metashift.csv
│   ├── metadata_metashift_captioning.csv
│   ├── te_metadata_metashift_captioning.csv
│   ├── va_metadata_metashift_captioning_blip.csv
│   ├── va_metadata_metashift_captioning_gpt.csv
│   ├── va_metadata_metashift_captioning_GPT.csv
│   └── MetaShift-Cat-Dog-indoor-outdoor/
│       ├── train/
│       └── test/
├── nih/
│   ├── mimic-cxr-chexpert.csv
│   └── nih_processed_v2.csv
├── RSNA_Cancer_Detection/
│   └── rsna_w_upmc_concepts_breast_clip.csv
├── Vindr/
│   └── vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/
│       ├── breast-level_annotations.csv
│       ├── finding_annotations.csv
│       └── vindr_detection_v1_folds_abnormal.csv
└── waterbirds/
    ├── metadata_waterbirds.csv
    ├── va_metadata_waterbirds_captioning_blip.csv
    ├── va_metadata_waterbirds_captioning_GPT.csv
    └── waterbird_complete95_forest2water2/
        ├── 001.Black_footed_Albatross/
        ├── 002.Laysan_Albatross/
        ├── 003.Sooty_Albatross/
        ├── ...
        ├── 200.Common_Yellowthroat/
        └── metadata.csv
```

### Datasets download
We rely heavily on the [Subpopulation Shift Benchmark
](https://github.com/YyzHarry/SubpopBench) codebase for downloading and processing the datasets. We included the necessary code changes in `src/codebase/SubpopBench-main` to ensure compatibility with our project.:
- Waterbids and Metashift
```bash
python ./src/codebase/SubpopBench-main/subpopbench/scripts/download.py \
--datasets "waterbirds" "metashift" \
--data_path "Ladder/data/new" \
--download True

```
 CelebA - [url](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [NIH](https://www.kaggle.com/datasets/nih-chest-xrays/data), 
- RSNA-Mammo, VinDr-Mammo
- Mention metadata files for them