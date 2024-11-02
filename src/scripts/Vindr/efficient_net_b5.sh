# Step 1: Create blackbox model using EfficientNet B5 architecture on VinDr dataset
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



# Step 2: Save Image Reps EN B5
python ./src/codebase/save_img_reps.py \
  --seed=0 \
  --dataset="VinDr" \
  --classifier="efficientnet-b5" \
  --classifier_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
  --flattening-type="adaptive" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --clip_check_pt "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0/b5-model-best-epoch-7.tar" \
  --data_dir="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset/External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}" \
  --tokenizers="/restricted/projectnb/batmanlab/shawn24/PhD" \
  --cache_dir="/restricted/projectnb/batmanlab/shawn24/PhD"


# Step 3: Train aligner
python /restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/src/codebase/learn_aligner.py \
  --seed=0 \
  --epochs=30 \
  --dataset="VinDr" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/ViNDr/Neurips/fold{0}/cancer/best/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/ViNDr/Neurips/fold{0}/cancer/best/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_classifier_embeddings.npy" \
  --clip_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out/ViNDr/Neurips/fold{0}/cancer/best/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_clip_embeddings.npy"

