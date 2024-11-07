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


# Step 3: Save Text Reps EN B5 valid
# Same as RSNA

# Step 4: Train aligner
python ./src/codebase/learn_aligner.py \
  --seed=0 \
  --epochs=30 \
  --dataset="VinDr" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_classifier_embeddings.npy" \
  --clip_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{0}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_clip_embeddings.npy"


# Step 5: Discover error slices valid
python ./src/codebase/discover_error_slices.py \
  --seed=0 \
  --topKsent=100 \
  --dataset="ViNDr" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_additional_info.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_classifier_embeddings.npy" \
  --language_emb_path="<save_path for report-embeddings>/sent_emb_word_ge_3.npy" \
  --sent_path="<save_path for reports>/sentences_word_ge_3.pkl" \
  --aligner_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth"


# Step 6: Validate error slices valid
python ./src/codebase/validate_error_slices_w_LLM.py \
  --seed=0 \
  --dataset="ViNDr" \
  --class_label="abnormal" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --key="sk-proj-QZNJ5JnRi76V4FTYzVRxT3BlbkFJajHNswRDrX19Z8GgD1el" \
  --clip_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0/b5-model-best-epoch-7.tar" \
  --top50-err-text="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/abnormal_error_top_100_sent_diff_emb.txt" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_additional_info.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy" \
  --aligner_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth" \
  --tokenizers="/restricted/projectnb/batmanlab/shawn24/PhD" \
  --cache_dir="/restricted/projectnb/batmanlab/shawn24/PhD"


# Step 7: Mitigate error slices valid
python ./src/codebase/mitigate_error_slices.py \
  --seed=0 \
  --epochs=30 \
  --n=75 \
  --mode="last_layer_finetune" \
  --dataset="ViNDr" \
  --classifier="efficientnet-b5" \
  --slice_names="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/abnormal_prompt_dict.pkl" \
  --classifier_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/efficientnetb5_seed_10_fold0_best_aucroc_ver084.pth" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_abnormal_dataframe_mitigation.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/ViNDr/fold{}/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy"
