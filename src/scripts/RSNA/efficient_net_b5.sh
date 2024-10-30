# Step 1: Create blackbox model using EfficientNet B5 architecture on RSNA dataset
python ./src/codebase/train_classifier_Mammo.py \
  --data-dir '/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset/' \
  --img-dir 'RSNA_Cancer_Detection/train_images_png' \
  --csv-file 'RSNA_Cancer_Detection/train_folds.csv' --start-fold 0 --n_folds 1 \
  --dataset 'RSNA' --arch 'tf_efficientnet_b5_ns-detect' --epochs 9 --batch-size 6 --num-workers 0 \
  --print-freq 10000 --log-freq 500 --running-interactive 'n' \
  --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n' \
  --tensorboard-path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0" \
  --checkpoints="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0" \
  --output_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0" \
  --label "cancer"



# Step 2: Save Image Reps EN B5 valid
python ./src/codebase/save_img_reps.py \
  --seed=0 \
  --dataset="RSNA" \
  --classifier="efficientnet-b5" \
  --classifier_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/efficientnetb5_seed_10_best_aucroc0.89_ver084.pth" \
  --flattening-type="adaptive" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --clip_check_pt "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0/b5-model-best-epoch-7.tar" \
  --data_dir="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset/RSNA_Cancer_Detection" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89" \
  --tokenizers="/restricted/projectnb/batmanlab/shawn24/PhD" \
  --cache_dir="/restricted/projectnb/batmanlab/shawn24/PhD"



# Step 3: Save Text Reps EN B5 valid
python ./src/codebase/save_text_reps.py \
  --seed=0 \
  --dataset="RSNA" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --clip_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0/b5-model-best-epoch-7.tar" \
  --csv="<csv file containing report texts>" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89" \
  --tokenizers="/restricted/projectnb/batmanlab/shawn24/PhD" \
  --cache_dir="/restricted/projectnb/batmanlab/shawn24/PhD"



# Step 4: Train aligner with valid
python ./src/codebase/learn_aligner.py \
  --seed=0 \
  --epochs=30 \
  --dataset="RSNA" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{0}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{0}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_classifier_embeddings.npy" \
  --clip_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{0}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{1}_clip_embeddings.npy"


# Step 5: Discover error slices valid
python ./src/codebase/discover_error_slices.py \
  --seed=0 \
  --topKsent=100 \
  --dataset="RSNA" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_additional_info.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/valid_classifier_embeddings.npy" \
  --language_emb_path="<save_path for report-embeddings>/sent_emb_word_ge_3.npy" \
  --sent_path="<save_path for reports>/sentences_word_ge_3.pkl" \
  --aligner_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth"



# Step 6: Validate error slices valid
python ./src/codebase/validate_error_slices_w_LLM.py \
  --seed=0 \
  --dataset="RSNA" \
  --class_label="cancer" \
  --clip_vision_encoder="tf_efficientnet_b5_ns-detect" \
  --key="<open-ai key>" \
  --clip_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0/b5-model-best-epoch-7.tar" \
  --top50-err-text="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/cancer_error_top_100_sent_diff_emb.txt" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_additional_info.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy" \
  --aligner_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/aligner_30.pth" \
  --tokenizers="/restricted/projectnb/batmanlab/shawn24/PhD" \
  --cache_dir="/restricted/projectnb/batmanlab/shawn24/PhD"


# Step 7: Mitigate error slices valid
python ./src/codebase/mitigate_error_slices.py \
  --seed=0 \
  --epochs=30 \
  --n=75 \
  --mode="last_layer_finetune" \
  --dataset="RSNA" \
  --classifier="efficientnet-b5" \
  --slice_names="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/cancer_prompt_dict.pkl" \
  --classifier_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/efficientnetb5_seed_10_best_aucroc0.89_ver084.pth" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_cancer_dataframe_mitigation.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold{}/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/{}_classifier_embeddings.npy"
