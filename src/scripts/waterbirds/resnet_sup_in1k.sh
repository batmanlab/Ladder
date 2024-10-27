# Captioning using blip captioner
python ./src/codebase/caption_images.py \
  --seed=0 \
  --dataset="Waterbirds" \
  --img-path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/waterbird_complete95_forest2water2" \
  --csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/metadata_waterbirds.csv" \
  --save_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/va_metadata_waterbirds_captioning_blip.csv" \
  --split="va" \
  --captioner="blip"

# Captioning using GPT4 captioner
python ./src/codebase/caption_images_gpt_4.py \
  --seed=0 \
  --dataset="Waterbirds" \
  --img-path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/waterbird_complete95_forest2water2" \
  --csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/metadata_waterbirds.csv" \
  --save_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/va_metadata_waterbirds_captioning_GPT.csv" \
  --split="va" \
  --model="gpt-4o" \
  --api_key="xxxxx"

# Save Image Reps ViT-B/32
python ./src/codebase/save_img_reps.py \
  --seed=0 \
  --dataset="Waterbirds" \
  --classifier="resnet_sup_in1k" \
  --classifier_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/model.pkl" \
  --flattening-type="adaptive" \
  --clip_vision_encoder="ViT-B/32" \
  --data_dir="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}"


# Save Text Reps ViT-B/32 captioning
python ./src/codebase/save_text_reps.py \
  --seed=0 \
  --dataset="Waterbirds" \
  --clip_vision_encoder="ViT-B/32" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}" \
  --prompt_sent_type="captioning" \
  --captioning_type="gpt-4o" \
  --prompt_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/waterbirds/va_metadata_waterbirds_captioning_GPT.csv"


# Train aligner
python ./src/codebase/learn_aligner.py \
  --seed=0 \
  --epochs=30 \
  --dataset="Waterbirds" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32" \
  --clf_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32/{1}_classifier_embeddings.npy" \
  --clip_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32/{1}_clip_embeddings.npy"


# Discover error slices captions
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


# Validate error slices caption
# class: landbirds
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

# class: waterbirds
python ./src/codebase/validate_error_slices_w_LLM.py \
  --seed=0 \
  --LLM="gpt-4o" \
  --dataset="Waterbirds" \
  --class_label="waterbirds" \
  --clip_vision_encoder="ViT-B/32" \
  --key="<open-ai key>" \
  --top50-err-text="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/waterbirds_error_top_200_sent_diff_emb.txt" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_additional_info.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_classifier_embeddings.npy" \
  --aligner_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/aligner_30.pth"


# Mitigate error slices captions
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
