# Step 1a: Captioning using blip captioner
python ./src/codebase/caption_images.py \
  --seed=0 \
  --dataset="MetaShift" \
  --img-path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/metashift/MetaShift-Cat-Dog-indoor-outdoor" \
  --csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/metashift/metadata_metashift.csv" \
  --save_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/metashift/metadata_metashift_captioning_blip.csv" \
  --split="va" \
  --captioner="blip"


# Step 1b:Captioning using GPT-4o captioner
python ./src/codebase/caption_images_gpt_4.py \
  --seed=0 \
  --dataset="MetaShift" \
  --img-path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/metashift/MetaShift-Cat-Dog-indoor-outdoor" \
  --csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/metashift/metadata_metashift.csv" \
  --save_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/metashift/va_metadata_metashift_captioning_GPT.csv" \
  --split="va" \
  --api_key="<open-ai key>"


# Step 2:Save Image Reps ViT-B/32
python ./src/codebase/save_img_reps.py \
  --seed=0 \
  --dataset="MetaShift" \
  --classifier="resnet_sup_in1k" \
  --classifier_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/model.pkl" \
  --flattening-type="adaptive" \
  --clip_vision_encoder="ViT-B/32" \
  --data_dir="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}"


# Step 3:Save Text Reps ViT-B/32
python ./src/codebase/save_text_reps.py \
  --seed=0 \
  --dataset="MetaShift" \
  --clip_vision_encoder="ViT-B/32" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}" \
  --prompt_sent_type="captioning" \
  --captioning_type="gpt-4o" \
  --prompt_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/metashift/va_metadata_metashift_captioning_GPT.csv"

# Step 4:Train aligner
python ./src/codebase/learn_aligner.py \
  --seed=0 \
  --epochs=30 \
  --dataset="MetaShift" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32" \
  --clf_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32/{1}_classifier_embeddings.npy" \
  --clip_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{0}/clip_img_encoder_ViT-B/32/{1}_clip_embeddings.npy"


# Step 5: Discover error slices captions
python ./src/codebase/discover_error_slices.py \
  --seed=0 \
  --topKsent=50 \
  --dataset="MetaShift" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/test_additional_info.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy" \
  --language_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/sent_emb_captions_gpt-4o.npy" \
  --sent_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/sentences_captions_gpt-4o.pkl" \
  --aligner_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/aligner_30.pth"

# Step 6: Validate error slices captions
# class: dog
python ./src/codebase/validate_error_slices_w_LLM.py \
  --seed=0 \
  --dataset="MetaShift" \
  --class_label="dog" \
  --clip_vision_encoder="ViT-B/32" \
  --key="<open-ai key>" \
  --top50-err-text="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/dog_error_top_50_sent_diff_emb.txt" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_additional_info.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_classifier_embeddings.npy" \
  --aligner_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/aligner_30.pth"

# class: cat
python ./src/codebase/validate_error_slices_w_LLM.py \
  --seed=0 \
  --dataset="MetaShift" \
  --class_label="cat" \
  --clip_vision_encoder="ViT-B/32" \
  --key="<open-ai key>" \
  --top50-err-text="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/cat_error_top_50_sent_diff_emb.txt" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_additional_info.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/{}_classifier_embeddings.npy" \
  --aligner_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/MetaShift/resnet_sup_in1k_attrNo/MetaShift_ERM_hparams0_seed{}/clip_img_encoder_ViT-B/32/aligner_30.pth"

