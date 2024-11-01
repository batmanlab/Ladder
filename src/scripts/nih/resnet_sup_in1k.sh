# Step 1: Create blackbox model using ResNet50 architecture on NIH dataset with supervised pretraining on ImageNet-1k
python ./src/codebase/train_classifier_CXR.py --img-size 224 --arch ResNet50 --lr 1e-5



#  Step 2: Save Image Reps swin mc
python ./src/codebase/save_img_reps.py \
  --seed=0 \
  --dataset="NIH" \
  --classifier="ResNet50" \
  --classifier_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/chk_pt-best-auc0.8674.pt" \
  --clip_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/swint_mc.tar" \
  --flattening-type="adaptive" \
  --clip_vision_encoder="swin-tiny-cxr-clip_mc" \
  --data_dir="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data/nih" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}" \
  --tokenizers="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/tokenizers/scc/huggingface/tokenizers" \
  --cache_dir="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/tokenizers/scc/huggingface/"

#  Step 3: Save Text Reps swin mc
python ./src/codebase/save_text_reps.py \
  --seed=0 \
  --report-word-ge=2 \
  --dataset="NIH" \
  --clip_vision_encoder="swin-tiny-cxr-clip_mc" \
  --clip_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/swint_mc.tar" \
  --csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed0/mimic-cxr-chexpert.csv" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}" \
  --tokenizers="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/tokenizers/scc/huggingface/tokenizers" \
  --cache_dir="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/tokenizers/scc/huggingface/"


# Step 4: Train aligner
python ./src/codebase/learn_aligner.py \
  --seed=0 \
  --epochs=200 \
  --lr=0.01 \
  --dataset="NIH" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{0}/clip_img_encoder_swin-tiny-cxr-clip_mc" \
  --clf_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{0}/clip_img_encoder_swin-tiny-cxr-clip_mc/{1}_classifier_embeddings.npy" \
  --clip_reps_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{0}/clip_img_encoder_swin-tiny-cxr-clip_mc/{1}_clip_embeddings.npy"


# Step 5: Discover error slices
python ./src/codebase/discover_error_slices.py \
  --seed=0 \
  --topKsent=100 \
  --dataset="NIH" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc/valid_additional_info.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc/valid_classifier_embeddings.npy" \
  --language_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc/sent_emb_word.npy" \
  --sent_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc/sentences.pkl" \
  --aligner_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc/aligner_200.pth"


# Step 6: Validate error slices
python ./src/codebase/validate_error_slices_w_LLM.py \
  --seed=0 \
  --dataset="NIH" \
  --LLM="gemini-vertex" \
  --class_label="pneumothorax" \
  --clip_vision_encoder="swin-tiny-cxr-clip_mc" \
  --key="<vertex-project>" \
  --clip_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/swint_mc.tar" \
  --top50-err-text="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc/pneumothorax_error_top_100_sent_diff_emb.txt" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc/{}_additional_info.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc/{}_classifier_embeddings.npy" \
  --aligner_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc/aligner_200.pth"\
  --tokenizers="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/tokenizers/scc/huggingface/tokenizers" \
  --cache_dir="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/tokenizers/scc/huggingface/"


# Step 7: Mitigate error slices
python ./src/codebase/mitigate_error_slices.py \
  --seed=0 \
  --epochs=20 \
  --n=120 \
  --mode="last_layer_finetune" \
  --dataset="NIH" \
  --classifier="ResNet50" \
  --slice_names="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed0/clip_img_encoder_swin-tiny-cxr-clip_mc/pneumothorax_prompt_dict.pkl" \
  --classifier_check_pt="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/chk_pt-best-auc0.8674.pt" \
  --save_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc" \
  --clf_results_csv="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc/{}_pneumothorax_dataframe_mitigation.csv" \
  --clf_image_emb_path="/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed{}/clip_img_encoder_swin-tiny-cxr-clip_mc/{}_classifier_embeddings.npy"
