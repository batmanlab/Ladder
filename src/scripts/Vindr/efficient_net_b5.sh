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


