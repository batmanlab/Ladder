# Create blackbox model using EfficientNet B5 architecture on RSNA dataset
python /ocean/projects/asc170022p/shg121/PhD/Multimodal-mistakes-debug/src/codebase/train_classifier_RSNA.py \
  --img-dir 'RSNA_Cancer_Detection/train_images_png' \
  --csv-file 'RSNA_Cancer_Detection/train_folds.csv' --start-fold 0 --n_folds 1 \
  --dataset 'RSNA' --arch 'tf_efficientnet_b5_ns-detect' --epochs 9 --batch-size 6 --num-workers 0 \
  --print-freq 10000 --log-freq 500 --running-interactive 'n' \
  --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n'