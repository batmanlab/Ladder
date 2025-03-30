python -m subpopbench.train \
--dataset NIH_dataset \
--data_dir  /restricted/projectnb/batmanlab/chyuwang/Subpop_data/subpopbench/third_party/ \
--train_attr no --algorithm <algo> --output_dir /restricted/projectnb/batmanlab/chyuwang/Subpop_data/subpopbench/output/<date>/ \
--es_metric overall:AUROC --use_es --seed <num>



python /restricted/projectnb/batmanlab/shawn24/PhD/Subpop_data/subpopbench/train.py \
--dataset NIH_dataset \
--data_dir  /restricted/projectnb/batmanlab/chyuwang/Subpop_data/subpopbench/third_party/ \
--train_attr yes --algorithm ERM --output_dir /restricted/projectnb/batmanlab/shawn24/PhD/Subpop_data/subpopbench/output/resnet_simclr_in1k/ \
--image_arch "resnet_simclr_in1k" \
--es_metric overall:AUROC --use_es --seed 0



python /restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "ERM" \
       --dataset "NIH_dataset" \
       --train_attr yes \
       --data_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/SubpopBench-main/third_party" \
       --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH_Cxrclip" \
       --output_folder_name "resnet_simclr_in1k" \
       --image_arch "resnet_simclr_in1k"
