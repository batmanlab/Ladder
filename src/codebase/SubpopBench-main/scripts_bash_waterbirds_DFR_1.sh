#!/bin/sh
#SBATCH --output=/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/psc_logs/subpopbench/waterbirds_%j.out

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")

echo $CURRENT

slurm_output_train1=/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/psc_logs/subpopbench/waterbirds_seed0_$CURRENT.out
slurm_output_train2=/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/psc_logs/subpopbench/waterbirds_seed1_$CURRENT.out
slurm_output_train3=/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/psc_logs/subpopbench/waterbirds_seed2_$CURRENT.out

echo "Save image reps"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate /restricted/projectnb/batmanlab/shawn24/breast_clip_rtx_6000

# done
python /restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "DFR" \
       --dataset "Waterbirds" \
       --train_attr no \
       --data_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data" \
       --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds" \
       --output_folder_name "vit_sup_in1k" \
       --image_arch "vit_sup_in1k" >$slurm_output_train1

# done
python /restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "DFR" \
       --dataset "Waterbirds" \
       --train_attr no \
       --data_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data" \
       --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds" \
       --output_folder_name "resnet_sup_in1k" \
       --image_arch "resnet_sup_in1k"

# redo
python /restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "DFR" \
       --dataset "Waterbirds" \
       --train_attr yes \
       --data_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data" \
       --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds" \
       --output_folder_name "resnet_sup_in21k" \
       --image_arch "resnet_sup_in21k"

# done
python /restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "DFR" \
       --dataset "Waterbirds" \
       --train_attr yes \
       --data_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data" \
       --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds" \
       --output_folder_name "resnet_simclr_in1k" \
       --image_arch "resnet_simclr_in1k"

# redo
python /restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "DFR" \
       --dataset "Waterbirds" \
       --train_attr yes \
       --data_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data" \
       --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds" \
       --output_folder_name "resnet_barlow_in1k" \
       --image_arch "resnet_barlow_in1k"

python /restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "DFR" \
       --dataset "Waterbirds" \
       --train_attr yes \
       --data_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data" \
       --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/Waterbirds" \
       --output_folder_name "resnet_dino_in1k" \
       --image_arch "resnet_dino_in1k"










