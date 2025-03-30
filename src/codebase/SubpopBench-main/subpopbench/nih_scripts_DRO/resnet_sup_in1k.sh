#!/bin/sh
#SBATCH --output=/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/psc_logs/subpopbench/nih-GroupDRO/nih-resnet_sup_in1k-%j.out

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")

echo $CURRENT

slurm_output_train1=/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/psc_logs/subpopbench/nih-GroupDRO/nih-resnet_sup_in1k-$CURRENT.out

echo "Save image reps"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate /restricted/projectnb/batmanlab/shawn24/breast_clip_rtx_6000



python /restricted/projectnb/batmanlab/shawn24/PhD/Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "GroupDRO" \
       --dataset "NIH_dataset" \
       --train_attr yes \
       --data_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/data" \
       --output_dir "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH_Cxrclip" \
       --output_folder_name "resnet_sup_in1k" \
       --image_arch "resnet_sup_in1k" \
       --es_metric overall:AUROC --use_es >$slurm_output_train1



