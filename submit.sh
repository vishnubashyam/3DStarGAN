#! /bin/bash
#$ -l h_vmem=32G
#$ -l gpu=1

#cd mydir # local dir (directory with test.py)
cd /cbica/home/bashyamv/comp_space/1_Projects/20_Medical_Image_Harmonization/2DStarGAN/
#Load Virtual ENV
#source mydir/dlenv/bin/activate
source /cbica/home/bashyamv/ENV/torch_lightning_env/torch_lightning_env/bin/activate

python main.py \
--experiment 2DStarGAN  \
--experiment_tag 2DStarGAN \
--img_size 256 \
--latent_dim 16 \
--hidden_dim 1024 \
--style_dim 16 \
--lambda_reg 1 \
--lambda_cyc 1 \
--lambda_sty 1 \
--lambda_ds 1 \
--batch_size 32 \
--lr 0.0002 \
--num_outs_per_domain 5 \
--mode train \
--num_workers 1 \
--csv_path /cbica/home/bashyamv/comp_space/1_Projects/20_Medical_Image_Harmonization/2DStarGAN/Data/2D_Training_List.csv \
--img_dir /cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/Preprocessing/ \
--sample_every 1000 \
--log_every 500 \
--save_every 1000 \
--model_name 2dStarGAN \
--max_steps 200000 \
--device gpu \
--log_initial_debug_images 