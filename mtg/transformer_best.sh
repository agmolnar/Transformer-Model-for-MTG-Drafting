#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate transformer_full

cd /home/u571882/Projects/Transformer/mtg

python mtg/scripts/train_drafter.py --expansion_fname mtg/data/expansiontrain_full.pkl --model_name mtg/data/draft_model_base_best --batch_size 16 --emb_dim 128 --num_encoder_heads 16 --num_decoder_heads 16 --pointwise_ffn_width 256 --num_encoder_layers 2 --num_decoder_layers 1 --emb_dropout 0.1 --transformer_dropout 0.1 --lr_warmup 1000 --emb_margin 0.7 --emb_lambda 0.1 --epochs 1 --cardperf False --scryfall False