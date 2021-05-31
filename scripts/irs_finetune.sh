#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 400\
                --batch_size 2\
                --checkpoint irs_ft\
                --num_workers 2\
                --dataset irs\
                --dataset_directory PATH_TO_IRS\
                --ft\
                --resume sceneflow_pretrained_model.pth.tar