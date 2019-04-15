import numpy as np
import torch
import torch.utils.data
import tqdm
import os
import argparse
import logging
from ConvKNRM.modules import ConvKNRM

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Train ConvKNRM")
	parser.add_argument("train_file", help="File that contains training data")
    parser.add_argument("dev_file", help="File that contains dev data")
    parser.add_argument("embedding_file", help="File that contains pre-trained embeddings")
	parser.add_argument('--seed', type=int, default=6, help='Random seed for the experiment')
    parser.add_argument('--epochs', type=int, default=20, help='Train data iterations')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--dev_batch_size', type=int, default=32, help='Batch size for dev')
    parser.add_argument('--pos_emb_size', type=int, default=50, help='Embedding size for pos tags')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for embedding layers')    
    parser.add_argument('--log_file', type=str, default="RMR.log", help='path to the log file')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--load_model', type=str, default="", help='path to the log file')
