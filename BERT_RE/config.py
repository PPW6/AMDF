
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--bert_path", type=str, default="../BERT_model/CopperBERT")
parser.add_argument("--re_path", type=str, default="../BERT_model/CopperBERT_re")
parser.add_argument("--all_data", type=str, default='./data/all_data.txt')
parser.add_argument("--train_file", type=str, default='./data/train_mat.json')
parser.add_argument("--dev_file", type=str, default='./data/val_mat.json')
parser.add_argument("--test_file", type=str, default='./data/test_mat.json')
parser.add_argument("--target_file", type=str, default='./data/relation_mat.txt')
parser.add_argument("--log_dir", type=str, default='log')
parser.add_argument("--model_best_bin", type=str, default='./model/CopperBERT/model_best.bin')
parser.add_argument("--model_best_checkpoint_bin", type=str, default='./model/CopperBERT/checkpoint_best.bin')
parser.add_argument("--model_last_bin", type=str, default='./model/CopperBERT/model_last.bin')
parser.add_argument("--model_last_checkpoint_bin", type=str, default='./model/CopperBERT/checkpoint_last.bin')

# model
parser.add_argument('--embedding_dim', type=int, default=768, required=False, help='embedding_dim')
parser.add_argument("--max_len", type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.1, required=False, help='dropout')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=[2e-5, 3e-5, 4e-5, 5e-5])
parser.add_argument("--weight_decay", type=float, default=0)
hparams = parser.parse_args()
