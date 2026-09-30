
import argparse
import os.path


def parsers():
    parser = argparse.ArgumentParser(description="Bert model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join("./data", "train_mat.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join("./data", "dev_mat.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join("./data", "test_mat.txt"))
    parser.add_argument("--data_pkl", type=str, default=os.path.join("./data", "dataParams.pkl"))
    parser.add_argument("--bert_pred", type=str, default="../BERT_model/MatsciBERT", help="bert 预训练模型")
    parser.add_argument('--architecture', default='bert-crf', help=['bert', 'bert-crf'])
    parser.add_argument("--max_len", type=int, default=520, help="句子的最大长度")
    parser.add_argument("--batch_size", type=int, default=52)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learn_rate", type=float, default=[2e-5, 3e-5, 4e-5, 5e-5])
    parser.add_argument("--save_model_best", type=str, default=os.path.join("./model/MatSciBERT+crf", "best_model_bert_crf.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join("./model/MatSciBERT+crf", "last_model_bert_crf.pth"))
    parser.add_argument('--seeds', nargs='+', default=None, type=int)
    args = parser.parse_args()
    return args
