
import argparse
import os.path


def parsers():
    parser = argparse.ArgumentParser(description="Bert model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join("./data/process", "train.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join("./data/process", "dev.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join("./data/process", "test.txt"))
    parser.add_argument("--classification", type=str, default=os.path.join("./data/process", "class.txt"))
    parser.add_argument("--bert_pred", type=str, default="../BERT_model/MatsciBERT")
    # parser.add_argument("--bert_pred", type=str, default=r"H:\scientificProject\SteelScientist-main\model_saved\steelbert")
    parser.add_argument("--max_len", type=int, default=510)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learn_rate", type=float, default=[1e-5, 2e-5, 3e-5])
    parser.add_argument("--num_filters", type=int, default=768)
    parser.add_argument("--save_model_best", type=str, default=os.path.join("model", "process", "best_model.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join("model", "process", "last_model.pth"))
    args = parser.parse_args()
    return args
