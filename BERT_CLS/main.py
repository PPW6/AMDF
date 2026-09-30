

import torch
from utils import read_data, MyDataset
from config import parsers
from torch.utils.data import DataLoader
from model import MyModel
from torch.optim import AdamW
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, precision_score
import time
from test1 import test_data
from tqdm import tqdm


if __name__ == "__main__":
    start = time.time()
    args = parsers()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_text, train_label, max_len = read_data(args.train_file)
    dev_text, dev_label = read_data(args.dev_file)
    if args.max_len <= max_len:
        args.max_len
    else:
        args.max_len = max_len

    train_dataset = MyDataset(train_text, train_label, args.max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    dev_dataset = MyDataset(dev_text, dev_label, args.max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    best_lr = None
    best_dev_list = None
    best_test_list = None
    f1_max = float("-inf")
    for lr in args.learn_rate:
        model = MyModel().to(device)
        print(f"lr: {lr}")
        opt = AdamW(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(args.epochs):

            model.train()
            loss_sum = 0.0

            train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs} lr={lr}")

            for batch_index, (batch_text, batch_label) in enumerate(train_bar):
                batch_label = batch_label.to(device)

                pred = model(batch_text)
                loss = loss_fn(pred, batch_label)

                opt.zero_grad()
                loss.backward()
                opt.step()

                loss_sum += loss.item()

                # tqdm显示loss
                train_bar.set_postfix(loss=loss.item(), avg_loss=loss_sum / (batch_index + 1))

            model.eval()
            all_pred, all_true = [], []

            with torch.no_grad():
                dev_bar = tqdm(dev_dataloader, desc="Evaluating")

                for batch_text, batch_label in dev_bar:
                    batch_label = batch_label.to(device)

                    pred = model(batch_text)

                    pred = torch.argmax(pred, dim=1).cpu().numpy().tolist()
                    label = batch_label.cpu().numpy().tolist()

                    all_pred.extend(pred)
                    all_true.extend(label)
            average = 'macro'
            dev_acc = accuracy_score(all_true, all_pred)
            dev_precision = precision_score(all_true, all_pred, average=average)
            dev_recall = recall_score(all_true, all_pred, average=average)
            dev_f1 = f1_score(all_true, all_pred, average=average)
            target_names = open(args.classification, "r", encoding="utf-8").read().split("\n")
            cla_report = classification_report(all_true, all_pred, target_names=target_names)
            print('cla_report:', cla_report)
            print(f"dev precision:{dev_precision:.4f}")
            print(f"dev rec:{dev_recall:.4f}")
            print(f"dev f1:{dev_f1:.4f}")
            if dev_f1 > f1_max:
                best_lr = lr
                f1_max = dev_f1
                best_dev_list = (dev_f1, dev_precision, dev_recall)
                torch.save(model.state_dict(), args.save_model_best)
                test_data()
                print(f"save_best_model")

    torch.save(model.state_dict(), args.save_model_last)
    print(f"save_last_model\n best_model: best_lr: {best_lr}\ndev f1:{best_dev_list[0]}, precision:{best_dev_list[1]}, recall:{best_dev_list[2]}" )

    end = time.time()
    print(f"运行时间：{(end-start)/60%60:.4f} min")
    test_data()
