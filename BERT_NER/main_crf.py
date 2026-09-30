
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
from tqdm import tqdm
from config import parsers
from utils_crf import read_data, MyDataset, build_label_index
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from bert import BertNerModel
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from bert_crf import Bert_CRF, Bert_CRF_SPLIT
from transformers import BertTokenizer, set_seed
from torch.utils.tensorboard import SummaryWriter

def prepare_data(is_split_into_words):
    global args
    if args.bert_pred.lower()[2:] == 'scibert':
        to_normalize = False
    elif args.bert_pred.lower()[2:] == 'matscibert':
        to_normalize = True
    elif args.bert_pred.lower()[2:] == 'bert-base-uncased':
        to_normalize = False
    train_text, train_label = read_data(args.train_file,norm=to_normalize)
    dev_text, dev_label = read_data(args.dev_file,norm=to_normalize)
    test_text, test_label = read_data(args.test_file,norm=to_normalize)
    label_to_index, index_to_label = build_label_index(train_label)

    trainDataset = MyDataset(train_text, label_to_index, is_split_into_words=is_split_into_words, labels=train_label, with_labels=True)
    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)

    devDataset = MyDataset(dev_text, label_to_index, is_split_into_words=is_split_into_words, labels=dev_label, with_labels=True)
    devLoader = DataLoader(devDataset, batch_size=args.batch_size, shuffle=False)

    testDataset = MyDataset(test_text, label_to_index, is_split_into_words=is_split_into_words, labels=test_label, with_labels=True)
    testLoader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)

    return trainLoader, devLoader, testLoader, label_to_index, index_to_label
def prd_label(tag_label,pred_label, is_split_into_words):
    batch_size = args.batch_size
    # 格式转化，转化为List(str)
    index2tag = {v: k for k, v in label_index.items()}
    temp_tags = []
    final_tags = []
    for index in range(len(tag_label)):
        if is_split_into_words:
            length = len(pred_label[index])
            temp_tags.append(tag_label[index][1:length])
        else:
            # predictions先去掉头，再去掉尾
            pred_label[index].pop()
            length = len(pred_label[index])
            temp_tags.append(tag_label[index][1:length])
            pred_label[index].pop(0)
        temp_tags[index] = [index2tag[l] for (p, l) in zip(pred_label[index], temp_tags[index]) if l != -100]
        pred_label[index] = [index2tag[p] for (p, l) in zip(pred_label[index], temp_tags[index]) if l != -100]
        final_tags.append(temp_tags[index])
    return final_tags, pred_label

if __name__ == "__main__":
    start = time.time()
    args = parsers()
    tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    is_split_into_words = True
    train_loader, dev_loader, test_loader, label_index, index_label = prepare_data(is_split_into_words)
    if args.architecture == 'bert':
        model = BertNerModel(len(label_index)).to(device)
    elif args.architecture == 'bert-crf':
        if is_split_into_words:
            model = Bert_CRF_SPLIT(len(label_index), device).to(device)
        else:
            model = Bert_CRF(len(label_index), device).to(device)

    loss_fun = nn.CrossEntropyLoss()
    f1_max = float("-inf")
    best_lr = None
    best_dev_list = None
    best_test_list = None

    for lr in args.learn_rate:
        print(f'lr: {lr}')
        opt = torch.optim.AdamW(model.parameters(), lr)
        plt_loss=[]
        for epoch in range(args.epochs):
            model.train()
            loss_sum, num = 0, 0
            pbar = tqdm(train_loader)
            for batch_text, batch_label, token_texts in pbar:
                batch_text = batch_text.to(device)
                batch_text = batch_text.squeeze(1)

                id2token = tokenizer.convert_ids_to_tokens(batch_text[0])
                batch_label = batch_label.to(device)
                token_texts = token_texts.to(device)

                loss, predictions = model(token_texts,batch_label)#需要3维
                opt.zero_grad()
                loss.backward()
                opt.step()

                loss_sum += loss
                num += 1

                pbar.set_description('epoch: {}/{}'.format(epoch + 1, args.epochs))  # set_description()设置进度条前方信息
                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss)})  # set_postfix()设置进度条后方信息

            loss_avg = loss_sum / num
            plt_loss.append(loss_avg.cpu().detach().numpy().tolist())
            # print(loss_avg.cpu().detach().numpy().tolist())
            print(f"train epoch:{epoch+1}\tloss:{loss_avg:.2f}")

            model.eval()

            all_pre = []
            all_tag = []
            for batch_text, batch_label, token_texts in dev_loader:
                batch_text = batch_text.to(device)
                token_texts = token_texts.to(device)
                batch_label = batch_label.to(device)
                loss,pred_label = model(token_texts,batch_label)

                # pred_label = torch.argmax(pred, dim=-1).cpu().numpy().tolist()
                tag_label = batch_label.cpu().numpy().tolist()
                final_tags, pred_label = prd_label(tag_label, pred_label, is_split_into_words)
                all_tag+=final_tags
                all_pre+=pred_label
                # for pred, tag in zip(pred_label, tag_label):
                #     p = [index_label[i] for i in pred]
                #     t = [index_label[i] for i in tag]
                #     all_pre.append(p)
                #     all_tag.append(t)

            dev_f1 = f1_score(all_tag, all_pre)
            dev_precision = precision_score(all_tag, all_pre)
            dev_recall = recall_score(all_tag, all_pre)
            cla_report = classification_report(all_tag, all_pre)
            print('cla_report:', cla_report)
            print(f"dev f1:{dev_f1}, precision:{dev_precision}, recall:{dev_recall}")

            model.eval()

            all_pre = []
            all_tag = []
            for batch_text, batch_label, token_texts in test_loader:
                token_texts = token_texts.to(device)
                batch_text = batch_text.to(device)
                batch_label = batch_label.to(device)
                loss, pred_label = model(token_texts, batch_label)

                # pred_label = torch.argmax(pred, dim=-1).cpu().numpy().tolist()
                tag_label = batch_label.cpu().numpy().tolist()
                final_tags, pred_label = prd_label(tag_label, pred_label, is_split_into_words)
                all_tag += final_tags
                all_pre += pred_label
                # for pred, tag in zip(pred_label, tag_label):
                #     p = [index_label[i] for i in pred]
                #     t = [index_label[i] for i in tag]
                #     all_pre.append(p)
                #     all_tag.append(t)

            test_f1 = f1_score(all_tag, all_pre)
            test_precision = precision_score(all_tag, all_pre)
            test_recall = recall_score(all_tag, all_pre)
            print(f"test f1:{test_f1}, precision:{test_precision}, recall:{test_recall}")

            if f1_max < dev_f1:
                f1_max = dev_f1
                best_lr = lr
                best_dev_list = (dev_f1,dev_precision,dev_recall)
                best_test_list = (test_f1,test_precision,test_recall)
                torch.save(model.state_dict(), args.save_model_best)
                print(f"save_best_model")


    torch.save(model.state_dict(), args.save_model_last)
    print(f"save_last_model\n best_model: best_lr: {best_lr}\ndev f1:{best_dev_list[0]}, precision:{best_dev_list[1]}, recall:{best_dev_list[2]} \n test f1:{best_test_list[0]}, precision:{best_test_list[1]}，recall:{best_test_list[2]}")

    end = time.time()
    print(f"运行时间：{(end - start) / 60 % 60:.4f} min")

