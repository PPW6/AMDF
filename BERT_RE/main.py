
import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # 不能有中文路径
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils_mat import SentenceREDataset, load_checkpoint, save_checkpoint
from model import SentenceRE
from config import hparams
from train_4_re import split_data


def train(hparams):
    start = time.time()
    train_file = hparams.train_file
    dev_file = hparams.dev_file
    test_file = hparams.test_file
    if not os.path.exists(train_file):
        split_data(hparams.all_data)

    device = hparams.device
    bert_path = hparams.bert_path

    target_file = hparams.target_file#relation.txt
    log_dir = hparams.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        os.mkdir(os.path.split(hparams.model_best_bin)[0][2:])

    model_best_bin = hparams.model_best_bin
    model_best_checkpoint_bin = hparams.model_best_checkpoint_bin
    model_last_bin = hparams.model_last_bin
    model_last_checkpoint_bin = hparams.model_last_checkpoint_bin

    batch_size = hparams.batch_size
    epochs = hparams.epochs

    learning_rate = hparams.learning_rate
    weight_decay = hparams.weight_decay

    # train_dataset
    train_dataset = SentenceREDataset(train_file, target_path=target_file)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    dev_dataset = SentenceREDataset(dev_file, target_path=target_file)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)

    test_dataset = SentenceREDataset(test_file, target_path=target_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    # model
    model = SentenceRE(hparams).to(device)

    # load checkpoint if one exists
    if os.path.exists(model_best_checkpoint_bin):
        checkpoint_dict = load_checkpoint(model_best_checkpoint_bin)
        best_f1 = checkpoint_dict['best_f1']
        epoch_offset = checkpoint_dict['best_epoch'] + 1
        model.load_state_dict(torch.load(model_best_bin))
    else:
        checkpoint_dict = {}
        best_f1 = 0.0
        epoch_offset = 0
    best_dev_list = None
    # optimizer
    for lr in learning_rate:
        print(f'lr: {lr}')

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

        for epoch in range(epoch_offset, epochs):
            model.train()
            msg = ""
            train_loss, train_los_agv = 0, 0
            for i_batch, sample_batched in enumerate(tqdm(train_loader, desc='Training')):
                token_ids = sample_batched['input_ids'].to(device)
                token_type_ids = sample_batched['token_type_ids'].to(device)
                attention_mask = sample_batched['attention_mask'].to(device)
                e1_mask = sample_batched['e1_masks'].to(device)
                e2_mask = sample_batched['e2_masks'].to(device)
                e1_pos = sample_batched['e1_pos'].to(device)
                e2_pos = sample_batched['e2_pos'].to(device)
                tag_ids = sample_batched['labels'].to(device)

                opt.zero_grad()
                preds = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask,e1_pos, e2_pos )
                loss = criterion(preds, tag_ids)
                loss.backward()
                opt.step()

                train_loss += loss.item()
                train_los_agv += loss.item()
                if i_batch % 10 == 9:
                    writer.add_scalar('Training/training loss', train_loss / 10, epoch * len(train_loader) + i_batch)
                    train_loss = 0.0

            msg = f"[{epoch}/{epochs}] train_loss: {train_los_agv / len(train_loader):.4f}"
            print('msg:', msg)
            # if epoch % 2 == 0:#每两个epoch评价一次
            model.eval()
            tags_true, tags_pred = [], []
            save_best = False
            with torch.no_grad():
                for val_i_batch, val_sample_batched in enumerate(tqdm(dev_loader, desc='valid')):
                    token_ids = val_sample_batched['input_ids'].to(device)
                    token_type_ids = val_sample_batched['token_type_ids'].to(device)
                    attention_mask = val_sample_batched['attention_mask'].to(device)
                    e1_mask = val_sample_batched['e1_masks'].to(device)
                    e2_mask = val_sample_batched['e2_masks'].to(device)
                    e1_pos = val_sample_batched['e1_pos'].to(device)
                    e2_pos = val_sample_batched['e2_pos'].to(device)
                    tag_ids = val_sample_batched['labels']

                    preds = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask, e1_pos, e2_pos)
                    preds_tag_ids = preds.argmax(1)
                    tags_true.extend(tag_ids.tolist())
                    tags_pred.extend(preds_tag_ids.tolist())
            average='macro'
            dev_f1 = f1_score(tags_true, tags_pred, average=average)
            dev_precision = precision_score(tags_true, tags_pred, average=average)
            dev_recall = recall_score(tags_true, tags_pred, average=average)
            dev_accuracy = accuracy_score(tags_true, tags_pred)
            target_names = open(hparams.target_file, "r", encoding="utf-8").read().split("\n")
            cla_report = classification_report(tags_true, tags_pred, target_names=target_names)
            print('\n','cla_report:', cla_report, '\n','dev_precision:', dev_precision, ',dev_recall:',dev_recall, 'dev_f1:', dev_f1)

            writer.add_scalar('valid/f1', dev_f1, epoch)
            writer.add_scalar('valid/precision', dev_precision, epoch)
            writer.add_scalar('valid/recall', dev_recall, epoch)
            writer.add_scalar('valid/accuracy', dev_accuracy, epoch)

            if checkpoint_dict.get('epoch_f1'):
                checkpoint_dict['epoch_f1'][epoch] = dev_f1
            else:
                checkpoint_dict['epoch_f1'] = {epoch: dev_f1}
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                best_lr = lr
                checkpoint_dict['best_f1'] = best_f1
                checkpoint_dict['best_epoch'] = epoch
                best_dev_list = (dev_f1, dev_precision, dev_recall)
                torch.save(model.state_dict(), model_best_bin)
                print(f"save_best_model")
                save_best = True
            save_checkpoint(checkpoint_dict, model_best_checkpoint_bin)

            msg += f" dev_precision: {dev_precision:.4f} dev_recall: {dev_recall:.4f} dev_f1: {dev_f1:.4f} {'*' if save_best else ''}"
            print('msg:', msg)


        model.eval()
        tags_true, tags_pred = [], []
        with torch.no_grad():
            for sample_batched in tqdm(test_loader, desc='test'):
                token_ids = sample_batched['input_ids'].to(device)
                token_type_ids = sample_batched['token_type_ids'].to(device)
                attention_mask = sample_batched['attention_mask'].to(device)
                e1_mask = sample_batched['e1_masks'].to(device)
                e2_mask = sample_batched['e2_masks'].to(device)
                e1_pos = sample_batched['e1_pos'].to(device)
                e2_pos = sample_batched['e2_pos'].to(device)
                tag_ids = sample_batched['labels']

                preds = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask, e1_pos, e2_pos)
                preds_tag_ids = preds.argmax(1)
                tags_true.extend(tag_ids.tolist())
                tags_pred.extend(preds_tag_ids.tolist())
        average = 'macro'
        test_f1 = f1_score(tags_true, tags_pred, average=average)
        test_precision = precision_score(tags_true, tags_pred, average=average)
        test_recall = recall_score(tags_true, tags_pred, average='macro')
        test_accuracy = accuracy_score(tags_true, tags_pred)
        print(f"{lr}_last_epoch: test precision: {test_precision: .4f}, test recall: {test_recall: .4f}, test f1: {test_f1: .4f}")
        writer.add_scalar('test f1', test_f1)
        writer.add_scalar('test precision', test_precision)
        writer.add_scalar('test recall', test_recall)
        writer.add_scalar('test acc', test_accuracy)
        writer.close()

    torch.save(model.state_dict(), model_last_bin)
    save_checkpoint(checkpoint_dict, model_last_checkpoint_bin)
    print(f"save_last_model\n best_model: best_lr: {best_lr}\ndev f1:{best_dev_list[0]}, precision:{best_dev_list[1]}, recall:{best_dev_list[2]} \n ")


    end = time.time()
    print(f"运行时间：{(end - start) / 60 % 60:.4f} min")



if __name__ == "__main__":
    train(hparams)
