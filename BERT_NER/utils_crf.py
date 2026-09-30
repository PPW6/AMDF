
import pickle as pkl
import numpy as np
import torch
from config import parsers
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from normalize_text import normalize

def read_data(file, norm=False):
    # 读取文件
    all_data = open(file, "r", encoding="utf-8").read().split("\n")
    # 得到所有文本、所有标签
    texts, labels = [], []
    text_one, label_one = [], []
    for data in all_data:
        if data != '':
            text, label = data.split()
            # print(text, label)
            text_one.append(text)
            label_one.append(label)
        elif text_one:
            texts.append(text_one)
            labels.append(label_one)
            text_one, label_one = [], []
    texts = textnorm(texts, norm=norm)
    return texts, labels

def textnorm(texts, norm):
    if norm == False:
        return texts
    # norm_data = []
    # for split in texts:
    norm_split = []
    for s in texts:
        norm_split.append(normalize('\n'.join(s)).split('\n'))
        # norm_data.append(norm_split)
    return norm_split

def build_label_index(labels):
    # label_to_index = {"PAD": 0, "UNK": 1}#不要pad和unk试试
    label_to_index = {}
    for label in labels:
        for i in label:
            if i not in label_to_index:
                label_to_index[i] = len(label_to_index)
    index_to_label = sorted(list(label_to_index))
    label_to_index = {k:index_to_label.index(k)  for k in index_to_label }
    pkl.dump([label_to_index, index_to_label], open(parsers().data_pkl, "wb"))
    return label_to_index, index_to_label


class MyDataset(Dataset):
    def __init__(self, texts, label_to_index, is_split_into_words=True, with_labels=True, labels=None, ):
        self.all_text = texts
        self.all_label = labels
        self.with_labels = with_labels
        self.bert_tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)
        self.auto_tokenizer = AutoTokenizer.from_pretrained(parsers().bert_pred)
        self.label_index = label_to_index
        self.max_len = parsers().max_len
        self.is_split_into_words = is_split_into_words

    def encode_tags(self, tags, encodings, text):
        labels = [self.label_index[tag] for tag in tags]
        offset = encodings.offset_mapping[0]
        doc_enc_labels = np.ones(len(offset), dtype=int) * -100
        arr_offset = np.array(offset)
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = labels
        return doc_enc_labels

    def __getitem__(self, index):
        text = self.all_text[index]
        if self.is_split_into_words == True:
            tokenizer = self.auto_tokenizer
            return_offsets_mapping = True
        else:
            tokenizer = self.bert_tokenizer
            return_offsets_mapping = False
        text_id = tokenizer.encode(text, is_split_into_words=self.is_split_into_words,add_special_tokens=True, max_length=self.max_len + 2,
                                        padding="max_length", truncation=True, return_tensors="pt")

        # 查看分词器如何分词
        # token = self.tokenizer.tokenize(text)#str序列输入，而不是word_list的输入
        # 分词变成input_id
        # indexes = self.tokenizer.convert_tokens_to_ids(token)
        # input_id变成token
        id2token = tokenizer.convert_ids_to_tokens(text_id[0])
        token_texts = []

        # self.encode_tags(tags, encodings)
        tokenized = tokenizer.encode_plus(text=text,
                                          is_split_into_words=self.is_split_into_words,
                                          max_length=self.max_len+2,
                                          return_offsets_mapping=return_offsets_mapping,
                                          return_token_type_ids=True,
                                          return_attention_mask=True,
                                          return_tensors='pt',
                                          padding='max_length',
                                          truncation=True)
        if self.with_labels:  # True if the dataset has labels
            label = self.all_label[index][:self.max_len]
            if self.is_split_into_words == True:
                label_id=self.encode_tags(label, tokenized, text) #token用is_split_into_words=True,
                tokenized.pop('offset_mapping')
            else:
                label_id = np.array([-1] + [self.label_index.get(i, 1) for i in label] + [-1] +
                                    [-1] * (self.max_len - len(text)))#token用is_split_into_words=False
            label_id=label_id.tolist()
            label_id = torch.tensor(label_id, dtype=torch.int64)
            return text_id, label_id, tokenized
        else:
            return text_id,tokenized

    def __len__(self):
        # 得到文本的长度
        return len(self.all_text)

if __name__ == "__main__":
    args = parsers()
    train_text, train_label = read_data(args.train_file)
    dev_text, dev_label = read_data(args.dev_file)
    test_text, test_label = read_data(args.test_file)
    label_index, index_label = build_label_index(train_label)

    trainDataset = MyDataset(train_text, label_index, labels=train_label, with_labels=True)
    trainLoader = DataLoader(trainDataset, batch_size=2000, shuffle=False)

    for batch_text, batch_label, id  in trainLoader:
        print(batch_text, batch_label)
        break
