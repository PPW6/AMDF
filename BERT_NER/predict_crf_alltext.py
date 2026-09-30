from model import BertNerModel
from transformers import BertTokenizer, AutoTokenizer
import pickle as pkl
import torch
from config import parsers
import time
from bert_crf import Bert_CRF, Bert_CRF_SPLIT
from normalize_text import normalize
import numpy as np
import nltk
import xlrd
import re
import os
import json

def load_model(model_path, class_num):
    global device
    if args.architecture == 'bert':
        model = BertNerModel(class_num).to(device)
    elif args.architecture == 'bert-crf':
        if is_split_into_words:
            model = Bert_CRF_SPLIT(class_num, device).to(device)
        else:
            model = Bert_CRF(class_num, device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def text_class_name(texts, pred, index_label, id2token):
    # result = torch.argmax(pred, dim=-1)  # dim的改变会让预测标签超出label，dim=n是求第n个维度最大值index
    # result = result.cpu().numpy().tolist()[0]

    pred_label = [index_label[i] for i in pred[0]]
    text_label=[]
    for i in range(len(pred_label)):
        text_label.append(texts[i]+':'+pred_label[i])
    # print("模型预测结果：")
    # print(f"文本：{texts}\t预测的类别为：{tetx_label[:len(texts)]}") #为str用这行代码
    # print(f"文本：{texts}\t预测的类别为：{text_label}")#用word_list用这行代码
    return texts, text_label

def read_data(process_path, property_path, name):
    with open(os.path.join(process_path, name), 'r', encoding='utf-8') as f:
        line = f.read()
        process = json.loads(line.strip())
    with open(os.path.join(property_path,name), 'r', encoding='utf-8') as f:
        line = f.read()
        property = json.loads(line.strip())
    text_list = process['sents_list']
    process_id=process['process_one']
    processm_id = process['process_multiple']
    property_id = property['property']
    process_text=[]
    property_text=[]
    for id in process_id:
        process_text.append({'process_one':text_list[id], 'id':id})
    for id in processm_id:
        process_text.append({'process_multiple':text_list[id], 'id':id})
    for id in property_id:
        property_text.append({'property': text_list[id], 'id':id})
    return process, property, process_text, property_text

def pred_one(text):
    global args

    dataset = pkl.load(open(args.data_pkl, "rb"))

    label_index, index_label = dataset[0], dataset[1]

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_pred)
    auto_tokenizer = AutoTokenizer.from_pretrained(args.bert_pred)

    # if len(text)<1000:
    #     max_len = len(text)
    # else:
    #     max_len = 512
    max_len = 510
    if is_split_into_words == True:
        tokenizer = auto_tokenizer
        return_offsets_mapping = True
    else:
        tokenizer = bert_tokenizer
        return_offsets_mapping = False
    text_id = tokenizer.encode(text, is_split_into_words=is_split_into_words, add_special_tokens=True,
                               max_length=max_len + 2,
                               padding="max_length", truncation=True, return_tensors="pt")
    # input_id变成token
    id2token = tokenizer.convert_ids_to_tokens(text_id[0])

    tokenized = tokenizer.encode_plus(text=text,
                                      is_split_into_words=is_split_into_words,
                                      max_length=max_len + 2,
                                      return_offsets_mapping=return_offsets_mapping,
                                      return_token_type_ids=True,
                                      return_attention_mask=True,
                                      return_tensors='pt',
                                      padding='max_length',
                                      truncation=True)
    if is_split_into_words == True:
        mask_label = encode_tags(tokenized, text)  # token用is_split_into_words=True,
        tokenized.pop('offset_mapping')
    model = load_model(args.save_model_best, len(label_index))
    tokenized = tokenized.to(device)
    mask_label = mask_label.tolist()
    mask_label = torch.tensor([mask_label], dtype=torch.int64)
    mask_label = mask_label.to(device)
    with torch.no_grad():
        pred = model(tokenized, mask_label, predit=True)
    texts, text_label=text_class_name(text, pred, index_label, id2token)
    return texts, text_label

def encode_tags(encodings, text):
    offset = encodings.offset_mapping[0]
    labels = np.ones(len(text), dtype=int) * 1
    doc_enc_labels = np.ones(len(offset), dtype=int) * -100
    arr_offset = np.array(offset)
    # 保留的是那些起始为0 且 终止位置非0的token（通常表示一个完整的词的起始）
    valid_token_mask = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
    # 有效 token 的数量
    num_valid_tokens = valid_token_mask.sum()
    truncated_labels = labels[:num_valid_tokens]
    doc_enc_labels[valid_token_mask] = truncated_labels
    return doc_enc_labels

def textnorm(texts, norm):
    if norm == False:
        return texts
    norm_split = normalize('\n'.join(texts)).split('\n')
    return norm_split

def save_data(dic_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as fb:
        json.dump(dic_list, fb, ensure_ascii=False)
    return dic_list

def predict_crf(dic_list, dir_path, file_name, text):
    path = os.path.join(dir_path, file_name[:-5] + '_NER.json')
    count_len = []
    key_list = []
    for item in text:
        for key in item:
            if key not in key_list and key != 'id':
                key_list.append(key)
    if len(text) == 0:
        count_len = [0]
        save_data(dic_list, path)
        return
    for name in key_list:
        target_texts, target_labels, ids = [], [], []
        for sents in text:
            if name in list(sents.keys()):
                if not re.search(r'[\u4e00-\u9fff]', sents[name]):# 过滤掉包含中文的句子
                    sent_list = nltk.word_tokenize(sents[name])
                    sent_norm = textnorm(sent_list, norm=True)
                    sent_norm = [word for word in sent_norm if word]
                    text_pre, text_label = pred_one(sent_norm)# 预测一条文本
                    # target_label = [word.split(':')[-1] for word in text_label[1:-1]]
                    target_label = [word.split(':')[-1] for word in text_label]
                    # 在NER任务中查找预测的句子全不全，有没有MAt 有DSC的句子可以往前面句子找MAT
                    if 'B-MAT' not in target_label and 'B-DSC' in target_label:
                        positioner = Hobbs_algorithm(dic_list['sents_list'], sent_idx=sents['id'], text_pre=sent_list, target_label=target_label)  # 进行指代消解
                        resolve_labels, resolve_words = positioner.hobbs_algorithm()
                        text_new = ' '.join(resolve_words)
                        dic_list['sents_list'][sents['id']] = text_new
                        target_labels.append(resolve_labels)
                        ids.append(sents['id'])
                        target_texts.append(resolve_words)
                        count_len.append(len(sents[name]))
                    else:
                        target_labels.append(target_label)
                        ids.append(sents['id'])
                        target_texts.append(text_pre)
                        count_len.append(len(sents[name]))
            else:
                continue
        if dic_list[name] == ids:
            for n in range(len(dic_list[name])):
                dic_list[name][n]={'id':ids[n], 'lable':target_labels[n]}
    count_len = sorted(count_len)
    # print(max(count_len), np.mean(count_len))
    save_data(dic_list, path)
    return target_texts, target_labels, ids

class Hobbs_algorithm:
    def __init__(self, sents_list, sent_idx, text_pre, target_label):
        self.sents_list = sents_list
        self.sent_idx = sent_idx
        self.text_pre = text_pre
        self.target_label = target_label

    def hobbs_algorithm(self):
        sents_list = self.sents_list
        sent_idx = self.sent_idx
        word_list = self.text_pre
        word_label = self.target_label

        combined_labels = []
        combined_words = []

        i = 0
        while i < len(word_label):
            if word_label[i] == 'B-DSC':
                combined_labels.append('DSC')
                combined_words.append('DSC')
                i += 1
                while i < len(word_label) and word_label[i] == 'I-DSC':
                    i += 1
            else:
                combined_labels.append(word_label[i])
                combined_words.append(word_list[i])
                i += 1

        for i, word in enumerate(combined_labels):
            if word == 'DSC':
                resolve_labels, resolve_words = Hobbs_algorithm.resolve_pronoun(self, sents_list, word_list, combined_words, combined_labels, i)
                return resolve_labels, resolve_words

    # Helper function to resolve pronouns using the Hobbs algorithm
    def resolve_pronoun(self, sent_list, word_list, combined_words, combined_labels, pronoun_idx):
        sent_idx = self.sent_idx
        mat_sequence = []
        mat_word = []
        # print("代词在第", sent_idx, "句:")
        # print("代词在word_list中的位置:", pronoun_idx)
        if sent_idx == 0:
            max_idx = min(sent_idx + 6, len(sent_list))
            for k in range(sent_idx + 1, max_idx, 1):
                word_list = sent_list[k]
                # print("代词的后句:", k, word_list)
                word_list = nltk.word_tokenize(word_list)
                sent_norm = textnorm(word_list, norm=True)
                sent_norm = [word for word in sent_norm if word]
                if len(sent_norm) > 512:
                    continue
                text_next, next_label = pred_one(sent_norm)  # 预测一条文本
                # target_label = [word.split(':')[-1] for word in next_label[1:-1]]
                target_label = [word.split(':')[-1] for word in next_label]
                # Find the first occurrence of B-MAT in target_label and include all adjacent I-MAT elements
                for i, label in enumerate(target_label):
                    if label == 'B-MAT':
                        mat_sequence = [label]
                        mat_word = [word_list[i]]
                        j = i + 1
                        while j < len(target_label) and target_label[j] == 'I-MAT':
                            mat_sequence.append(target_label[j])
                            mat_word.append(word_list[j])
                            j += 1
                        break
                if mat_sequence:
                    # Replace DSC in word_label with the found B-MAT and I-MAT sequence
                    combined_labels[pronoun_idx:pronoun_idx + len(mat_sequence)] = mat_sequence
                    combined_words[pronoun_idx:pronoun_idx + len(mat_sequence)] = mat_word
                    resolve_labels, resolve_words = combined_labels, combined_words
                    # print("指代消解新句子:",[resolve_words[i] + ':' + resolve_labels[i] for i in range(len(resolve_words))])
                    return resolve_labels, resolve_words
            return self.target_label, self.text_pre

        else:
            # 找到代词在的句子，往它的上5句找alloy_writing_type
            if sent_idx <6:
                end_id = -1
            else:
                end_id = sent_idx-6
            for k in range(sent_idx - 1, end_id, -1):
                word_list = sent_list[k]
                # print("代词的前句:", k, word_list)
                word_list = nltk.word_tokenize(word_list)
                sent_norm = textnorm(word_list, norm=True)
                sent_norm = [word for word in sent_norm if word]
                if len(sent_norm) > 512:
                    continue
                text_next, next_label = pred_one(sent_norm)  # 预测一条文本
                # target_label = [word.split(':')[-1] for word in next_label[1:-1]]
                target_label = [word.split(':')[-1] for word in next_label]
                mat_sequence = []
                mat_word = []
                # Find the first occurrence of B-MAT in target_label and include all adjacent I-MAT elements
                for i, label in enumerate(target_label):
                    if label == 'B-MAT':
                        mat_sequence = [label]
                        mat_word = [word_list[i]]
                        j = i + 1
                        while j < len(target_label) and target_label[j] == 'I-MAT':
                            mat_sequence.append(target_label[j])
                            mat_word.append(word_list[j])
                            j += 1
                        break
                if mat_sequence:
                    # Replace DSC in word_label with the found B-MAT and I-MAT sequence
                    combined_labels[pronoun_idx:pronoun_idx + len(mat_sequence)] = mat_sequence
                    combined_words[pronoun_idx:pronoun_idx + len(mat_word)] = mat_word
                    resolve_labels, resolve_words = combined_labels, combined_words
                    # print("指代消解新句子:", [resolve_words[i] + ':' + resolve_labels[i] for i in range(len(resolve_words))])
                    return resolve_labels, resolve_words
            return self.target_label, self.text_pre

if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    process_path = '../output_data_time/process'
    property_path = '../output_data_time/property'
    is_split_into_words = True
    length=os.listdir(process_path)
    length = [len for len in length if 'NER' not in len and 'RE' not in len and 'EX' not in len and 'json' in len]

    for i in range(35, len(length)):#有问题从i开始
    # idx = length.index('10.1007-s10853-022-08087-7.json')
    # for i in range(idx,idx+1):
        name = length[i]
        # print(f'{i}, resolve {name}')
        process, property, process_text, property_text = read_data(process_path, property_path, name)
        doi = process['doi']
        predict_crf(process, process_path, name, process_text)
        predict_crf(property, property_path, name, property_text)

    end = time.time()
    print(f"耗时为：{end - start} s")