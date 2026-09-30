
import json
from itertools import product
import torch
from nltk import word_tokenize
from utils_mat import MyTokenizer, get_idx2tag, convert_pos_to_mask
from model import SentenceRE
from config import hparams
import os
import nltk
import time
from nltk.tokenize import WhitespaceTokenizer

def read_data(dic_list,name):
    item_list=[]
    data = dic_list[name]
    for line in data:
        line['text'] = dic_list['sents_list'][line['id']]
        item = convert_data(line)
        item_list.append(item)
    item_list = [item for item in item_list if len(item) != 0]
    return item_list

def convert_data(line):
    #‘text’，label，找到MAT、PRO、PRC，两两分别建立联系
    id = line['id']
    text = line['text']
    # text = ' '.join(word_list)
    word_list = nltk.word_tokenize(text)
    label = line['lable']
    target = ['MAT', 'PRO', 'PUM', 'PRC', 'RUM']
    target_dict=[]
    target_id ={}
    for name in target:
        id_list = [(i,j) for i,j in enumerate(label) if j == 'B-'+name]
        for B_id in id_list:
            id_ = [B_id[0]]
            for n in range(B_id[0]+1,len(label)):
                if label[n] == 'I-'+name:
                    id_.append(n)
                    # print(label[B_id[0] + 1],n)
                if name not in label[n]:
                    break
            # if id_[0] == 50-1:  #max_len句子最大长度
            #     id_.append(50)
            # else:
            id_.append(n)
            target_id[name] = id_
            target_id['text'] = word_list[id_[0]:id_[-1]]
            target_dict.append(target_id)
            target_id={}

    MAT_num = len([i for i,j in enumerate(label) if j == 'B-'+target[0]])
    PUM_num = len([i for i,j in enumerate(label) if j == 'B-'+target[1]])
    PRC_num = len([i for i,j in enumerate(label) if j == 'B-'+target[2]])
    RUM_num = len([i for i,j in enumerate(label) if j == 'B-'+target[3]])
    MAT, PRO, PUM, PRC, RUM =[], [], [], [], []
    for one_dict in target_dict:
        for key,value in one_dict.items():
            # print(key,value)
            if key==target[0]:
                MAT.append((one_dict[key],one_dict['text']))
            elif key==target[1]:
                PRO.append((one_dict[key], one_dict['text']))
            elif key==target[2]:
                PUM.append((one_dict[key],one_dict['text']))
            elif key==target[3]:
                PRC.append((one_dict[key],one_dict['text']))
            elif key==target[4]:
                RUM.append((one_dict[key], one_dict['text']))
    if MAT:
        mat2pum = list(product(MAT, PUM))
        mat2pum_list = item_conv(mat2pum, id, text, label)
    else:
        mat2pum_list = []
    if PRO:
        pro2pum = list(product(PRO, PUM))
        pro2pum_list = item_conv(pro2pum, id, text, label)
    else:
        pro2pum_list = []
    if PRC:
        prc2rum = list(product(PRC, RUM))
        prc2rum_list = item_conv(prc2rum, id, text, label)
    else:
        prc2rum_list = []
    if PRC and MAT:
        mat2prc = list(product(MAT, PRC))
        mat2prc_list = item_conv(mat2prc, id, text, label)
    else:
        mat2prc_list = []
    if PRC and PUM:
        prc2pum = list(product(PRC, PUM))
        prc2pum_list = item_conv(prc2pum, id, text, label)
    else:
        prc2pum_list = []
    return mat2pum_list + prc2rum_list + pro2pum_list + mat2prc_list + prc2pum_list
    # return mat2pum_list
def item_conv(re, id, text, label):
    item_list = []
    for n in re:
        name_head, name_tail, pos_head, pos_tail =n[0][1], n[1][1], n[0][0], n[1][0]
        item = {
            'id': id,
            'h': {
                'name': ' '.join(name_head),
                'pos': keep_first_and_last(pos_head)
            },
            't': {
                'name': ' '.join(name_tail),
                'pos': keep_first_and_last(pos_tail)
            },
            'text': text,
            'label': label
        }
        item_list.append(item)
        item ={}

    return item_list

def keep_first_and_last(lst):
    if len(lst) >= 2:
        return [lst[0], lst[-1]]
    elif len(lst) == 1:
        return lst
    else:
        return []

def process_data(item, tokenizer, device):
    # 编码格式
    tokens, pos_e1, pos_e2, e1_marker_pos, e2_marker_pos = tokenizer.tokenize(item)
    encoded = tokenizer.bert_tokenizer.encode_plus(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)

    word_ids = encoded.word_ids()
    e1_mask = [1 if wid is not None and pos_e1[0] <= wid < pos_e1[1] else 0 for wid in word_ids]
    e2_mask = [1 if wid is not None and pos_e2[0] <= wid < pos_e2[1] else 0 for wid in word_ids]

    e1_pos = next(i for i, wid in enumerate(word_ids) if wid == e1_marker_pos)
    e2_pos = next(i for i, wid in enumerate(word_ids) if wid == e2_marker_pos)

    input_ids = encoded["input_ids"].to(device)
    token_type_ids = encoded["token_type_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    e1_mask = torch.tensor([e1_mask], dtype=torch.long, device=device)

    e2_mask = torch.tensor([e2_mask], dtype=torch.long, device=device)

    e1_pos = torch.tensor([e1_pos], dtype=torch.long, device=device)

    e2_pos = torch.tensor([e2_pos],dtype=torch.long,device=device)

    return (input_ids, token_type_ids, attention_mask, e1_mask, e2_mask, e1_pos, e2_pos)

def predict(hparams, dic_list, output_path, file_name):
    device = hparams.device
    target_file = hparams.target_file

    bert_path = hparams.bert_path
    model_best_bin = hparams.model_best_bin

    idx2tag = get_idx2tag(target_file)
    model = SentenceRE(hparams).to(device)
    model.load_state_dict(torch.load(model_best_bin), strict=False)
    model.eval()
    tokenizer = MyTokenizer(bert_path)

    for dic_name in dic_list['class_name']:
        result_file = os.path.join(output_path, file_name[:-9] + '_RE.json')
        if dic_name != 'other':
            item_list= read_data(dic_list, dic_name)
            if len(item_list) != 0:
                for one in item_list:
                    for one_dic in one:
                        (input_ids, token_type_ids, attention_mask, e1_mask, e2_mask, e1_pos, e2_pos) = process_data(one_dic, tokenizer, device)
                        with torch.no_grad():
                            preds = model(input_ids, token_type_ids, attention_mask, e1_mask, e2_mask, e1_pos, e2_pos)
                        preds = preds.to(torch.device('cpu'))
                        relation = idx2tag[preds.argmax(dim=1).item()]

                        print("在{}【{}】中【{}】与【{}】的关系为：{}".format(dic_list['doi'], one_dic['text'], one_dic['h'], one_dic['t'], relation))
                        one_dic['relation']=relation
            item_list = [[one_dic for one_dic in item if one_dic['relation'] != 'None'] for item in item_list]
            item_list = [item for item in item_list if len(item)!= 0]
            dic_list[dic_name+'_relation']= item_list
    save_data(result_file, dic_list)

def save_data(result_file, dic_list):
    with open(result_file, 'w', encoding='utf-8') as f:
        json_str = json.dumps(dic_list, ensure_ascii=False)
        f.write('{}\n'.format(json_str))

if __name__ == "__main__":
    start = time.time()
    process_path = '../output_data/process'
    property_path = '../output_data/property'
    length = os.listdir(process_path)
    length = [len for len in length if 'NER' in len]

    for i in range(0, len(length)):
    # for i in range(0,439):
        file_name = length[i]
        print(f'{i}, resolve {file_name}')
        with open(os.path.join(process_path, file_name), 'r', encoding='utf-8') as f:
            process = json.load(f)
        with open(os.path.join(property_path, file_name), 'r', encoding='utf-8') as f:
            property = json.load(f)
        predict(hparams, process, process_path, file_name)
        predict(hparams, property, property_path, file_name)
    end = time.time()
    print(f"耗时为：{end - start} s")
