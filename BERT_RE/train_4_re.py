import os
import json
import tqdm
import re
import nltk
from nltk.tokenize import WhitespaceTokenizer
from itertools import product


def split_data(file):
    file_dir = '.\\train_data'
    train_file = os.path.join(file_dir, 'train_1869.json')
    val_file = os.path.join(file_dir, 'val_1869.json')
    test_file = os.path.join(file_dir, 'test_1869.json')

    # with open(file, 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     lines = [line.strip() for line in lines]
    with open(file, 'r') as f:
        lines = json.load(f)


    train_lines = lines[:len(lines) * 6 // 10]
    val_lines = lines[len(lines) * 6 // 10:len(lines) * 8 // 10]
    test_lines = lines[len(lines) * 8 // 10:]

    train_lines = sorted(train_lines, key=lambda x: len(x['label']))
    val_lines = sorted(val_lines, key=lambda x: len(x['label']))
    test_lines = sorted(test_lines, key=lambda x: len(x['label']))

    save_data(train_lines, train_file)
    save_data(val_lines, val_file)
    save_data(test_lines, test_file)

def convert_data(line):
    #‘text’，label，找到MAT、PRO、PRC，两两分别建立联系
    text = line['text']
    word = WhitespaceTokenizer()
    word_list = word.tokenize(text)
    # word_list = nltk.word_tokenize(text)
    label = line['label']
    target = ['MAT', 'PRO', 'PUM', 'PRC', 'RUM']
    target_dict=[]
    target_id ={}
    for name in target:
        id = [(i,j) for i,j in enumerate(label) if j == 'B-'+name]
        for B_id in id:
            id_ = [B_id[0]]
            if id_[0] == 50 - 1:  # max_len句子最大长度
                id_.append(50)
            else:
                for n in range(B_id[0]+1,len(label)):
                    if label[n] == 'I-'+name:
                        id_.append(n)
                        # print(label[B_id[0] + 1],n)
                    if name not in label[n]:
                        break
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
    if len(MAT)>1:
        mat2pum = list(product(MAT, PUM))
        mat2pum_list=item(mat2pum, 'mat2pum',text)
    else:
        mat2pum_list=[]
    if len(PRO)>1:
        pro2pum = list(product(PRO, PUM))
        pro2pum_list=item(pro2pum, 'pro2pum',text)
    else:
        pro2pum_list=[]
    if len(PRC) > 1:
        prc2rum = list(product(PRC, RUM))
        prc2rum_list = item(prc2rum, 'prc2rum', text)
    else:
        prc2rum_list = []
    if len(PRC) > 1 and len(MAT) > 0:
        mat2prc = list(product(MAT, PRC))
        mat2prc_list = item(mat2prc, 'mat2prc', text)
    else:
        mat2prc_list = []
    if len(PRC) > 1 and len(PUM) > 0:
        prc2pum = list(product(PRC, PUM))
        prc2pum_list = item(prc2pum, 'prc2pum', text)
    else:
        prc2pum_list = []

    return mat2pum_list+prc2rum_list+pro2pum_list+mat2prc_list+prc2pum_list

def item(re, str_re,text):
    item_list = []

    for n in re:
        name_head, name_tail, pos_head, pos_tail =n[0][1], n[1][1], n[0][0], n[1][0]
        item = {
            'h': {
                'name': ' '.join(name_head),
                'pos': keep_first_and_last(pos_head)
            },
            't': {
                'name': ' '.join(name_tail),
                'pos': keep_first_and_last(pos_tail)
            },
            'relation': str_re,
            'text': text
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

def save_data(lines, file):
    with open(file, 'w', encoding='utf-8') as f:
        # for line in tqdm(lines, total=len(lines), desc=file):
        for line in lines:
            item_list = convert_data(line)
            for item in item_list:
                json_str = json.dumps(item, ensure_ascii=False)
                f.write('{}\n'.format(json_str))

if __name__ == '__main__':
    file = 'all_data_1869.json'

    split_data(file)