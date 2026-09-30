
from model import MyModel
from config import parsers
import torch
from transformers import BertTokenizer
import time
import os
import nltk
import json
import xlrd
import pandas as pd

def load_model(device, model_path):
    myModel = MyModel().to(device)
    myModel.load_state_dict(torch.load(model_path))
    myModel.eval()
    return myModel


def process_text(text, bert_pred):
    tokenizer = BertTokenizer.from_pretrained(bert_pred)
    # token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(text))
    # mask = [1] * len(token_id) + [0] * (args.max_len + 2 - len(token_id))
    # token_ids = token_id + [0] * (args.max_len + 2 - len(token_id))
    # token_ids = torch.tensor(token_ids).unsqueeze(0)
    # mask = torch.tensor(mask).unsqueeze(0)
    # text_input = torch.stack([token_ids, mask])
    # return text_input
    # Automatically handle truncation in tokenizer
    encoded_inputs = tokenizer(text,
                               return_tensors='pt',  # Return PyTorch tensors
                               truncation=True,  # Automatically truncate if text is too long
                               max_length=args.max_len + 2,  # Set maximum length for input
                               padding='max_length')  # Pad to max_length if shorter

    input_ids = encoded_inputs['input_ids']  # Token IDs
    attention_mask = encoded_inputs['attention_mask']  # Attention Mask

    return input_ids, attention_mask


def text_class_name(pred, classification):
    result = torch.argmax(pred, dim=1)
    result = result.cpu().numpy().tolist()
    result = classification[result[0]]
    classification_dict = dict(zip(range(len(classification)), classification))
    # print(f"文本：{text}\t预测的类别为：{classification_dict[result[0]]}")
    return result
    
if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(device, args.save_model_best)
    # file_name = os.path.basename(args.save_model_best)
    file_name = args.save_model_best
    file_name = file_name[:-4].split("\\")[-2]
    text_path = "../output_files"
    length = os.listdir(text_path)
    length = [len for len in length if 'NER' not in len and 'RE' not in len and 'EX' not in len and 'txt' in len]
    elsevier_data = pd.read_excel("../elsevier.xlsx", engine='openpyxl')
    springer_data = pd.read_csv('../springer.csv', encoding='ISO-8859-1')
    # 获取文章元数据
    meta_doi = elsevier_data['DOI'].tolist()
    meta_title = elsevier_data['Title'].tolist()
    meta_author = elsevier_data['Authors'].tolist()
    meta_publisher = elsevier_data['Sourcetitle'].tolist()
    meta_date = elsevier_data['Date'].tolist()
    meta_volume = elsevier_data['Volume'].tolist()
    meta_url = elsevier_data['Link'].tolist()

    meta_doi += springer_data['DOI'].tolist()
    meta_title += springer_data['Title'].tolist()
    meta_author += springer_data['Authors'].tolist()
    meta_publisher += springer_data['Sourcetitle'].tolist()
    meta_date += springer_data['Date'].tolist()
    meta_volume += springer_data['Volume'].tolist()
    meta_url += springer_data['URL'].tolist()

    # dict_paragraph_class = []
    #获取文章摘要
    for i in range(0, len(length)):#如果有错误，从i开始调试
    # for i in range(72,73):
        with open(os.path.join(text_path, length[i]), 'r', encoding='utf-8') as file:
            print(f'{i}, resolve {length[i]}')
            doi_ = length[i].split('.txt')[0]
            filter_txt = file.readlines()
            abstract = filter_txt[0]
            abstract.replace('/n', ' ')
            filter_txt = ' '.join(filter_txt)
        texts = nltk.sent_tokenize(filter_txt)
        classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
        category_indices = {name: [] for name in classification}
        category_indices.update({'class_name': classification})
        for idx, text in enumerate(texts):
            text_input = process_text(text, args.bert_pred)
            with torch.no_grad():
                pred = model(text_input)
            result = text_class_name(pred, classification)
            # Append the index to the appropriate category list
            category_indices[result].append(idx)
        for idx in meta_doi:
            name = idx.replace("/", "-")
            name = name.replace(":", "-")
            if name == doi_:
                doi=idx
                break
        id = meta_doi.index(doi)
        dic = {
            'doi': doi,
            'title':meta_title[id],
            'author':meta_author[id],
            'publisher':meta_publisher[id],
            'date':meta_date[id],
            'volume':meta_volume[id],
            'url':meta_url[id],
            'abstract':abstract,
            'sents_list':texts,
        }
        dic.update(category_indices)
        sorted_dict = dict(sorted(dic.items(), key=lambda item: item[0]))
        # dict_paragraph_class.append(dic)
        # filename = f'dict_class_{file_name}.json'
        with open(os.path.join(f'../output_data_time/{file_name}', f'{doi_}.json'), 'w', encoding='utf-8') as f:
            json_str=json.dumps(sorted_dict, ensure_ascii=False, indent=4)
            f.write('{}\n'.format(json_str))
    end = time.time()
    print(f"耗时为：{end - start} s")