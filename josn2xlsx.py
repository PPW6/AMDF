
####################筛选仅包含mpa的用于机器学习建模的元素表########################################
import os
import json
process_path = '../output_data/process'
property_path = './output_data/property'
length = os.listdir(property_path)
length = [len for len in length if 'RE' in len]
output_excel='./output_data/data.xlsx'
re_length = os.listdir(property_path)
re_length = [len for len in re_length if 'EX' in len]
relation_ = []
for i in range(0, len(re_length)):
    file_name = re_length[i]
    print(file_name)
    with open(os.path.join(property_path, file_name), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        dict = [json.loads(line.strip()) for line in lines]
        relation_ += dict
relation_ = [item for item in relation_ if item['alloy'] != 'None' and item['property_num'] != 'None']
relation_mpa = []
for num in relation_:
    if num['alloy'] != 'None' and 'mpa' in num['property_num'].lower():  # 接下来用hobbs算法解决B-MAT和B-DSC的问题！！！！ 主要丢失MAT太多,在NER识别完之后就用hobbs算法。
        relation_mpa.append(num)
import pandas as pd
import re

# 读取 Excel 文件
ml_data = pd.read_excel('./output_data/filtered_data.xlsx', engine='openpyxl')

# 提取合金名称数据
alloy_names = ml_data['alloy'].tolist()

# 元素周期表列表
elements_list = ['Cu', 'Cr', 'Zr', 'Ti', 'Cd', 'Co', 'Si', 'Sn', 'Zn', 'Ag', 'S', 'Sb', 'Ni', 'Mg', 'Bi', 'Sc', 'Al',
                 'Fe', 'V', 'Be', 'Y', 'B', 'O', 'Mo', 'Ge', 'Mn', 'P', 'Nb', 'Ca', 'Ce', 'Tr', 'RE', 'Al2O3']
elements_pattern = '|'.join(elements_list)

def extract_composition(alloy_name):
    composition = {}
    total_content = 0.0

    # 第一种情况：匹配含量先于元素的情况
    pattern_1 = re.compile(r'([0-9]{0,2}\.?\d+)[-\s%wt pct]*(' + elements_pattern + r')')
    matches_1 = pattern_1.findall(alloy_name)

    if matches_1 and alloy_name[-1].isalpha():
        for match in matches_1:
            content, element = match
            content = float(content)
            composition[element] = content
            total_content += content

        if 'Cu' not in composition:
            composition['Cu'] = 100 - total_content
        else:
            composition['Cu'] = max(100 - total_content, 0)
        return composition

    # 第二种情况：匹配元素先于含量的情况
    pattern_2 = re.compile(r'(' + elements_pattern + r')([0-9]{0,2}\.?\d+)')
    matches_2 = pattern_2.findall(alloy_name)
    if matches_2 and 'Cu' in alloy_name:
        for match in matches_2:
            element, content = match
            content = float(content)
            composition[element] = content
        return composition

    # 如果没有找到任何匹配，返回一个空字典
    return {}

# 提取合金成分并存储在 DataFrame 中
composition_data = []
for alloy_name in alloy_names:
    if type(alloy_name) != float:
        composition = extract_composition(alloy_name)
        composition_data.append(composition)
    else:

        composition_data.append({})
# 使用 fillna(0) 确保空值用 - 填充
df = pd.DataFrame(composition_data).fillna('-')

# 确保列名符合元素列表
df = df.reindex(columns=elements_list, fill_value='-')
# 将 composition_df 插入到 ml_data 的 'alloy' 列后面
ml_data = pd.concat([ml_data.iloc[:, :3], df, ml_data.iloc[:, 3:]], axis=1)
# 保存为 Excel 文件
ml_data.to_excel('./output_data/alloy_composition.xlsx', index=False)

print(df)


