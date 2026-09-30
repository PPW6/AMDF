import pandas as pd
from ollama import Client as oClient
import json
import re
from openpyxl import Workbook, load_workbook
####################################使用远程连接代理#######################################
####################################少样本示例测试#######################################
def api_generate(oclient, options, history,model,stream=False, format = ''):
    """
        调用 Llama模型生成回答，支持上下文记忆。
    """
    try:

        # 调用模型接口
        response = oclient.chat(
            model=model,
            messages=history,
            format=format,
            stream= stream,
            options=options,
        )
        # 非流式模式：直接返回完整内容
        if not stream:
            answer = response['message']['content']
            print(answer)
            # 将模型回答添加到历史记录

        else:
            print("生成内容中，请稍等...\n")
            answer = ""
            # 确保 message 和 content 存在，避免 KeyError
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    print(content, end="")  # 实时打印内容
                    answer += content
            print("\n生成完成。")
        return response
    except Exception as e:
        # 如果 e 对象包含响应信息
        if hasattr(e, 'status_code'):  # 检查是否有 status_code 属性
            print(f"An error occurred, error code: {e.status_code}")
        elif hasattr(e, 'error') and e.error:
            print(f"An error occurred, response details: {e.error}")
        else:
            # 默认错误信息
            print(f"An error occurred: {e}")
        return None

def read_xlsx(text_path, column_name):
    origin_data = pd.read_excel(text_path, engine='openpyxl')
    # Extract the specific column and drop rows where the column contains NaN values
    text = origin_data[column_name].dropna()
    return text

def save2json(save_xlsx, save_cut):
    json_list = []
    all_json = pd.read_excel(save_xlsx, engine='openpyxl')['extract_text']

    for i, items in enumerate(all_json):
        try:
            # 用正则提取 JSON
            match = re.search(r"\[.*\]", items, re.DOTALL)
            if match:
                json_obj = json.loads(match.group(0))  # 解析为 JSON
                json_list.extend(json_obj)  # 使用 extend 而不是循环和 append

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Excel: {e} - in item {i + 1}: {items}")
        except TypeError as e:
            print(f"Type error: {e} - possibly due to bad data format in item {i + 1}: {items}")
        except Exception as e:
            print(f"An unexpected error occurred: {e} - in item {i + 1}: {items}")

    try:
        wb = load_workbook(save_cut)
    except FileNotFoundError:
        wb = Workbook()
    if len(json_list) == 0:
        return
    # Select the active sheet
    sheet = wb.active

    key_title = [key for key in json_list[0]]
    # Write headers if the sheet is new
    if sheet.max_row == 1 and sheet.max_column == 1:
        sheet.append(key_title)

    # Calculate the starting row based on the offset
    start_row = sheet.max_row
    # Write each article to the sheet
    for i, item in enumerate(json_list):
        try:
            # 检查每个条目是否包含所有必要的键
            if not all(key in item for key in key_title):
                missing_keys = [key for key in key_title if key not in item]
                print(f"Skipping item {i + 1} due to missing keys: {missing_keys}")
                continue
            # 所有键存在，写入数据
            for n, key in enumerate(key_title):
                sheet.cell(row=start_row + i + 1, column=n + 1, value=item[key])
        except Exception as e:
            # 记录错误，但不中断整个过程
            print(f"Error processing item {i + 1}: {e}")
            continue
    # Save the workbook
    wb.save(save_cut)
    return

def save2xlsx(save_xlsx, all_res):
    df = pd.DataFrame(all_res, columns=['extract_text'])
    df.to_excel(save_xlsx, index=False)
    return

if __name__ == '__main__':
    oclient = oClient(host='Your URL')
    model = 'deepseek-r1:671b'
    model_name = model.replace(':', '_')
    options = {
        'temperature': 0,
        'max_tokens': 2000,
        'top_p': 0.9,
        'frequency_penalty': 0,
        'presence_penalty': 0
    }
    column_name = 'PP_text'  # 可选MP_text，PP_text，MPP_text
    text_path = './GPT_prompt/LLM_data.xlsx'
    few_num = '1'
    save_xlsx = f'./GPT_prompt/{few_num}/{model_name}/LLM_data_{column_name}.xlsx'
    save_cut = f'./GPT_prompt/{few_num}/{model_name}/LLM_data_{column_name}_divide.xlsx'

    texts = read_xlsx(text_path, column_name)
    all_res = []
    for i,text in enumerate(texts):
        # 初始化对话历史
        history = [
            {"role": "system",
             "content": """ You are a copper alloy expert and need to extract alloy, process, process_num, property and property_num information from text.
                            Do NOT include anything other than a json object in your output.
                            For sentences with multiple properties, create a separate JSON object for each property.
                            Include the original sentence in the "text" field of each JSON object.
                            If a field cannot be extracted, return "None".


                            Input Example 1:
                            "After solid solutionizing , rolling and aging at 450°C for 1 h , the Cu-Fe-Zr had a tensile strength of 515MPa and an conductivity of 72 % IACS ."
                            Output Example 1:
                            [
                                {
                                    "text": "After solid solutionizing , rolling and aging at 450°C for 1 h , the Cu-Fe-Zr had a tensile strength of 515MPa and an conductivity of 72 % IACS .",
                                    "alloy": "Cu-Fe-Zr",
                                    "process": "solid solutionizing, rolling, aging",
                                    "process_num": "None, None, 450°C for 1h",
                                    "property": "tensile strength",
                                    "property_num": "515MPa"
                                },
                                {
                                    "text": "After solid solutionizing , rolling and aging at 450°C for 1 h , the Cu-Fe-Zr had a tensile strength of 515MPa and an conductivity of 72 % IACS .",
                                    "alloy": "Cu-Fe-Zr",
                                    "process": "solid solutionizing, rolling, aging",
                                    "process_num": "None, None, 450°C for 1h",
                                    "property": "conductivity",
                                    "property_num": "72 % IACS"
                                }
                            ]
                            """
             },
            {"role": "user", "content": '什么是构效关系'}
        ]
        response = api_generate(oclient, options, history,model, stream=False, format='')
        all_res.append(response['message']['content'])

    save2xlsx(save_xlsx, all_res)
    save2json(save_xlsx, save_cut)
