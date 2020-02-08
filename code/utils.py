from operator import itemgetter
import json



def load_dict(dict_path: str):
    """加载字典
    
    Args:
        dict_path (str): 字典路径
    
    Returns:
        dict: 字典
    """
    with open(dict_path, "r", encoding="utf-8") as fr:
        return json.load(fr)


def load_data(data_path: str, char_dict: dict, label_dict: dict):
    """加载数据

    Args:
        data_path (str): 数据路径
        char_dict (str): 字符字典
        label_dict (str): 标签字典

    Returns:
        tuple:
            seq_data: list，shape=(data_size,seq_len)
            label_data: 形状和seq_data一样
    """

    text_data = []
    label_data = []

    with open(data_path, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr.readlines()):
            line = eval(line)
            text = line['text']
            label = line['label']

            if len(text) > 1 and len(label) > 1 and len(text) == len(label):
                text = list(itemgetter(*text)(char_dict))
                label = list(itemgetter(*label)(label_dict))
                text_data.append(text)
                label_data.append(label)

    return text_data, label_data
