from utils import load_data, load_dict
from HMM_model import HMM_NER
from operator import itemgetter

if __name__ == "__main__":

    char_dict_path = 'C:/Users/Horie/PythonWorkspace/Algorithms/hmm/data/char.dic'
    label_dict_path = 'C:/Users/Horie/PythonWorkspace/Algorithms/hmm/data/label.dic'
    data_path = 'C:/Users/Horie/PythonWorkspace/Algorithms/hmm/data/train_data.txt'

    char_dict = load_dict(char_dict_path)
    label_dict = load_dict(label_dict_path)
    train_text, train_label = load_data(data_path, char_dict, label_dict)

    print(len(train_label), "条数据")

    # 字典大小
    n_chars = len(char_dict)
    n_labels = len(label_dict)
    print(n_chars)

    # 定义模型
    model = HMM_NER(n_chars, n_labels)
    # 训练模型
    model.fit(train_text, train_label)

    text = "昨天在世卫组织执委会第146届会议上，世卫组织总干事谭德塞再次强调，防范是必要的，但无需过度反应。世卫组织不建议各国采取任何旅行或者贸易限制措施，呼吁各国采取基于证据、令人信服的措施。"
    text_idx = itemgetter(*text)(char_dict)

    result = model.predict(text_idx)

    idx2lbl = {v: k for k, v in label_dict.items()}
    for i, j in zip(text, result):
        print(i, ' | ', idx2lbl[j])
