import numpy as np


class HMM_NER:
    """隐马尔科夫模型
    """

    def __init__(self, n_chars: int, n_labels: int):
        """初始化

        Args:
            n_chars (int): 词典大小
            n_labels (int): 标签类别数量
        """
        self.n_labels = n_labels
        self.n_chars = n_chars

        # 初始化转移矩阵A、发射矩阵B、初始矩阵Pi
        self.transition = np.zeros((self.n_labels, self.n_labels))
        self.emission = np.zeros((self.n_labels, self.n_chars))
        self.pi = np.zeros(self.n_labels)
        # 偏置，用来防止log(0)或乘0的情况
        self.epsilon = 1e-8

    def fit(self, text_data: np.ndarray, label_data: np.ndarray):
        """训练模型

        Args:
            text_data (np.ndarray): 观测序列
            label_data (np.ndarray): 隐藏状态
        """
        # 估计转移概率矩阵
        self.estimate_transition_and_initial_probs(label_data)
        # 估计发射概率矩阵 初始概率矩阵
        self.estimate_emission_probs(text_data, label_data)

        # 取log防止计算结果下溢
        self.pi = np.log(self.pi)
        self.transition = np.log(self.transition)
        self.emission = np.log(self.emission)

    def estimate_transition_and_initial_probs(self, label_data: np.ndarray):
        """计算转移概率矩阵和初始概率矩阵 P(Y_t+1 | Y_t)
        """
        print("estimate_transition_and_initial_probs")
        for line in label_data:
            # 统计初始状态
            self.pi[line[0]] += 1
            # 统计转移状态
            for cur, nxt in zip(line[:-1], line[1:]):
                self.transition[cur, nxt] += 1

        self.pi[self.pi == 0] = self.epsilon
        self.pi /= np.sum(self.pi)

        self.transition[self.transition == 0] = self.epsilon
        self.transition /= np.sum(self.transition, axis=1, keepdims=True)

    def estimate_emission_probs(self, text_data: np.ndarray, label_data: np.ndarray):
        """计算发射矩阵 P(Observation | Hidden_state)
        """
        print("estimate_emission_probs")
        for seq_line, lbl_line in zip(text_data, label_data):
            for char, lbl in zip(seq_line, lbl_line):
                self.emission[lbl, char] += 1
        self.emission[self.emission == 0] = self.epsilon
        self.emission /= np.sum(self.emission, axis=1, keepdims=True)

    def predict(self, text_idx):
        """预测算法

        Args:
            text_idx (list): 已经转成索引的句子数据，整型列表
        Returns:
            list: 隐状态列表，整型列表
        """
        return viterbi_decode(self.transition, self.emission, self.pi, text_idx)


def viterbi_decode(transition: np.ndarray, emission: np.ndarray, pi: np.ndarray, sequence: list):
    """维特比解码算法

    Args:
        transition (np.ndarray): 状态转移矩阵，已经做过log
        emission (np.ndarray): 发射矩阵，已经做过log
        pi (np.ndarray): 初始状态矩阵，已经做过log
        sequence (list): 已经转成索引的句子数据，整型列表
    """

    seq_len = len(sequence)
    hidden_state_size = transition.shape[0]

    # 初始化T1表格和T2表格
    T1 = np.zeros(hidden_state_size)
    T2 = np.zeros((hidden_state_size, seq_len))

    # 初始发射概率
    start_p_Obs_State = emission[:, sequence[0]]

    # T1表格和T2表格第一步的值
    # 已经做过log，直接相加
    T1 = pi + start_p_Obs_State
    T2[:, 0] = np.nan

    for i in range(1, seq_len):
        char = sequence[i]
        prev_score = np.expand_dims(T1, axis=-1)
        p_Obs_State = np.expand_dims(emission[:, char], axis=0)

        curr_score = prev_score + p_Obs_State + transition

        T1 = np.max(curr_score, axis=0)
        T2[:, i] = np.argmax(curr_score, axis=0)

    # 回溯寻路
    best_label = int(np.argmax(T1))
    labels = [best_label]
    for i in range(seq_len - 1, 0, -1):
        best_label = int(T2[best_label, i])
        labels.append(best_label)
    return list(reversed(labels))
