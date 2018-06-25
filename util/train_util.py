import tensorflow as tf
import numpy as np
import pickle
#from sklearn.cross_validation import KFold

def get_embedding(matrix_path, token_name, dim=100):
    """
    获取数据的embedding represent。
    :param matrix_path: 预训练的词向量的路径。
    :param token_name: 要获取的token的名字，用来获得map
    :param dim: 词向量的维数，默认为100维
    :return: 获得的词向量表。
    """
    token2id = pickle.load(open(token_name + '2id', 'rb'))
    #id2token = pickle.load(open('id2' + token_name, 'rb'))
    # 默认为100维
    emb_matrix = np.zeros((len(token2id), dim))
    with open(matrix_path, "r", encoding='utf8') as infile:
        for row in infile:
            row = row.strip()
            items = row.split()
            token = items[0]
            emb_vec = np.array([float(val) for val in items[1:]])
            if token in token2id.keys():
                emb_matrix[token2id[token]] = emb_vec
    for i in range(1, len(token2id)):
        if emb_matrix[i][1] == 0:
            emb_matrix[i] = np.random.rand(dim)
    return emb_matrix


def shuffle_train_data(data):
    """
    将训练数据进行打乱
    :param data: 一个列表，用来保存所有要打乱的数据
    :return: 和data一样的格式
    """
    index = np.arange(len(data[0]))
    np.random.shuffle(index)
    new_data = []
    for i in data:
        new_data.append(i[index])
    return new_data


def get_next_batch(data, start_index, batch_size):
    """
    获得下一批数据。
    :param data: 保存要处理的data的列表
    :param start_index: 开始的索引
    :param batch_size: batch的大小
    :return: 一个该batch的列表
    """
    batch_data = []
    if start_index + batch_size >= len(data[0]):
        start_index = len(data[0]) - batch_size
    for i in data:
        batch_data.append(i[start_index: start_index + batch_size])
    return batch_data


def padding(sample, seq_max_len):
    """
    补0
    :param sample: 要处理的numpy list，二维
    :param seq_max_len: 最大长度
    :return: 补0后的numpy list
    """
    new_list = []
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
        new_list.append(sample[i])
    return np.array(new_list)


def get_cross_validation(data, n, shuffle=True):
    """
    获得交叉验证的训练集和测试集。
    :param data: 列表，输入的data
    :param n:
    :param shuffle:
    :return:
    """
    kflod = KFold(len(data[0]), n_folds=n, shuffle=shuffle)
    new_data_train = []
    new_data_val = []
    for i in data:
        new_cross_train = []
        new_cross_val = []
        for train_index, test_index in kflod:
            new_cross_train.append(i[train_index])
            new_cross_val.append(i[test_index])
        new_data_train.append(new_cross_train)
        new_data_val.append(new_cross_val)
    return new_data_train + new_data_val

if __name__ == '__main__':
    data = np.array([[i for i in range(10)] for j in range(20)])
    data2 = np.array([[2 * i for i in range(10)] for j in range(20)])

    train1, train2, val1, val2 = get_cross_validation([data, data2], 4)
    print(train2)
    print(val2)