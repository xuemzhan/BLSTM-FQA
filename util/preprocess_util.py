import tensorflow as tf
import numpy as np
import pickle


def build_map(tokens, name):
    """
    生成id和tokens的映射
    :param tokens: 输入的一个序列，为一个列表，不带重复数据
    :param name: tokens的名称，用来保存为相应的文件
    :return:
    """
    id2tokens = {0: '<pad>'}
    tokens2id = {'<pad>': 0}
    count = 1
    for token in tokens:
        id2tokens[count] = token
        tokens2id[token] = count
        count += 1
    id2tokens[count] = '<new>'
    tokens2id['<new>'] = count
    pickle.dump(id2tokens, open('id2' + name, 'wb'))
    pickle.dump(tokens2id, open(name + '2id', 'wb'))



