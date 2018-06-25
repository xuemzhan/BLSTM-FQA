#-*- coding:utf-8 -*-
import pickle

def cut_vector(matrix_path):
    token2id = pickle.load(open('word2id', 'rb'))
    f2 = open('wiki1.zh.text.vector', 'a')
    with open(matrix_path, "r", encoding='utf-8') as infile:
        for row in infile:
            row = row.strip()
            items = row.split()
            token = items[0]
            if token in token2id.keys():
                f2.write(row)
                f2.write("\n")
    f2.close()

if __name__ == '__main__':
    emb_path = 'wiki.zh.text.vector'
    cut_vector(emb_path)