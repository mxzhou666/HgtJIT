import csv
import json
import os
import re
import sys
import time

import dgl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from hgt_model import HGTClassfication, HGTTest, HGTTrain, GraphDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from models import CodeChangeModel, build_load_model
from transformers import RobertaTokenizer

from collections import defaultdict
import random
import concurrent.futures
from functools import partial
import pickle
csv.field_size_limit(500 * 1024 * 1024)
import nltk
import string
from nltk.tokenize import word_tokenize
# nltk.download('punkt')

logsPath = './logs_hgt/'
# testPath = './testdata/'
testPath = './dataset/testdata_ast/'
mdlsPath = './models/'

def file_operate(path=testPath):
    train_ids = []
    test_ids = []
    new_commit = []
    with open('../../CodeJIT-main/Data/data_split/train_time_id.txt', 'r') as file:
        line = file.readline()
        while line:
            train_ids.append(line.replace('\n',''))
            line = file.readline()
    with open('../../CodeJIT-main/Data/data_split/test_time_id.txt', 'r') as file:
        line = file.readline()
        while line:
            test_ids.append(line.replace('\n',''))
            line = file.readline()
    # with open('dataset/data_split/new_commit.txt', 'r') as file:
    #     line = file.readline()
    #     while line:
    #         new_commit.append(line.replace('\n',''))
    #         line = file.readline()
    # train_files = []
    # test_files = []
    # for id in new_commit:
    #     if id in train_ids:
    #         train_files.append(id)
    #     elif id in test_ids:
    #         test_files.append(id)
    files = []
    for root, _, filelist in os.walk(testPath):
        # if len(labels) >= 50:
        #     break
        # if root !=  './dataset/testdata_ast/0072d2a9fce4835ab2b9ee70aaca0169fb25fa0c':
        #     continue
        # print('load data:' + str(len(labels)))
        for file in filelist:
            # if file[-7:] == '_np.npz' and root.split('/')[-1] in new_commit:
            if file[-7:] == '_np.npz':
                files.append(root.split('/')[-1])
                # if root.split('/')[-1] in train_ids:
                #     train_files.append(root.split('/')[-1])
                # elif root.split('/')[-1] in test_ids:
                #     test_files.append(root.split('/')[-1])
            # if file[-9:] == '_cct5.npz' or file[-8:] == '_mid.npz' or file[-11:] == '_cct5v2.npz':
            #     os.remove(os.path.join(root, file))
            #     file_size = os.path.getsize(os.path.join(root, file))
            #     if file_size > 50000000:
            #         print('remove:' + root)
            #         os.remove(os.path.join(root, file))
            #         print(convert_bytes(file_size))

    # random.seed(21)
    # # random.seed(24)
    #
    # # Create an index list
    # num_graphs = len(files)
    # index_list = list(range(num_graphs))
    #
    # # Shuffle the index list
    # random.shuffle(index_list)
    #
    # # Define the split ratios (e.g., 80% train, 10% validation, 10% test)
    # train_ratio = 0.8
    # # val_ratio = 0.1
    # val_ratio = 0.2
    # # test_ratio = 0.1
    #
    # # Calculate the sizes of each split
    # train_size = int(num_graphs * train_ratio)
    # val_size = int(num_graphs * val_ratio)
    #
    # # Split the index list into train, validation, and test sets
    # train_indices = index_list[:train_size]
    # val_indices = index_list[train_size:train_size + val_size]
    # test_indices = index_list[train_size + val_size:]
    #
    # # Use the indices to get the corresponding graphs and labels
    #
    # train_files = [files[i] for i in train_indices]
    # val_files = [files[i] for i in val_indices]
    # test_files = [files[i] for i in test_indices]

    with open('dataset/data_split/new_commit_time_train.txt', 'w') as file:
        for id in files:
            if id in train_ids:
                file.write(id + '\n')
    with open('dataset/data_split/new_commit_time_test.txt', 'w') as file:
        for id in files:
            if id in test_ids:
                file.write(id + '\n')


    # with open('dataset/data_split/train_random21_id.txt', 'w') as file:
    #     for id in train_files:
    #         file.write(id + '\n')
    #
    # # with open('../../CodeJIT-main/Data/data_split/val_random21_id.txt', 'w') as file:
    # #     for id in val_files:
    # #         file.write(id + '\n')
    #
    # with open('dataset/data_split/test_random21_id.txt', 'w') as file:
    #     for id in val_files:
    #         file.write(id + '\n')

    print(111)

def convert_bytes(byte_size):
    # 列出不同单位的标签
    labels = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    while byte_size >= 1024 and i < len(labels) - 1:
        byte_size /= 1024.0
        i += 1
    return f"{byte_size:.2f} {labels[i]}"

def load_data():
    train_ids = []
    val_ids = []
    test_ids = []
    # with open('../../CodeJIT-main/Data/data_split/train_random21_id.txt', 'r') as file:
    # with open('dataset/data_split/train_random21_id.txt', 'r') as file:
    # with open('dataset/data_split/new_commit_cross_train.txt', 'r') as file:
    with open('dataset/data_split/new_commit_time_train.txt', 'r') as file:
        line = file.readline()
        while line:
            train_ids.append(line.replace('\n', ''))
            line = file.readline()
    # with open('../../CodeJIT-main/Data/data_split/val_random_id.txt', 'r') as file:
    #     line = file.readline()
    #     while line:
    #         val_ids.append(line.replace('\n', ''))
    #         line = file.readline()
    # with open('../../CodeJIT-main/Data/data_split/test_random42_id.txt', 'r') as file:
    # with open('dataset/data_split/new_commit_cross_test.txt', 'r') as file:
    with open('dataset/data_split/new_commit_time_test.txt', 'r') as file:
        line = file.readline()
        while line:
            test_ids.append(line.replace('\n', ''))
            line = file.readline()


    ids = []
    labels = []
    msgs = []
    codes = []
    data = []
    true_cnt = 0
    false_cnt = 0
    multi_fun = 0
    new_commit = []  # 只有单函数更改组成的数据集
    with open('./dataset/C_dataset.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            commit_id = row[0]
            if commit_id == 'commit_id':
                continue
            if commit_id not in train_ids:
            # if commit_id not in test_ids:
                continue
            # if commit_id != '08cb69e870c1b2fdc3574780a3662b92bfd6ef79':
            #     continue
            # print('开始处理'+commit_id)
            commit_url = row[1]
            label = row[2]
            # if label == '0':
            #     false_cnt += 1
            #     continue
            # else:
            #     true_cnt += 1
            commit_message = row[3]
            # if 'CVE' not in commit_message:
            #     continue
            # if 'free' not in commit_message:
            #     continue
            filename = row[4]
            date = row[5]
            prev_code = row[6]
            # if len(prev_code) > 1400 or len(prev_code) <= 1200:
            #     continue
            patch_code = row[7]
            diff = row[8]

            # 多函数
            # if(extract_functions(diff) != 1):
            #     multi_fun += 1
            #     # print(commit_id)
            #     continue
            # if label == '0':
            #     false_cnt += 1
            # else:
            #     true_cnt += 1
            # print(commit_id)

            new_commit.append(commit_id)
            # if len(diff) > 450 or len(diff) <= 420:
            #     i = 1
            #     continue
            # CCT5
            dict_1 = find_diff2(diff)
            dict_1['oldf'] = ''
            dict_1['msg'] = commit_message
            dict_1['y'] = int(label)
            data.append(dict_1)
            # print(commit_id)

            # deepjit & cc2vec
            # ids.append(commit_id)
            # labels.append(int(label))
            # msgs.append(commit_message)
            # # codes.append(find_diff(diff))
            # codes.append(find_diff2(diff))
    # print("multi_fun:", multi_fun)
    # print("true_cnt:", true_cnt)
    # print("false_cnt", false_cnt)
    # print("total_cnt", true_cnt + false_cnt)
    # with open('dataset/data_split/new_commit.txt', 'w') as file:
    #     for id in new_commit:
    #         file.write(id + '\n')

    # deepjit
    # data = [ids, labels, msgs, codes]
    # # 初始化字符到数字映射
    char_to_number = {}
    current_number = 0

    # for text in msgs:
    #     # 分词并去除标点符号
    #     words = [word.lower() for word in word_tokenize(text) if word not in string.punctuation]
    #
    #     for word in words:
    #         if word not in char_to_number:
    #             char_to_number[word] = current_number
    #             current_number += 1
    # char_to_number['<NULL>'] = current_number
    # current_number += 1
    # dict_msg = char_to_number.copy()
    # for code in codes:
    #     for text in code:
    #     # 分词并去除标点符号
    #         words = [word.lower() for word in word_tokenize(text) if word not in string.punctuation]
    #
    #         for word in words:
    #             if word not in char_to_number:
    #                 char_to_number[word] = current_number
    #                 current_number += 1
    # dictionary = {'dict_msg': dict_msg,'dict_code': char_to_number}
    # dictionary = [dict_msg, char_to_number]
    # # 打印字符到数字映射
    # print(dictionary)
    # with open('../baselines/data/dataset_dict.pkl', 'wb') as file:
    #     pickle.dump(dictionary, file)

    # cct5
    with open('../baselines/data_cct5_time/changes_train_fixed.jsonl', 'w') as file:
    # with open('../baselines/data_cct5_time/changes_test_fixed.jsonl', 'w') as file:
        json.dump(data, file)

    # deepjit
    # with open('../baselines/data_cross/features_test.pkl', 'wb') as file:
    # with open('../baselines/data_time/features_train.pkl', 'wb') as file:
    #     pickle.dump(data, file)
    # with open('../baselines/data/features_test.pkl', 'rb') as file:
    #     data = pickle.load(file)
    #     print('111')

def extract_functions(diff_text):
    function_names = set()
    diff_lines = diff_text.split('\n')

    for line in diff_lines:
        if line.startswith('@@'):
            # 使用正则表达式匹配函数名或代码块
            matches = re.findall(r'@@.*?(\b\w+\b)\(', line)
            function_names.update(matches)
    return len(function_names)

# deepJIT  不分增删
def find_diff(text):
    lines = text.split('\n')
    selected_lines = [line[1:].strip() for line in lines if
                      (line.startswith('-') or line.startswith('+'))
                      and (not line.startswith('--') and not line.startswith('++'))
                      and len(line.strip()) > 1]
    return selected_lines

    # 打印提取的行
    # for line in selected_lines:
    #     print(line)

# cc2vec 分增删
def find_diff2(text):
    lines = text.split('\n')
    list = []
    add_lines = [line[1:].strip() for line in lines if
                      line.startswith('-')
                      and not line.startswith('--')
                      and len(line.strip()) > 1]
    removed_lines = [line[1:].strip() for line in lines if
                      line.startswith('+')
                      and not line.startswith('++')
                      and len(line.strip()) > 1]

    # cct5
    add_line = ''
    removed_line = ''
    for addline in add_lines:
        add_line += addline+'\n'
    for removedline in removed_lines:
        removed_line += removedline+'\n'
    dict = {'added_code':add_line, 'removed_code': removed_line}
    return dict

    # deepjit cc2vec
    # dict = {'added_code': add_lines, 'removed_code': removed_lines}
    # list.append(dict)
    # return list

    # 打印提取的行
    # for line in selected_lines:
    #     print(line)

if __name__ == '__main__':

    # file_operate(testPath)
    load_data()



