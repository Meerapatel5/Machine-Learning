#!/usr/bin/env python
# -*- coding:utf-8 -*-
import csv
import time
from random import random
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, average_precision_score
from sklearn.model_selection import KFold
import os
from generatedata import GenerateData
import sklearn.metrics as skm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import json

COLOR = {-1: "r", 1: "b"}
master_original = "D:/Machine Learning/LearningWithNoisyLabelsFINAL/LearningWithNoisyLabels/TestingData/master_train.csv"
master_test = "D:/Machine Learning/LearningWithNoisyLabelsFINAL/LearningWithNoisyLabels/TestingData/master_test.csv"
'''
DataSet A needs master_test.csv & master_train.csv
DataSet B needs master_test_b.csv & master_train_b.csv
'''
final_accuracy = []
now = int(time.time())
feature_size = int(42)   #DataSet:A - 42 & DataSet:B - 22
final_structure_feature_size = int(43)      #DataSet:A -43 & DataSet:B -23
no_of_class = 11

def get_all_scores(test, one_to_all):
    raw_data = pd.read_csv(master_test)
    result = raw_data["Labels"].values.tolist()
    for x in range(0, len(result)):
        if int(result[x]) == one_to_all:
            result[x] = 1
        else:
            result[x] = -1

    ans = skm.f1_score(result, test, average='weighted')
    return ans


def test_result_list(one_to_all):
    raw_data = pd.read_csv(master_test)
    result = raw_data["Labels"].values.tolist()
    for x in range(0, len(result)):
        if int(result[x]) == one_to_all:
            result[x] = 1
        else:
            result[x] = -1
    return result


def view_data(filename):
    raw_data = pd.read_csv(filename)
    columns = list(raw_data)
    print(columns)
    return None


def generate_feature_list(filename, select_column):
    raw_data = pd.read_csv(filename)
    columns = list(raw_data)
    resulted_list = [select_column] + columns[:feature_size]
    return resulted_list


def get_data_from_csv(filename, one_to_all, select_column):
    raw_data = pd.read_csv(filename)
    feature_list = generate_feature_list(filename, select_column)
    result = raw_data[feature_list].values.tolist()
    for x in range(0, len(result)):
        if int(result[x][0]) == one_to_all:
            result[x][0] = 1
        else:
            result[x][0] = -1
    return result


class TrainingModel(object):
    def __init__(self, train_column, test_column, one_to_all, data_size, is_random, po1, po2, dim=3, ws=(0.5, 0.5)):
        self.one_to_all = one_to_all
        self.data_maker = GenerateData(data_size)
        self.true_data_map = {}
        if is_random == 1:
            noise_free_data, self.n1, self.n2 = self.data_maker.random_data(dim, ws)
            self.set_random = True
            self.init_true_data_map(noise_free_data)
            noised_data = self.data_maker.add_noise(noise_free_data, po1, po2)
            self.noised_train_set, self.noised_test_set = self.data_maker.split_data(noised_data)

        elif is_random == 2:
            noise_free_data = self.data_maker.original_data()
            self.n1 = self.n2 = self.data_maker.data_size / 2
            self.set_random = False
            self.init_true_data_map(noise_free_data)
            noised_data = self.data_maker.add_noise(noise_free_data, po1, po2)
            self.noised_train_set, self.noised_test_set = self.data_maker.split_data(noised_data)

        elif is_random == 3:
            noise_free_data = get_data_from_csv(master_original, one_to_all, select_column="B")
            self.set_random = True
            self.init_true_data_map(noise_free_data)
            self.noised_train_set = get_data_from_csv(master_original, one_to_all, select_column=train_column)
            self.noised_test_set = get_data_from_csv(master_test, one_to_all, select_column=test_column)

        self.nosiy_test_map = {tuple(d[1:final_structure_feature_size]): d[0] for d in self.noised_test_set}
        self.unbiased_loss_pred_map = {}

    def init_true_data_map(self, data):
        for d in data:
            self.true_data_map[tuple(d[1:final_structure_feature_size])] = d[0]

    def trainByNormalSVM(self, train_set):
        train_X = [tuple(d[1:final_structure_feature_size]) for d in train_set]
        train_y = [d[0] for d in train_set]
        clf = svm.SVC()
        clf.fit(train_X, train_y)
        test_X = [tuple(d[1:final_structure_feature_size]) for d in self.noised_test_set]
        pred_y = clf.predict(test_X)
        self.unbiased_loss_pred_map = {(xy[0], xy[1]): int(label) for label, xy in zip(pred_y, test_X)}
        print("5. Classifer has been trained!")
        return clf

    def selectClfByKFold(self, po1, po2):
        min_Rlf = float('inf')
        target_dataset = None
        data = np.array(self.noised_train_set)
        kf = KFold(n_splits=2)
        for train, test in kf.split(data):
            size = len(train)
            tr_data = data[train]
            p_y = 1.0 * sum(1 for d in tr_data if d[0] == -1) / size
            py = 1.0 * sum(1 for d in tr_data if d[0] == 1) / size
            Rlf = []
            for d in tr_data:
                try:
                    x = self.true_data_map[tuple(d[1:final_structure_feature_size])]
                    y = int(d[0])
                    Rlf.append(self.estLossFunction(x, y, py, p_y, po1, po2))
                except Exception as e:
                    pass
            if np.mean(Rlf) < min_Rlf:
                min_Rlf = np.mean(Rlf)
                target_dataset = tr_data
        print("4. Cross-validation finished!")
        return self.trainByNormalSVM(target_dataset)

    def lossFunction(self, fx, y):
        result = 0 if fx == y else 1
        return result

    def estLossFunction(self, x, y, py, p_y, po1, po2):
        p1, p2 = po1, po2
        return ((1 - p_y) * self.lossFunction(x, y) - py * self.lossFunction(x, -y)) / (1 - p1 - p2)

    def accuracy(self, pred, rst):
        match_cnt, all_cnt = 0, 0
        for a, b in zip(pred, rst):
            all_cnt += 1
            if a == b:
                match_cnt += 1
        rst = round(1.0 * match_cnt / all_cnt, 4)
        return rst

    def comparison_plot(self, clf, po1, po2, show_plot=False):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)

        # plot1
        x_o = [d[1] for d in self.noised_test_set]
        y_o = [d[2] for d in self.noised_test_set]
        label_o = []
        try:
            for d in self.noised_test_set:
                label_o.append(self.nosiy_test_map[tuple(d[1:final_structure_feature_size])])
        except Exception as e:
            pass

        color_o = [COLOR[d] for d in label_o]
        ax1.scatter(x_o, y_o, marker='+', c=color_o,
                    s=20, edgecolor='y')
        ax1.set_title('Noise-free')

        # plot2
        print("Final")
        x_n = x_o
        y_n = y_o
        label_n = [d[0] for d in self.noised_test_set]
        color_n = [COLOR[d] for d in label_n]
        ax2.scatter(x_n, y_n, marker='+', c=color_n,
                    s=20, edgecolor='y')
        ax2.set_title('Noise rate:' + str(po1) + " and " + str(po2))

        # plot3
        x_t1 = [tuple(d[1:final_structure_feature_size]) for d in self.noised_test_set]
        x_t1_graphics = x_o
        y_t1 = y_o
        x_t1_np = np.array(x_t1)
        x_t1_np.reshape(-1, 1)
        pred_label1_master = clf.predict(x_t1)
        pred_label1 = [int(x) for x in pred_label1_master]
        label_p1 = [COLOR[d] for d in pred_label1]
        rst1 = self.accuracy(label_o, pred_label1)
        unbaised_accuracy = skm.accuracy_score(test_result_list(self.one_to_all), pred_label1)
        print("Unbiased accuracy:--> " + str(unbaised_accuracy))
        unbaised_f1_score = get_all_scores(pred_label1, self.one_to_all)
        print("Unbaised f1 score:--> " + str(unbaised_f1_score))
        ax3.scatter(x_t1_graphics, y_t1, marker='+', c=label_p1,
                    s=20, edgecolor='y')
        ax3.set_title('Accuracy:' + str(rst1))

        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        if show_plot:
            plt.show()

        return unbaised_accuracy, unbaised_f1_score, test_result_list(self.one_to_all), pred_label1


def modular_test(one_to_all, error, train_column, test_column):
    cur_path = os.curdir
    os.chdir(cur_path)
    os.chdir("..")
    n, is_random, po1, po2 = 20, 3, error, error
    run = TrainingModel(train_column, test_column, one_to_all, n, is_random, po1, po2)
    clf = run.selectClfByKFold(po1, po2)
    accuracy, final_f1_score, true_result, predicted_result = run.comparison_plot(clf, po1, po2, show_plot=True)
    del run
    return accuracy, final_f1_score, true_result, predicted_result


if __name__ == "__main__":
    universal_column_name = ['B', 'C5', 'C10', 'C15', 'C20', 'C25', 'C30', 'C35', 'C40', 'A5', 'A10', 'A15', 'A20',
                             'A25', 'A30', 'A35', 'A40', 'N5', 'N10', 'N15', 'N20', 'N25', 'N30', 'N35', 'N40']
    universal_column_error = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                              0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    universal_class_list = [x for x in range(1, 11, 1)]  #For DataSet A - range(1, 11, 1), for DataSet B - range(1, 12, 1)
    universal_test_column_result = 'Labels'
    headers = ["True", "Predicted"]

    '''
    Test_column_name and test_column_error are siblings.
    They need same thing from parents or they cry.
    '''
    test_column_name = ['B']   #Take each column 1 by 1 from universal_column_name
    test_column_error = [0]    #Take corresponding universal_column_error 
    test_class_list = [10]     #Dataset:A - [10] & Dataset:B - [11]
    
    universal_result = {}
    for single_class in universal_class_list:
        temp_list = []
        for column, error in zip(test_column_name, test_column_error):        #two columns together
            try:
                accuracy, final_f1_score, true_result, predicted_result = modular_test(one_to_all=single_class,
                                                                                       error=error,
                                                                                       train_column=column,
                                                                                       test_column=universal_test_column_result)
                #print(accuracy, final_f1_score)
                temp_list.append({
                    "column": str(column),
                    "accuracy": str(accuracy),
                    "f1_score": str(final_f1_score),
                })

                df = pd.DataFrame(list(zip(true_result, predicted_result)),
                                  columns=['True', 'Predicted'])
                df.to_csv(str("D:/Machine Learning/LearningWithNoisyLabelsFINAL/LearningWithNoisyLabels/src/results"
                              "/final_result_" + str(now) + str(single_class) + str(column) + ".csv"),  
                          sep='\t',                 
                          encoding='utf-8')     

                
                '''
                final_result_B for DatasetB & final_result_ for DatasetA  #to separate each column
                '''
                
            except Exception as e:
                print(
                    "Error at class --> " + str(single_class) + " & column --> " + str(column) + " Error --> " + str(e))

        universal_result[single_class] = temp_list

    with open(str("D:/Machine Learning/LearningWithNoisyLabelsFINAL/LearningWithNoisyLabels/src/results/final_result_" + str(
            now) + ".json"), 'w', encoding='utf-8') as write_file:
        json.dump(universal_result, write_file, ensure_ascii=False, indent=4)
