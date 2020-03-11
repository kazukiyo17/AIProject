# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# input data
twonorm_data = pd.read_csv('twonorm.csv', sep=',')

# show details
print(twonorm_data.head())
row_num, col_num = twonorm_data.shape
print(row_num, col_num)
print(twonorm_data.describe())

# 特征的标准化
data = twonorm_data.iloc[:, 0:col_num - 1]
target = twonorm_data.iloc[:, col_num - 1]
max_abs_scaler = preprocessing.MaxAbsScaler()  # [-1,1]
data = max_abs_scaler.fit_transform(data)  # 归一化

c = Counter(target)
print(c)

# split
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


def caiculateMSE(te_y, pre_y):
    te_y = np.array(te_y)
    tmp = te_y - pre_y
    tmp1 = tmp * tmp
    tmp2 = sum(tmp1) / len(te_y)
    return tmp2


# KNN
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


def KNN(tr_x, tr_y, te_x, te_y):
    score_lst = []
    for i in range(140, 160, 1):
        knn = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(knn, tr_x, tr_y, cv=10, scoring="accuracy").mean()
        score_lst.append(score)
    neighbors = (score_lst.index(max(score_lst)) * 1) + 140

    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(tr_x, tr_y)
    y_pred_knn = knn.predict(te_x)  # 预测
    score_train = knn.score(tr_x, tr_y)
    score_test = metrics.accuracy_score(te_y, y_pred_knn)
    print("KNN MSE=", caiculateMSE(te_y, y_pred_knn))
    one_hot = OneHotEncoder(sparse=False)
    y_true = one_hot.fit_transform(np.array(te_y).reshape(-1, 1))
    y_pred = one_hot.fit_transform(np.array(y_pred_knn).reshape(-1, 1))
    print("KNN CE=", log_loss(y_true, y_pred))
    # print("n_neighbors=", neighbors)
    print("KNN 训练集上的表现：", score_train)
    print("KNN 测试集上的表现：", score_test)
    print("KNN 混淆矩阵：", metrics.confusion_matrix(te_y, y_pred_knn))


# SVM
from sklearn import svm


def SVM(tr_x, tr_y, te_x, te_y):
    score_lst = []
    C_lst = [ 0.1, 1, 10, 100]
    for C_num in C_lst:
        svm_model = svm.SVC(C=C_num, probability=True)
        # svm_model.fit(tr_x, tr_y)
        # score = svm_model.score(tr_x, tr_y)
        # score_lst.append(score)
        score = cross_val_score(svm_model, tr_x, tr_y, cv=10, scoring="accuracy").mean()
        score_lst.append(score)
    C_best = C_lst[score_lst.index(max(score_lst))]

    svm_model = svm.SVC(C=C_best, probability=True)
    svm_model.fit(tr_x, tr_y)
    y_pred_svm = svm_model.predict(te_x)  # 预测
    score_test = metrics.accuracy_score(te_y, y_pred_svm)
    score_train = svm_model.score(tr_x, tr_y)
    print("SVM C=", 1)
    print("SVM 训练集上的表现：", score_train)
    print("SVM 测试集上的表现：", score_test)
    print("SVM MSE=", caiculateMSE(te_y, y_pred_svm))
    one_hot = OneHotEncoder(sparse=False)
    y_true = one_hot.fit_transform(np.array(te_y).reshape(-1, 1))
    y_pred = one_hot.fit_transform(np.array(y_pred_svm).reshape(-1, 1))
    print("KNN CE=", log_loss(y_true, y_pred))
    print("SVM 混淆矩阵：", metrics.confusion_matrix(te_y, y_pred_svm))


# 随机森林
from sklearn.ensemble import RandomForestClassifier


def TREE(tr_x, tr_y, te_x, te_y):
    score_lst = []
    for i in range(150, 170, 2):
        tree_model = RandomForestClassifier(n_estimators=i, random_state=90)
        score = cross_val_score(tree_model, tr_x, tr_y, cv=10).mean()
        score_lst.append(score)
    estimators = (score_lst.index(max(score_lst)) * 1) + 150
    # estimators = ([*range(175, 186)][score_lst.index(max(score_lst))])
    print("TREE n_estimators=", estimators, max(score_lst))
    return estimators


def tree_fit(estimators, tr_x, tr_y, te_x, te_y):
    # param_grid_depth = {'max_depth': np.arange(1, 20, 2)}
    # tree_model = RandomForestClassifier(n_estimators=estimators)
    # GS = GridSearchCV(tree_model, param_grid_depth, cv=10)
    # GS.fit(tr_x, tr_y)
    # depth = GS.best_params_
    # print("TREE max_depth=", GS.best_params_, GS.best_score_)

    tree_model = RandomForestClassifier(n_estimators=estimators, max_depth=17)
    tree_model.fit(tr_x, tr_y)
    y_pred_tree = tree_model.predict(te_x)  # 预测
    score_test = metrics.accuracy_score(te_y, y_pred_tree)
    score_train = tree_model.score(tr_x, tr_y)
    print("TREE MSE=",caiculateMSE(te_y,y_pred_tree))
    one_hot = OneHotEncoder(sparse=False)
    y_true = one_hot.fit_transform(np.array(te_y).reshape(-1, 1))
    y_pred = one_hot.fit_transform(np.array(y_pred_tree).reshape(-1, 1))
    print("KNN CE=", log_loss(y_true, y_pred))
    print("TREE n_estimators=", estimators)
    print("TREE 训练集上的表现：", score_train)
    print("TREE 测试集上的表现：", score_test)
    print("TREE 混淆矩阵：", metrics.confusion_matrix(te_y, y_pred_tree))


KNN(x_train, y_train, x_test, y_test)
SVM(x_train, y_train, x_test, y_test)
# estima = TREE(x_train, y_train, x_test, y_test)
tree_fit(159, x_train, y_train, x_test, y_test)
