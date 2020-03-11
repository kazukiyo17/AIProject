# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# input data
matrix = pd.read_csv('MSE.csv', sep='\t')


def rank_matrix(matrix):
    cnum = matrix.shape[1]
    rnum = matrix.shape[0]
    ## 升序排序索引
    sorts = np.argsort(matrix)
    for i in range(rnum):
        k = 1
        n = 0
        flag = False
        nsum = 0
        for j in range(cnum):
            n = n + 1
            ## 相同排名评分序值
            if j < 1 and matrix[i, sorts[i, j]] == matrix[i, sorts[i, j + 1]]:
                flag = True
                k = k + 1
                nsum += j + 1
            elif (j == 1 or (j < 1 and matrix[i, sorts[i, j]] != matrix[i, sorts[i, j + 1]])) and flag:
                nsum += j + 1
                flag = False
                for q in range(k):
                    matrix[i, sorts[i, j - k + q + 1]] = nsum / k
                k = 1
                flag = False
                nsum = 0
            else:
                matrix[i, sorts[i, j]] = j + 1
                continue
    return matrix


def friedman(n, k, rank_matrix):
    # 计算每一列的排序和
    sumr = sum(list(map(lambda x: np.mean(x) ** 2, rank_matrix.T)))
    # sumr = [26, 16.5, 17.5]
    result = 12 * n / (k * (k + 1)) * (sumr - k * (k + 1) ** 2 / 4)
    result = (n - 1) * result / (n * (k - 1) - result)
    return result


def nemenyi(n, k, q):
    return q * (np.sqrt(k * (k + 1) / (6 * n)))


matrix = np.array(matrix)
# matrix_r = rank_matrix(matrix.T)
matrix_r = np.array(matrix)
Friedman = friedman(10, 3, matrix_r)
CD = nemenyi(10, 3, 2.344)

rank_x = list(map(lambda x: np.mean(x), matrix.T))
name_y = ["KNN", "Forest", "SVM"]
min_ = [x for x in rank_x - CD / 2]
max_ = [x for x in rank_x + CD / 2]

plt.title("Friedman")
plt.scatter(rank_x, name_y)
plt.hlines(name_y, min_, max_)
plt.show()
