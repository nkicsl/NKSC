from numpy import loadtxt
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
import csv
from matplotlib import pyplot as plt

# 数据导入
# 训练集
dataset_train = loadtxt('train.csv', delimiter=",", encoding="utf-8")
X_train = dataset_train[:, 0:12]
y_train = dataset_train[:, 12]
#测试集
dataset_test = loadtxt('test.csv', delimiter=",", encoding="utf-8")
X_test = dataset_test[:, 0:12]
y_test = dataset_test[:, 12]

#模型训练
model = KNeighborsClassifier(n_neighbors= 30)
start_time = time.perf_counter()
model.fit(X_train, y_train)
ans1 = model.predict(X_test)
end_time = time.perf_counter()
avg_time = (end_time-start_time)/(len(y_train)+len(y_test))
ans1 = ans1.astype(np.int64)

#参数打印
ravel = confusion_matrix(y_test,ans1)
tp, fn, fp, tn = ravel.ravel()
acc = (accuracy_score(y_test,ans1)*100)
fnr = ((fn/(tp+fn))*100)
fpr = ((fp/(fp+tn))*100)
recall = metrics.recall_score(y_test, ans1)
f1 = metrics.f1_score(y_test, ans1)
print("Accuracy: %.2f %%" % (accuracy_score(y_test,ans1)*100))
print("RECALL")
print(metrics.recall_score(y_test, ans1))
print("F1")
print(metrics.f1_score(y_test, ans1))
print("time")
print(avg_time)
precision = ((tp/(fp+tp))*100)
