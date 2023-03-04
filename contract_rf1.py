from numpy import loadtxt
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import csv

# 数据导入
# 训练集
dataset_train = loadtxt("train.csv", delimiter=",", encoding="utf-8")
X_train = dataset_train[:, 0:12]
y_train = dataset_train[:, 12]
#测试集
dataset_test = loadtxt("test.csv", delimiter=",", encoding="utf-8")
X_test = dataset_test[:, 0:12]
y_test = dataset_test[:, 12]

forest = RandomForestClassifier(n_estimators=1000,random_state=1,n_jobs=-1)
start_time = time.clock()
forest.fit(X_train,y_train)
end_time = time.clock()
avg_time = (end_time-start_time)/(len(y_train)+len(y_test))
ans1 = forest.predict(X_test)
ans1 = ans1.astype(np.int64)


ravel = confusion_matrix(y_test,ans1)
tp, fn, fp, tn = ravel.ravel()
acc = (accuracy_score(y_test,ans1)*100)
fnr = ((fn/(tp+fn))*100)
fpr = ((fp/(fp+tn))*100)
recall = metrics.recall_score(y_test, ans1)
f1 = metrics.f1_score(y_test, ans1)
precision = ((tp/(fp+tp))*100)
print("Accuracy: %.2f %%" % (accuracy_score(y_test,ans1)*100))
print("RECALL")
print(metrics.recall_score(y_test, ans1))
print("F1")
print(metrics.f1_score(y_test, ans1))
print("time")
print(avg_time)


