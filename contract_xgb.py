from matplotlib import pyplot as plt
import xgboost as xgb
from numpy import loadtxt
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, recall_score
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import datetime
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

print(len(y_test))
params = {
    'booster': 'gbtree',
    'learning_rate': 0.4,
    'gamma': 0.1,
    'max_depth': 20,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'objective': 'multi:softmax',
    'random_state': 7,
    'slient': 0,
    'num_class': 2,
    'eta': 0.8,
    'importance_type' : 'gain'
}

plst = list(params.items())

start_time = time.clock()
dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)

dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)
# ans_p = model.predict_proba(dtest)
# model = XGBClassifier()
# model.fit()
# ans = model.predict(y_test)
end_time = time.clock()
time = end_time - start_time

cnt1 = 0
cnt2 = 0

for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1
ravel = confusion_matrix(y_test, ans)
tp, fn, fp, tn = ravel.ravel()
acc = (accuracy_score(y_test,ans)*100)
fnr = ((fn/(tp+fn))*100)
fpr = ((fp/(fp+tn))*100)
precision = ((tp/(fp+tp))*100)
recall = metrics.recall_score(y_test, ans)
f1 = metrics.f1_score(y_test, ans)
print("Accuracy: %.2f %%" % (accuracy_score(y_test, ans) * 100))
print("RECALL")
print(recall_score(y_test, ans))
print("F1")
print(metrics.f1_score(y_test, ans))
print("avg_TIME:")
avg_time = time/(len(y_train) + len(y_test))
print(avg_time)


