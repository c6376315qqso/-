import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

cls1, cls2 = 'won', 'nowin'

class decision_stump:
    def __init__(self):
        self.feature_name = None
        self.child = {}

    def train(self, X_train, Y_train, w):
        cls = np.unique(Y_train)
        min_error = 123465789
        for feature in X_train.columns:
            feature_val = np.unique(X_train[feature])
            child = {}
            error = 0
            for val in feature_val:
                idx = X_train[feature] == val
                cost = [sum((Y_train != c) * w * idx) for c in cls]
                child[val] = cls[np.argmin(cost)]
                error += min(cost)
            if error < min_error:
                min_error = error
                self.child = child
                self.feature_name = feature
        return min_error

    def predict_one(self, sample):
        return self.child[sample[self.feature_name]]

    def predict(self, X_test):
        Y_test = []
        for _, row in X_test.iterrows():
            Y_test.append(self.predict_one(row))
        return Y_test


class Adaboost:
    def __init__(self):
        self.base_learner = []
        self.alpha = []
        self.num_learner = None

    def train(self, X_train, Y_train, m):
        self.num_learner = m
        w = [1/X_train.shape[0]] * X_train.shape[0]
        Y_trans = [1 if i == cls1 else -1 for i in Y_train]
        for i in range(m):
            ds = decision_stump()
            error = ds.train(X_train, Y_train, w)
            self.base_learner.append(ds)
            self.alpha.append(1/2 * np.log(1 / error - 1))
            pred = ds.predict(X_train)
            pred = [1 if i == cls1 else -1 for i in pred]
            w = [w[j] * np.exp(-self.alpha[-1] * Y_trans[j] * pred[j]) for j in range(len(pred))]
            w /= sum(w)

    def predict_one(self, sample):
        res = 0
        for i in range(self.num_learner):
            res += self.alpha[i] * (1 if self.base_learner[i].predict_one(sample) == cls1 else -1)
        return cls1 if res >= 0 else cls2

    def predict(self, X_test):
        Y_test = []
        for _, row in X_test.iterrows():
            Y_test.append(self.predict_one(row))
        return Y_test


def precision(res1, res2):
    unequal = [res1[i] == res2[i] for i in range(len(res1))]
    return sum(unequal) / len(res1)


data = pd.read_csv(r'dataset\kr-vs-kp.data', names=[str(i+1) for i in range(37)])

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
ab = Adaboost()
ab.train(X_train, Y_train, 10)

print('训练集准确率：', precision(ab.predict(X_train), list(Y_train)))
print('测试集准确率：', precision(ab.predict(X_test), list(Y_test)))
