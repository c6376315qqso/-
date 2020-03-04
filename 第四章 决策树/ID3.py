import pandas as pd
import numpy as np
import matplot as mp

label_of_cls = 'class'

class ID3:
    class Node:
        def __init__(self, tag=None):
            self.feature_name = None
            self.child = {}
            self.tag = tag
            self.main_cls = None

    def __init__(self):
        self.root = None

    def __entropy(self, dataset):
        # return entropy of given dataset
        _, cnt = np.unique(dataset[label_of_cls], return_counts=True)
        ent = 0
        for num in cnt:
            p = num / sum(cnt)
            ent += -p * np.log2(p)
        return ent

    def __gain(self, dataset, feature):
        # return gain of given feature and dataset
        ent_d = self.__entropy(dataset)
        uni_fea, cnt = np.unique(dataset[feature], return_counts=True)
        gain = ent_d
        for i in range(len(uni_fea)):
            data_v = dataset.loc[dataset[feature] == uni_fea[i]]
            gain -= (cnt[i] / sum(cnt)) * self.__entropy(data_v)
        return gain

    def choose_divide_feature(self, dataset):
        gain = [self.__gain(dataset, feature) for feature in dataset.columns if feature != label_of_cls]
        return dataset.columns[np.argmax(gain)]

    def __tree_generate(self, dataset):

        node = self.Node()
        uni_cls, cnt = np.unique(dataset[label_of_cls], return_counts=True)
        # if all sample belong to one class
        if len(cnt) == 1:
            node.tag = uni_cls[0]
            return node
        # if feature is empty or the dataset is all same in feature list
        if len(dataset.columns) == 1 or dataset.drop(label_of_cls, axis=1).drop_duplicates().shape[1] == 1:
            node.tag = dataset[label_of_cls].mode()[0]
            return node

        node.feature_name = self.choose_divide_feature(dataset)
        feature_val = np.unique(dataset[node.feature_name])
        main_tag = uni_cls[np.argmax(cnt)]
        node.main_cls = main_tag
        for val in feature_val:
            data_v = dataset[dataset[node.feature_name] == val].drop(node.feature_name, axis=1)
            if data_v.empty:
                node.child[val] = self.Node(main_tag)
            else:
                node.child[val] = self.__tree_generate(data_v)
        return node

    def fit(self, X_train, Y_train):
        dataset = pd.concat([X_train, Y_train], axis=1)
        self.root = self.__tree_generate(dataset)

    def __predict_one(self, sample, node):
        if node.tag is not None:
            return node.tag
        feature_val = sample[node.feature_name]
        try:
            nxt_node = node.child[feature_val]
            return self.__predict_one(sample, nxt_node)
        except:
            return node.main_cls

    def predict(self, X_test):
        Y_test = []
        for _, row in X_test.iterrows():
            Y_test.append(self.__predict_one(row, self.root))
        return Y_test

    def evaluate(self, X_test, Y_test):
        result = np.array(self.predict(X_test))
        Y_test = np.array(Y_test.to_list())
        return sum(result == Y_test) / len(result)



decision_tree = ID3()
data_train = pd.read_csv(r'C:\Users\Jame Black\Desktop\hayes-roth.data.csv', names=['~,', 'hobby', 'age', 'education level', 'marital status', 'class'])
data_train.drop(data_train.columns[0], axis=1, inplace=True)
data_test = pd.read_csv(r'C:\Users\Jame Black\Desktop\hayes-roth.test.csv', names=['hobby', 'age', 'education level', 'marital status', 'class'])

X_train = data_train.iloc[:, :-1]
Y_train = data_train.iloc[:, -1]
X_test = data_test.iloc[:, :-1]
Y_test = data_test.iloc[:, -1]
decision_tree.fit(X_train, Y_train)
print(decision_tree.evaluate(X_test, Y_test))
print(decision_tree.evaluate(X_train, Y_train))
