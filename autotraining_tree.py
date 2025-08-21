
import torch
import numpy as np
import os
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
def train(root = "."):
        inputs = []
        targets = []
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".pt"):

                        path = dirpath+ "/"+filename
                        print(path)
                        data = torch.load(path,weights_only=False)
                        input_array = data["input"]
                        print(np.shape(input_array))
                        input = input_array.reshape(-1)
                        input = input.numpy()
                        target = data["output"]
                        target = target.numpy()
                        inputs.append(input)
                        targets.append(target)
        X = np.stack(inputs)
        y = np.stack(targets)


        X_train, X_test, y_train_frac, y_test_frac = train_test_split(X, y, test_size=0.2)
        y_train = np.zeros(np.shape(y_train_frac))
        y_test = np.zeros(np.shape(y_test_frac))
        for i in range(len(y_train_frac)):
                for j in range(len(y_train_frac[i])):
                        y_train[i][j] = round(y_train_frac[i][j])
        for i in range(len(y_test_frac)):
                for j in range(len(y_test_frac[i])):
                        y_test[i][j] = round(y_test_frac[i][j])
        base_tree = DecisionTreeClassifier(max_depth=3)
        clf = MultiOutputClassifier(base_tree)
        clf.fit(X_train, y_train)
        joblib.dump(clf, "modele_multioutput_classification.pkl")

        reg = MultiOutputRegressor(
                DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=0)
        ).fit(X_train, y_train_frac)
        joblib.dump(reg, "modele_multioutput_regression.pkl")
