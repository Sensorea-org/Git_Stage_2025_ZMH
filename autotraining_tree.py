
import torch
import numpy as np
import os
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

root = "C:/Users/hugom/OneDrive/Documents/Stage_2025/dev_Cnn/dataset/"
inputs = []
targets = []
for dirpath, dirnames, filenames in os.walk(root):
    for filename in filenames:
        if filename.endswith(".pt"):

                path = root + "/" + filename
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
base_tree = DecisionTreeClassifier(max_depth=3)
clf = MultiOutputClassifier(base_tree)
clf.fit(X_train, y_train)
joblib.dump(clf, "modele_multioutput_test.pkl")

