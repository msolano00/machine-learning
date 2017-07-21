# Import dependencies and training data
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

# Raw data and test data
iris = load_iris()
test_idx = [0, 50, 100]

# Training set
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Test set
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Desicion tree classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# Test results :D
print(test_target)
print(clf.predict(test_data))

# Viz
import pydotplus 
#from ipython.display import Image
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf")
