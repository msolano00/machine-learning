from sklearn import tree

# Features:
# 1 = bumpy
# 0 = smooth
features = [[150, 1], [170, 1], [140, 0], [130, 0]]

# Labels:
# 1 = orange
# 0 = apple
labels = [1, 1, 0, 0]

# Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# Prints the prediction of the classifier
print(clf.predict(160, 1))
