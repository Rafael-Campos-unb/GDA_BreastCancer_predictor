import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import os

classes = ['Malignant', 'Benign']
file = pd.read_csv('Breast Cancer Wisconsin (Diagnostic) Data Set.csv')
y = file['diagnosis']
y = [0 if i == 'B' else 1 for i in y]
print(y)
file = file.drop('diagnosis', axis=1)
file = (file - file.mean()) / file.std()
X = file
X = X[['radius_mean', 'texture_mean']]
print(X.iloc[:, 0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print(file)
clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(predictions)
accuracy = accuracy_score(y_test, predictions) * 100
precision = precision_score(y_test, predictions) * 100
recall = recall_score(y_test, predictions) * 100
# Print Metrics
print(accuracy)
print(precision)
print(recall)


# plot decision boundaries results
def plot_decision_boundaries(X, y, model, title):
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    ax = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.legend(handles=ax.legend_elements()[0], labels=[i for i in classes], loc='lower right')
    plt.xlabel('radius_mean')
    plt.ylabel('texture_mean')


plt.figure(figsize=(10, 4))
# Plot decision boundaries for QDA
plot_decision_boundaries(X_test, y_test, clf, "QDA Decision Boundary")

plt.tight_layout()
plt.savefig('QDA_decision_Boundary.png')
plt.show()

