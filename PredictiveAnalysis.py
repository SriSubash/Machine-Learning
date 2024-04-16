#SVM WITH PACKAGE:

from sklearn import svm
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("your_csv_file",nrows=5000);

X = data[['label1','label2']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

def plot_decision_boundary(X, y, clf):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Plot data points
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, palette="Set1", legend=False)

    # Plot decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.set_xlabel('vap')
    ax.set_ylabel('wet')
    plt.title('Decision Boundary')

# Plot data distribution
def plot_data_distribution(X, y):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Plot data points
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, palette="Set1", legend=False)
    plt.xlabel('vap')
    plt.ylabel('wet')
    plt.title('Data Distribution')

# Call the functions to plot
plot_decision_boundary(X_train, y_train, clf)
plt.show()

plot_data_distribution(X_train, y_train)
plt.show()

#LINEAR REGRESSION WITH PACKAGE:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

data = pd.read_csv("your_csv_file",nrows=5000);

X = data[['label1','label2']]
y = data['target']

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

print(y_pred)

mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

#LINEAR REGRESSION WITHOUT PACKAGE:

import numpy as np
import pandas as pd

def linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    return theta

data = pd.read_csv("your_csv_file",nrows=5000);

X = data[['label1','label2']]
y = data['target']

theta = linear_regression(X, y)

X_b = np.c_[np.ones((X.shape[0], 1)), X]

y_pred = X_b.dot(theta)

print(y_pred)

mse = np.mean((y_pred - y) ** 2)
print("Mean Squared Error:", mse)

# Scatter plot for actual vs predicted values
def plot_actual_vs_predicted(y, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y, y=y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

# Residual plot
def plot_residuals(y, y_pred):
    plt.figure(figsize=(8, 6))
    sns.residplot(x=y_pred, y=y - y_pred, lowess=True, line_kws={'color': 'red', 'lw': 1})
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

# Call the functions to plot
plot_actual_vs_predicted(y, y_pred)
plot_residuals(y, y_pred)

#KNN WITH PACKAGE:

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = pd.read_csv("your_csv_file",nrows=5000);

X = data[['label1','label2']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size = 0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))

def plot_decision_boundary(X, y, classifier, title):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    h = .02  # step size in the mesh
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot the training points
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel('vap')
    plt.ylabel('wet')
    plt.show()

# Plot decision boundary for KNN classifier
plot_decision_boundary(X_train, y_train, knn, 'KNN Decision Boundary')

#KNN WITHOUT PACKAGE:

import math
import pandas as pd

def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

def get_neighbors(training_set, test_instance, k):
    distances = []
    for training_instance in training_set:
        dist = euclidean_distance(test_instance, training_instance[:-1])
        distances.append((training_instance, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def predict_classification(training_set, test_instances, k):
    predictions = []
    for test_instance in test_instances:
        neighbors = get_neighbors(training_set, test_instance, k)
        class_votes = {}
        for neighbor in neighbors:
            label = neighbor[-1]
            if label in class_votes:
                class_votes[label] += 1
            else:
                class_votes[label] = 1
        sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
        predictions.append(sorted_votes[0][0])
    return predictions

def calculate_accuracy(actual_classes, predicted_classes):
    correct_predictions = 0
    for actual, predicted in zip(actual_classes, predicted_classes):
        if actual == predicted:
            correct_predictions += 1
    accuracy = correct_predictions / len(actual_classes)
    return accuracy

data = pd.read_csv("your_csv_file", nrows=5000)
training_set = data[['label1', 'label2', 'target']].values.tolist()
test_instances = data[['label1', 'label2']].values.tolist()

k = 2
predictions = predict_classification(training_set, test_instances, k)
print('Predicted classes:', predictions)

actual_classes = data['lumpy'].tolist()
accuracy = calculate_accuracy(actual_classes, predictions)
print('k-NN Score:', accuracy)
