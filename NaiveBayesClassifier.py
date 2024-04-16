#NAIVE BAYES CLASSIFICATION WITH PACKAGE

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv("your_csv_file")

X = data[['label1','label2']].values
y = data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = GaussianNB()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#----------------------------------------------------------------------------------------

#NAIVE BAYES CLASSIFICATION WITHOUT PACKAGE

import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.class_likelihoods = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = np.zeros(len(self.classes))
        self.class_likelihoods = []

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = len(X_c) / len(X)

            # Calculate mean and standard deviation for each feature
            class_likelihood = [(np.mean(feature), np.std(feature)) for feature in X_c.T]
            self.class_likelihoods.append(class_likelihood)

    def _likelihood(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def _predict_instance(self, x):
        posteriors = []

        for i, c in enumerate(self.classes):
            prior = np.log(self.class_priors[i])
            likelihood = np.sum(np.log(self._likelihood(x, mean, std)) for x, (mean, std) in zip(x, self.class_likelihoods[i]))
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        predictions = [self._predict_instance(x) for x in X]
        return np.array(predictions)

data = pd.read_csv("your_csv_file")

X = data[['label1','label2']].values
y = data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = NaiveBayes()
nb_classifier.fit(X_train, y_train)

predictions = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
