import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('your_csv_file')
print(dataset.head())

X = pd.DataFrame(dataset.iloc[:, 5:16].values)
y = dataset.iloc[:, 19].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ",cm)
print("Accuracy Score: ")
accuracy_score(y_test,y_pred)
