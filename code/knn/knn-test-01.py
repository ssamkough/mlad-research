import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('../../datasets/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True) # removes the id column to not skew results

x = np.array(df.drop(['class'], 1)) # features - everything except the class column
y = np.array(df['class']) # labels - just the class

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) # quickly shuffle the data and separate it into training and testing chunks

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)


accuracy = clf.score(X_test, y_test)
print(accuracy)
