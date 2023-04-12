import numpy as np
import func as func
import pandas as pd


traindata = pd.read_csv('data/mnist_train.csv')
testdata = pd.read_csv('data/mnist_test.csv')
print(testdata.info())
traindataset = traindata.sample(n=1000).to_numpy()
Ytrain = traindataset[:, 0]
Xtrain = traindataset[:, 1:].T
print(Ytrain.shape, Xtrain.shape)
Xtrain = Xtrain/255.
m_train = Ytrain.shape[0]

n0, n1, n2 = Xtrain.shape[0], 4, 1
alpha = 0.2
max_iter = 30000

train_labels0 = np.array([1 if i == 0 else 0 for i in Ytrain]).reshape(1, m_train)
train_labels1 = np.array([1 if i == 1 else 0 for i in Ytrain]).reshape(1, m_train)
train_labels2 = np.array([1 if i == 2 else 0 for i in Ytrain]).reshape(1, m_train)
train_labels3 = np.array([1 if i == 3 else 0 for i in Ytrain]).reshape(1, m_train)
train_labels4 = np.array([1 if i == 4 else 0 for i in Ytrain]).reshape(1, m_train)
train_labels5 = np.array([1 if i == 5 else 0 for i in Ytrain]).reshape(1, m_train)
train_labels6 = np.array([1 if i == 6 else 0 for i in Ytrain]).reshape(1, m_train)
train_labels7 = np.array([1 if i == 7 else 0 for i in Ytrain]).reshape(1, m_train)
train_labels8 = np.array([1 if i == 8 else 0 for i in Ytrain]).reshape(1, m_train)
train_labels9 = np.array([1 if i == 9 else 0 for i in Ytrain]).reshape(1, m_train)

iv = []

iv.append(func.descent(Xtrain, train_labels0, max_iter, alpha, n0, n1, n2, m_train))
iv.append(func.descent(Xtrain, train_labels1, max_iter, alpha, n0, n1, n2, m_train))
iv.append(func.descent(Xtrain, train_labels2, max_iter, alpha, n0, n1, n2, m_train))
iv.append(func.descent(Xtrain, train_labels3, max_iter, alpha, n0, n1, n2, m_train))
iv.append(func.descent(Xtrain, train_labels4, max_iter, alpha, n0, n1, n2, m_train))
iv.append(func.descent(Xtrain, train_labels5, max_iter, alpha, n0, n1, n2, m_train))
iv.append(func.descent(Xtrain, train_labels6, max_iter, alpha, n0, n1, n2, m_train))
iv.append(func.descent(Xtrain, train_labels7, max_iter, alpha, n0, n1, n2, m_train))
iv.append(func.descent(Xtrain, train_labels8, max_iter, alpha, n0, n1, n2, m_train))
iv.append(func.descent(Xtrain, train_labels9, max_iter, alpha, n0, n1, n2, m_train))
iv = np.array(iv, dtype=object)
np.save('train_data/train_data.npy', iv)
