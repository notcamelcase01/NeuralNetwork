import numpy as np
import func as func
import pandas as pd

traindata = pd.read_csv('data/mnist_train.csv')
testdata = pd.read_csv('data/mnist_test.csv')
print(testdata.info())
testdataset = testdata.sample(n=10000).to_numpy()
Ytest = testdataset[:, 0]
Xtest = testdataset[:, 1:].T
Xtest = Xtest/255.
iv = np.load('train_data/train_data.npy',  allow_pickle=True)
y_hat = []
for i in range(10):
    z1, z2, a1, a2 = func.forward_propagation(Xtest, iv[i][0], iv[i][1], iv[i][2], iv[i][3])
    y_hat.append(a2)

y_hat = np.array(y_hat)
max_index = y_hat.argmax(axis=0)[0]
print(max_index)
print(Ytest)
counter = 0
for i in range(len(max_index)):
    if max_index[i] == Ytest[i]:
        counter += 1
print("Accuracy :", counter/len(max_index) * 100)