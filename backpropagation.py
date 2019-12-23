import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
data = sio.loadmat("mnist_all.mat")

# preprocess the data
# We only consider the 0 and 1 data in the MNIST
X_train = np.concatenate((data['train0'], data['train1']), axis=0)
y_train = np.zeros([12665,1])
for i in range(5922, 12665):
    y_train[i] = 1
    
training_data = np.concatenate((X_train, y_train), axis=1)
np.random.shuffle(training_data) # shuffle them in order to better train the model
X_train = training_data[:,:784]
y_train = training_data[:,784:]
X_train = X_train/255 # divided by 255 to make the value small

X_test = np.concatenate((data['test0'], data['test1']), axis=0)
y_test = np.zeros([2115,1])
for i in range(989, 2115):
    y_test[i] = 1

testing_data = np.concatenate((X_test, y_test), axis=1)
# not necessarily shuffle the test data
X_test = testing_data[:,:784]
y_test = testing_data[:,784:]
X_test = X_test/255

def sigmoid(x):
    return 1/(1+np.exp(-x))

def grad_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

# This loss function cannot bear the 0 and 1 case, so pay attention to all 0 and 1 outcome, which 
# will result in inf, nan to stop the model working
def loss_func(fx, y):
    return -np.sum(y*np.log(fx) + (1-y)*np.log(1-fx))/fx.shape[0]
    
# logistic loss
def grad_loss_func(fx, y):
    return (-y/fx) + ((1-y)/(1-fx))

def normalize(row):
    u = np.mean(row)
    std = np.std(row)
    row = (row - u) / std
    return row

def forward(X_train, Wij, Wjk, Wko, b1, b2, b3):
    # apply sigmoid function at each layer, where we have input layer i,
    # hidden layer, k, output layer, which is 784 - 25 - 20 - 1
    tmp = np.matmul(X_train, Wij) # 100x50
    h1 = sigmoid(tmp + b1.T)
    tmp = np.matmul(h1, Wjk) # 100x25
    h2 = sigmoid(tmp + b2.T)
    tmp = np.matmul(h2, Wko) # 100x1
    xi = tmp + b3.T
    fx = sigmoid(xi)
    return fx, xi, h2, h1
    
def backward(h1, h2, xi, fx, Wij, Wjk, Wko, b1, b2, b3, a, X_train, y_train, size):
    dfx = grad_loss_func(fx, y_train)/size # 100x1
    dxi = grad_sigmoid(xi) # 100x1
    
    delta = dfx * dxi * h2 # 100x20
    Wko_new = Wko + a * (np.sum(delta, axis=0)/size).reshape([20,1]) # 20x1
    b3_new = b3 + a * (np.sum(dfx * dxi)/size) #1x1
    
    dh2 = dfx * dxi * Wko.reshape([1,20]) * (h2 * (1-h2)) # 100x20
    tmp1 = (np.sum(dh2, axis=0)/size).reshape([1,20])
    tmp2 = (np.sum(h1, axis=0)/size).reshape([25,1])
    delta = (tmp2 * tmp1).reshape([25, 20])
    Wjk_new = Wjk + a * delta # 25x20
    b2_new = b2 + a * tmp1.reshape([20,1]) # 20x1
    
    dh1 = np.sum(Wjk * tmp1, axis=1) # 25x1
    tmp3 = dh1.reshape([1,25]) * (h1 * (1-h1)) # 100x25
    tmp4 = (np.sum(tmp3, axis=0)/size).reshape([1,25]) # 125
    X_train = np.sum(X_train, axis=0).reshape([784,1]) # 1x784
    delta = X_train * tmp4
    Wij_new = Wij + a * delta
    b1_new = b1 + a * tmp4.reshape([25,1]) # 25x1
    
    return Wij_new, Wjk_new, Wko_new, b1_new, b2_new, b3_new
    
# total training image is 12665
np.random.seed(42)
Wij= np.random.randn(784, 25)
Wjk = np.random.randn(25, 20)
Wko = np.random.randn(20, 1)
b1 = np.zeros([25,1])
b2 = np.zeros([20,1])
b3 = np.zeros([1,1])
a = -29.3
size = 100
epoch = 0
epoch_list = []
accu_list = []
correct = 0
total = 0
loss = 0 
for i in range (0, 127):
    print ("reading data from", i*size, "to", (i+1)*size)
    X_batch = X_train[i*size:(i+1)*size, :]
    y_batch = y_train[i*size:(i+1)*size, :]
    fx, xi, h2, h1 = forward(X_batch, Wij, Wjk, Wko, b1, b2, b3)
    loss = loss + loss_func(fx, y_batch)/size
    Wij, Wjk, Wko, b1, b2, b3 = backward(h1, h2, xi, fx, Wij, Wjk, Wko, b1, b2, b3, a, X_batch, y_batch, size)
    fx[fx > 0.5] = 1
    fx[fx <= 0.5] = 0
    correct = correct + np.sum(fx == y_batch)
    total = total + fx.shape[0]
    if i%10 == 0:
        epoch = epoch + 1
        epoch_list.append(epoch)
        accu = correct / total
        accu_list.append(accu)
        correct = 0
        total = 0
        print ("---------------------------------")
        print ("epoch:", epoch)
        print ("accuracy:", accu)
        print ("loss:", loss)
        print ("---------------------------------")
        loss = 0

# testing
fx, xi, h2, h1 = forward(X_test, Wij, Wjk, Wko, b1, b2, b3)
fx[fx > 0.5] = 1
fx[fx <= 0.5] = 0
test_accu = np.sum(fx == y_test) / fx.shape[0]
print ("test accuracy:", test_accu)

# plotting the accuracy on training set
import matplotlib.pyplot as plt
plt.plot(epoch_list, accu_list)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


