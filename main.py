import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

# 데이터 로드
data = pd.read_csv("./(지반)data.csv", names = ["재료","롤러반복 횟수","Elwd","Ev1","Ev2"])
data = np.array(data, dtype=np.float32)

# print(data)

# X : 재료,롤러반복횟수,Elwd / Y : Ev1,Ev2
X = data[:, :-2]
Y = data[:, -2:]


W = tf.Variable(tf.random.normal([3,2]))
b = tf.Variable(tf.random.normal([2]))


learning_rate = 0.000001

def predict(X):
    return tf.matmul(X,W) + b

n_epochs = 10000
for i in range(n_epochs+1):
    #record the gradient of the cost function
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X) - Y)))
    #Loss 계산
    W_grad, b_grad = tape.gradient(cost, [W,b])
    #update parameters (W and b)
    W.assign_sub(learning_rate*W_grad)
    b.assign_sub(learning_rate*b_grad)

    if i % 100 == 0:
        print("{:5}|{:10.4f}".format(i, cost.numpy()))
    
predicted_list = predict(X)
measured_list = Y

measured_ev1 = measured_list[:, 0]
measured_ev2 = measured_list[:, 1]

predicted_ev1 = predicted_list[:, 0]
predicted_ev2 = predicted_list[:, 1]

# ev1_r = polyfit(predicted_ev1, measured_ev1, 1000)
# ev2_r = polyfit(predicted_ev2, measured_ev2, 1000)

min_ev1 = min(Y[:, 0])
max_ev1 = max(Y[:, 0])

min_ev2 = min(Y[:, 1])
max_ev2 = max(Y[:, 1])


f1 = plt.figure(1)
plt.plot(np.arange(min_ev1, max_ev1), np.arange(min_ev1, max_ev1), 'r')
plt.plot(predicted_ev1, measured_ev1, 'o')
f1.show()

f2 = plt.figure(2)
plt.plot(np.arange(min_ev2, max_ev2), np.arange(min_ev2, max_ev2), 'r')
plt.plot(predicted_ev2, measured_ev2, 'o')
f2.show()








