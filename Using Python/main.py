import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv")
ages = data['age'].values
charges = (data['charges'].values)/100
print("Ages size :",ages.size ," \nCharges size :", charges.size)
m = ages.size
theta1 = 0.1
theta2 = 0.2
lr = 0.3
th_arr = np.array([[theta1, theta2]]).T
inputs = np.array([ages]).T
Y_out = np.array([charges]).T
Y = np.ones((m , 1))

H = inputs * Y_out
print("In :")
print(inputs)
print(H)

#train the data
inte = 50
def predict(X, Th):
    y = np.zeros(shape=(m, 1))
    o1 = np.ones((m, 1))
    xb = np.column_stack((o1, X))
    y = np.dot(xb, Th)
    return y
def cost(Y , Y_h):
    err = (Y - Y_h) * (Y - Y_h)
    return np.sum(err)/(2*m)
def gradient(X,Y ,Th):
    Pr = predict(X , Th)
    err = (Pr - Y)
    g1 = (np.dot(np.ones(shape=(1, m)),err))/m
    g2 = (np.dot(X.T , err))/m
    return np.concatenate((g1 , g2) , axis= 0)

for i in range(inte):
    outp = predict(ages, th_arr)
    # backprobagation
    c = cost(outp, Y_out)
    Grad = gradient(inputs, Y_out, th_arr)
    th_arr = th_arr - lr * Grad


Th0 = th_arr[0, 0]
Th1 = th_arr[1, 0]

i = 0
for v in ages:
    plt.scatter(v , charges[i] )
    i += 1


xplt = np.linspace(0,50,100)
yplt = Th1*xplt+Th0
plt.plot(xplt, yplt, '-r', label='y=2x+1')
plt.title('Lin')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()