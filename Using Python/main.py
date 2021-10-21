import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("california_housing_train.csv")
ages = data['total_rooms'].values
charges = (data['median_house_value'].values)/100

#print("Ages size :",ages.size ," \nCharges size :", charges.size)

DATA_COUNT = 600

XX = ages[:DATA_COUNT]
YY = charges[:DATA_COUNT]


ages = (XX)/1000
charges = (YY)/1000

plt.plot(XX , YY, 'r*')


m = ages.size
theta1 = 0.15
theta2 = 0.25
lr = 0.001
th_arr = np.array([[theta1, theta2]]).T

print(th_arr)
print("\n inputs ")

inputs = (np.array([ages]).T)

print(inputs)
print("\n Y_OUT")
Y_out = (np.array([charges]).T)
print(Y_out)
# print("\n H")
# #Y = np.ones((m , 1))
# H = inputs * Y_out
# print(H)
print("\n Predict :")

inte = 600

def predict(X, Th):
    y = np.zeros(shape=(m, 1))
    o1 = np.ones((m, 1))
    xb = np.column_stack((o1, X))
    # print(xb)
    # print("\ndot")
    y = np.dot(xb, Th)
    return y




def cost(Y , Y_h):
    err = (Y - Y_h) * (Y - Y_h)
    cc = (np.sum(err))/(2*m)
    return cc



#print(cost(predict(ages , th_arr) , Y_out))

def gradient(X,Y ,Th):
    Pr = predict(X , Th)
    err = (Pr - Y)
    g1 = (np.dot(np.ones(shape=(1, m)),err))/m
    g2 = (np.dot(X.T , err))/m
    return np.concatenate((g1 , g2) , axis= 0)



for i in range(inte):
    th_arr = th_arr - lr * gradient(inputs , Y_out ,  th_arr)
    print(th_arr)
    

xplt = np.linspace(0,15000,1400)
yplt = th_arr[0] * xplt + th_arr[1]
plt.plot(xplt,yplt)
#plt.axis([0, 6, 0, 20])




# xplt = np.linspace(0,50,100)
# yplt = Th1*xplt+Th0
# plt.plot(xplt, yplt, '-r', label='y=2x+1')
# plt.title('Lin')
# plt.xlabel('x', color='#1C2833')
# plt.ylabel('y', color='#1C2833')
# plt.legend(loc='upper left')
# plt.grid()
plt.show()