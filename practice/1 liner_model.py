import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])

def forward(x):
    return x * w

def myLoss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_list = []
mse_list = []

for w in np.arange(0.2, 4.1, 0.1):
    print("w = {}".format(w))
    total_loss = 0
    for x, y in zip(x_data, y_data):
        y_pred = forward(x)
        loss = myLoss(x, y)
        total_loss += loss
        print("\t{:.2f} {:.2f} {:.2f} {:.2f}".format(x, y, y_pred, loss))
    print("MSE = {:.2f}".format(total_loss / x_data.shape[0]))
    w_list.append(w)
    mse_list.append(total_loss / x_data.shape[0])

plt.figure(1)
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('weight')
plt.show()