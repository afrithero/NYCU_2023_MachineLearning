import numpy as np
import sys
import matplotlib.pyplot as plt

def gaussian_data_generator(mean,var):
    sum = np.sum(np.random.uniform(0,1,12)) - 6
    # sum ~ N(0,1)
    # Affine property -> am1 + b = m2, m1 = 0 -> b = m2
    # a^2var0 = var2 -> a = sqrt(var2)
    # y = ax+b 
    b = mean
    a = np.sqrt(var)
    sum = a * sum + b
    return sum

def poly_linear_data_generator(order, bias, weights):
    x = np.random.uniform(-1,1)
    y = 0
    for i in range(order):
        y += weights[i]*(x**i)
    y += gaussian_data_generator(0, bias)
    return x, y

class BayesianLinearRegression():
    def __init__(self, alpha, beta=1/4):
        self.alpha = alpha
        self.beta = beta

    # 在這個階段，由於不會一次得到所有資料，所以會透過 online learning 更新權重
    # 因為 prior 是高斯分佈，根據共軛性，posterior 也是高斯分佈
    # alpha 來自於 prior, beta 來自於 likelihood
    def fit(self,design_matrix,y_matrix,count):
        # count = 1 時，從 N(0,b^-1I) 抽樣 w 的 Prior
        if count == 1:
            dim = design_matrix.shape[0]
            self.posterior_cov = np.linalg.inv(self.alpha*np.eye(dim) + self.beta*design_matrix.T@design_matrix)
            self.posterior_mean = self.beta*self.posterior_cov@design_matrix.T@y_matrix
        else:
            last_posterior_precision = np.linalg.inv(self.posterior_cov)
            self.posterior_cov = np.linalg.inv(last_posterior_precision + self.beta*design_matrix.T@design_matrix)
            self.posterior_mean = self.posterior_cov@(last_posterior_precision@self.posterior_mean + self.beta*design_matrix.T@y_matrix) 

        print("Posterior mean:")
        for i in self.posterior_mean:
            print("{:.5f}".format(i[0]))
        print()
        print("Posterior variance:")
        for row in self.posterior_cov:
            print(",".join(["{:.8f}".format(i) for i in row]), end="")
            print()
        print()

    # 用 fit 好的 weight, 產出 predictive output，因為 fit 的權重會一直被更新，所以 predictive distribution 也會形成一個 predictive distribution
    def predict(self, design_matrix):
        mean_predicted = design_matrix@self.posterior_mean
        # 會是一個純量
        var_predicted = 1/self.beta + design_matrix@self.posterior_cov@design_matrix.T
        return mean_predicted, var_predicted

alpha = int(sys.argv[1]) # 影響 initial prior 精度的純量
order = int(sys.argv[2])
bias = int(sys.argv[3]) # 產生線性資料時會用到的的 bias(noise) 
weights = list(map(float, sys.argv[4].split(",")))
last_var_predicted = 0
count = 0
model = BayesianLinearRegression(alpha = alpha)
x_record = []
y_record = []

while True: 
    count += 1
    x, y = poly_linear_data_generator(order, bias, weights)
    x_record.append(x)
    y_record.append(y)

    print("Add data pint: ({},{})".format(x,y),end="\n")
    design_matrix = np.array([[x**i for i in range(order)]])
    y_matrix = np.array([[y]])
    model.fit(design_matrix,y_matrix, count)
    mean_predicted, var_predicted = model.predict(design_matrix)
    
    print("Predictive distribution ~ N({}, {})".format(mean_predicted[0][0], var_predicted[0][0]), end="\n")
    print("-----------------------------------------------------")
    if abs(var_predicted - last_var_predicted) < 1e-7:
        break

    last_var_predicted = var_predicted

    if count == 10:
        x_record_10 = x_record.copy()
        y_record_10 = y_record.copy()
        weights_mean_10 = model.posterior_mean
        var_predicted_10 = var_predicted[0][0]
    elif count == 50:
        x_record_50 = x_record.copy()
        y_record_50 = y_record.copy()
        weights_mean_50 = model.posterior_mean
        var_predicted_50 = var_predicted[0][0]

figure = plt.figure()

figure.add_subplot(221)
plt.title("Ground truth")
poly_func = np.poly1d(weights)
x_curve = np.linspace(-2.0,2.0,30)
y_curve = poly_func(x_curve)
plt.plot(x_curve,y_curve,'k')
plt.plot(x_curve,y_curve + 1/model.beta, 'r')
plt.plot(x_curve,y_curve - 1/model.beta, 'r')

figure.add_subplot(222)
plt.title("Predict result")
poly_func = np.poly1d(model.posterior_mean.squeeze())
x_curve = np.linspace(-2.0,2.0,30)
y_curve = poly_func(x_curve)
print(var_predicted)
plt.scatter(x_record,y_record, marker='o', c='m', alpha = 0.5)
plt.plot(x_curve,y_curve,'k')
plt.plot(x_curve,y_curve + var_predicted[0][0], 'r')
plt.plot(x_curve,y_curve - var_predicted[0][0], 'r')

figure.add_subplot(223)
plt.title("After 10 incomes")
poly_func = np.poly1d(weights_mean_10.squeeze())
x_curve = np.linspace(-2.0,2.0,30)
y_curve = poly_func(x_curve)
plt.scatter(x_record_10,y_record_10, marker='o', c='m', alpha = 0.5)
plt.plot(x_curve,y_curve,'k')
plt.plot(x_curve,y_curve + var_predicted_10, 'r')
plt.plot(x_curve,y_curve - var_predicted_10, 'r')

figure.add_subplot(224)
plt.title("After 50 incomes")
poly_func = np.poly1d(weights_mean_50.squeeze())
x_curve = np.linspace(-2.0,2.0,30)
y_curve = poly_func(x_curve)
plt.scatter(x_record_50,y_record_50, marker='o', c='m', alpha = 0.5)
plt.plot(x_curve,y_curve,'k')
plt.plot(x_curve,y_curve + var_predicted_50, 'r')
plt.plot(x_curve,y_curve - var_predicted_50, 'r')

plt.tight_layout()
plt.show()