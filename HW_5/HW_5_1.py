import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class GaussionProcess():
	def __init__(self):
		pass
	
	@staticmethod
	def create_kernel(X_m, X_n, alpha, sigma, length_scale):
		# ||X_m - X_n|| -> 求每一對點的距離
		# X_train -> 34x34, X_train,X_test -> 34x1000
		# 萬一長度不一樣，reshape(-1,1) 可以將維度轉換成 row 的數量，然後 col num 為 1
		dist = np.sum(X_m ** 2, axis=1).reshape(-1, 1) + np.sum(X_n ** 2, axis=1) - 2 * X_m @ X_n.T
		kernel = (sigma**2) * ((1 + dist / (2 * alpha * (length_scale ** 2))) ** (-1 * alpha))
		return kernel
	
	@staticmethod
	def neg_marginal_loglikelihood(args, X_train, Y_train, beta):
		alpha, sigma, length_scale = args
		n_samples = X_train.shape[0]
		kernel_train = GaussionProcess.create_kernel(X_train, X_train, alpha, sigma, length_scale)
		cov_train = kernel_train + (1/beta)*np.eye(X_train.shape[0])
		cov_train_inverse = np.linalg.inv(cov_train) 
		# 本來全都帶負號，再加上負號就變全部為正
		result = (1/2)*Y_train.T@cov_train_inverse@Y_train + (1/2)*np.log(np.linalg.det(cov_train)) + (n_samples/2)*np.log(2*np.pi)
		return result
	
	def fit(self, X_train, Y_train, X_test, beta=5, alpha=1, sigma=1, length_scale=1, optimizer=False):
		if optimizer:
			f = GaussionProcess.neg_marginal_loglikelihood
			result = minimize(f, 
												(alpha, sigma, length_scale), 
												args=(X_train, Y_train, beta))
			alpha, sigma, length_scale = result.x[0], result.x[1], result.x[2]

		kernel_train = GaussionProcess.create_kernel(X_train, X_train, alpha, sigma, length_scale)
		kernel_train_test = GaussionProcess.create_kernel(X_train, X_test, alpha, sigma, length_scale)
		kernel_test = GaussionProcess.create_kernel(X_test, X_test, alpha, sigma, length_scale)
		# 兩個隨機的高斯分佈：跟 y(x) 相關的高斯分佈和跟 noise 相關的高斯分佈是獨立的，可以直接相加
		cov_train = kernel_train + (1/beta)*np.eye(X_train.shape[0])
		cov_test = kernel_test + (1/beta)*np.eye(X_test.shape[0])
		mean = kernel_train_test.T@np.linalg.inv(cov_train)@Y_train
		var = cov_test - kernel_train_test.T@np.linalg.inv(cov_train)@kernel_train_test
		return mean, var, (alpha, sigma, length_scale)
	
def load_data():
	with open("./data/input.data",'r') as file:
		X = []
		Y = []
		for line in file:
			x, y = map(float, line.split(' '))
			X.append(x)
			Y.append(y)
	return np.array(X).reshape(-1,1), np.array(Y).reshape(-1,1)

def plot(X_train, X_test, y_test, var, args, y_test_2, var_2, args_2):
	confidence_interval = 1.96 * np.sqrt(np.diag(var))
	confidence_interval_2 = 1.96 * np.sqrt(np.diag(var_2))
	X_test = X_test.ravel()
	y_test = y_test.ravel()
	y_test_2 = y_test_2.ravel()
	figure = plt.figure()
	figure.suptitle('Machine Learning HW5', fontsize=16)
	figure.add_subplot(211)
	plt.title('sigma: {:.5f}, alpha: {:.5f}, length scale: {:.5f}'.format(args[0], args[1], args[2]))
	plt.scatter(X_train, Y_train, color='k', s=10)
	plt.plot(X_test, y_test, color='b')
	plt.plot(X_test, y_test + confidence_interval,color='r')
	plt.plot(X_test, y_test - confidence_interval,color='r')
	plt.fill_between(X_test, y_test + confidence_interval, y_test - confidence_interval, color='r', alpha=0.1)

	figure.add_subplot(212)
	plt.title('sigma: {:.3f}, alpha: {:.3f}, length scale: {:.3f}'.format(args_2[0], args_2[1], args_2[2]))
	plt.scatter(X_train, Y_train, color='k', s=10)
	plt.plot(X_test, y_test_2, color='b')
	plt.plot(X_test, y_test_2 + confidence_interval_2,color='r')
	plt.plot(X_test, y_test_2 - confidence_interval_2,color='r')
	plt.fill_between(X_test, y_test_2 + confidence_interval_2, y_test_2 - confidence_interval_2, color='r', alpha=0.1)

	plt.tight_layout()
	plt.show()

X_train, Y_train = load_data()
X_test = np.linspace(-60,60,1000).reshape(-1,1)
y_test, var, args = GaussionProcess().fit(X_train, Y_train, X_test)
y_test_2, var_2, args_2 = GaussionProcess().fit(X_train, Y_train, X_test, optimizer=True)
plot(X_train, X_test, y_test, var, args, y_test_2, var_2, args_2)

