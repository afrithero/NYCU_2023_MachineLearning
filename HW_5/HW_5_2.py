import numpy as np
from libsvm.svmutil import *

def read_data(file_path):
		with open(file_path,'r') as file:
				data = np.genfromtxt(file_path, delimiter=',')
		return data

class SVM():
		def __init__(self):
			pass
		
		def fit(self, X_train, y_train, kernel_type=0, optimal_args=None, user_defined_kernel=False):
				if user_defined_kernel:
						linear_kernel = SVM.linear_kernel(X_train, X_train)
						RBF_kernel = SVM.RBF_kernel(X_train, X_train, 0.1)
						args = "-t {}".format(4)
						X_kernel = np.hstack((np.arange(1, 5001).reshape((-1, 1)), linear_kernel + RBF_kernel))
						return svm_train(y_train, X_kernel , args)

				else:
					if kernel_type == 0:
						args = "-t {} -c {} -q".format(kernel_type, optimal_args["cost"])
						return svm_train(y_train, X_train, args)
					elif kernel_type == 1:
						args = "-t {} -c {} -g {} -d {} -r {} -q".format(kernel_type, optimal_args["cost"], optimal_args["gamma"], optimal_args["degree"], optimal_args["coef"])
						return svm_train(y_train, X_train, args)
					elif kernel_type == 2:
						args = "-t {} -c {} -g {} -q".format(kernel_type, optimal_args["cost"], optimal_args["gamma"])
						return svm_train(y_train, X_train, args)

		def predict(self, X_test, y_test, m):
			return svm_predict(y_test, X_test, m)

		@staticmethod
		def linear_kernel(X_m, X_n):
				return X_m @ X_n.T
		
		@staticmethod
		def RBF_kernel(X_m, X_n, gamma):
				dist = np.sum(X_m ** 2, axis=1).reshape(-1, 1) + np.sum(X_n ** 2, axis=1) - 2 * X_m @ X_n.T
				return np.exp((-1 * gamma * dist))

		@staticmethod
		def grid_search(X_train, y_train, kernel_type=0, k_fold=2):
			optimal_args = {"cost":0, "gamma":0, "degree":0, "coef":0}
			# as the same as scikit-learn
			cost_range = [0.001, 0.01, 0.1, 1, 10]
			gamma_range = [0.001, 0.01, 0.1, 1]
			degree_range = [2,3,4]
			coef_zero = [0,1,2]
			optimal_acc = 0 
			# linear kernel
			if kernel_type == 0:	
				for cost in cost_range:
					args = "-t {} -c {} -v {} -q".format(kernel_type, cost, k_fold)
					cv_acc = svm_train(y_train, X_train, args)
					if cv_acc > optimal_acc:
						optimal_acc = cv_acc
						optimal_args["cost"] = cost
			# poly kernel
			elif kernel_type == 1:
				for cost in cost_range:
					for gamma in gamma_range:
						for degree in degree_range:
							for coef in coef_zero:
								args = "-t {} -c {} -g {} -d {} -r {} -v {} -q".format(kernel_type, cost, gamma, degree, coef, k_fold)
								cv_acc = svm_train(y_train, X_train, args)
								if cv_acc > optimal_acc:
									optimal_acc = cv_acc
									optimal_args["cost"] = cost
									optimal_args["gamma"] = gamma
									optimal_args["degree"] = degree
									optimal_args["coef"] =coef
			# RBF kernel
			elif kernel_type == 2:
				for cost in cost_range:
					for gamma in gamma_range:
						args = "-t {} -c {} -g {} -v {} -q".format(kernel_type, cost, gamma, k_fold)
						cv_acc = svm_train(y_train, X_train, args)
						if cv_acc > optimal_acc:
							optimal_acc = cv_acc
							optimal_args["cost"] = cost
							optimal_args["gamma"] = gamma
			return optimal_args
					
X_train = read_data("./data/X_train.csv")
y_train = read_data("./data/Y_train.csv")
X_test = read_data("./data/X_test.csv")
y_test = read_data("./data/Y_test.csv")

print("------linear: ")
optimal_args = SVM.grid_search(X_train, y_train, kernel_type=0, k_fold=5)
print("optimal_args:{}".format(optimal_args))
model = SVM()
m = model.fit(X_train, y_train, kernel_type=0, optimal_args=optimal_args)
model.predict(X_test, y_test, m)

print("------polynomial: ")
optimal_args = SVM.grid_search(X_train, y_train, kernel_type=1, k_fold=5)
print("optimal_args:{}".format(optimal_args))
model = SVM()
m = model.fit(X_train, y_train, kernel_type=1, optimal_args=optimal_args)
model.predict(X_test, y_test, m)

print("------RBF:")
optimal_args = SVM.grid_search(X_train, y_train, kernel_type=2, k_fold=5)
print("optimal_args:{}".format(optimal_args))
model = SVM()
m = model.fit(X_train, y_train, kernel_type=2, optimal_args=optimal_args)
model.predict(X_test, y_test, m)

print("------linear+RBF:")
linear_kernel_s = SVM.linear_kernel(X_train, X_test).T
RBF_kernel_s = SVM.RBF_kernel(X_train, X_test, 0.1).T
X_kernel_s = np.hstack((np.arange(1, 2501).reshape((-1, 1)), linear_kernel_s + RBF_kernel_s))
model = SVM()
m = model.fit(X_train, y_train, user_defined_kernel=True)
model.predict(X_kernel_s, y_test, m)