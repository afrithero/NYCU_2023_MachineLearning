import numpy as np
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

class LogisticRegression:

	def __init__(self):
		self.weights_gd = None
		self.weights_hessian = None

	def fit(self, x, y, lr1, lr2, epoch = 1000):
		sample_size = x.shape[0]
		feature_size = x.shape[1]
		self.weights_gd = np.zeros((feature_size, 1),dtype=np.float64)
		self.weights_hessian = np.zeros((feature_size, 1),dtype=np.float64)
		for _ in range(epoch):
				prediction = self.sigmoid(x@self.weights_gd)
				gradient = 1/sample_size*(x.T@(prediction - y)) # 除以 sample size 是為了正規化
				self.weights_gd -= lr1 * gradient

		for _ in range(epoch*10):
			prediction = self.sigmoid(x@self.weights_hessian)
			gradient = 1/sample_size*(x.T@(prediction - y)) 
			hessian = np.diag(np.diag(x.T@prediction@(1-prediction).T@x)) # PRML 4.3.3 # 1 can be broadcasted # 第一層 diag 是先取對角元素，第二層 diag 是把對角元素陣列轉成方陣
			self.weights_hessian -= lr2*np.linalg.inv(hessian)@gradient

	def evaluate(self, x, y):
		x_ground_truth_c1 = x[np.where(y==0)[0],0:2]
		x_ground_truth_c2 = x[np.where(y==1)[0],0:2]

		y_pred_gd = self.sigmoid(x@self.weights_gd)
		y_pred_gd_discrete = np.rint(y_pred_gd).astype(np.int64)
		confusion_matrix_gd = np.zeros((2,2),dtype=np.int64)

		for index, data in np.ndenumerate(y):
				confusion_matrix_gd[data][y_pred_gd_discrete[index[0]][0]] += 1

		x_gd_c1 = x[np.where(y_pred_gd<0.5)[0],0:2]
		x_gd_c2 = x[np.where(y_pred_gd>=0.5)[0],0:2]
				
		print("Gradient Descent:")
		print("w:{},{},{}".format(self.weights_gd[0][0],self.weights_gd[1][0],self.weights_gd[2][0]))
		print("Confusion Matrix:")
		print("\t    Predict cluster 1 Predict cluster 2")
		print("Is cluster 1	  {} \t\t {}".format(confusion_matrix_gd[0][0],confusion_matrix_gd[0][1]))
		print("Is cluster 2	  {} \t\t {}".format(confusion_matrix_gd[1][0],confusion_matrix_gd[1][1]))
		print("Sensitivity (Successfully predict cluster 1: {})".format(confusion_matrix_gd[0][0] / (confusion_matrix_gd[0][0] + confusion_matrix_gd[0][1])))
		print("Specificity (Successfully predict cluster 2: {})".format(confusion_matrix_gd[1][1] / (confusion_matrix_gd[1][0] + confusion_matrix_gd[1][1])))
		
		print("---------------------------------------------------------")

		y_pred_hessian = self.sigmoid(x@self.weights_hessian)
		y_pred_hessian_discrete = np.rint(y_pred_hessian).astype(np.int64)
		confusion_matrix_hessian = np.zeros((2,2),dtype=np.int64)

		for index, data in np.ndenumerate(y):
				confusion_matrix_hessian[data][y_pred_hessian_discrete[index[0]][0]] += 1

		x_hessian_c1 = x[np.where(y_pred_hessian<0.5)[0],0:2]
		x_hessian_c2 = x[np.where(y_pred_hessian>=0.5)[0],0:2]

		print("Newton's Method:")
		print("w:{},{},{}".format(self.weights_hessian[0][0],self.weights_hessian[1][0],self.weights_hessian[2][0]))
		print("Confusion Matrix:")
		print("\t    Predict cluster 1 Predict cluster 2")
		print("Is cluster 1	  {} \t\t {}".format(confusion_matrix_hessian[0][0],confusion_matrix_hessian[0][1]))
		print("Is cluster 2	  {} \t\t {}".format(confusion_matrix_hessian[1][0],confusion_matrix_hessian[1][1]))
		print("Sensitivity (Successfully predict cluster 1: {})".format(confusion_matrix_hessian[0][0] / (confusion_matrix_hessian[0][0] + confusion_matrix_hessian[0][1])))
		print("Specificity (Successfully predict cluster 2: {})".format(confusion_matrix_hessian[1][1] / (confusion_matrix_hessian[1][0] + confusion_matrix_hessian[1][1])))

		fig = plt.figure(figsize=(6,6))
		fig.add_subplot(131)
		plt.scatter(x_ground_truth_c1[:,0],x_ground_truth_c1[:,1],c='red',label="c1")
		plt.scatter(x_ground_truth_c2[:,0],x_ground_truth_c2[:,1],c='blue',label="c2")
		plt.legend()
		plt.title("Ground Truth")
		fig.add_subplot(132)
		plt.scatter(x_gd_c1[:,0],x_gd_c1[:,1],c='red',label="c1")
		plt.scatter(x_gd_c2[:,0],x_gd_c2[:,1],c='blue',label="c2")
		plt.title("Gradient Descent")
		plt.legend()
		fig.add_subplot(133)
		plt.scatter(x_hessian_c1[:,0],x_hessian_c1[:,1],c='red',label="c1")
		plt.scatter(x_hessian_c2[:,0],x_hessian_c2[:,1],c='blue',label="c2")
		plt.title("Newton's Method")
		plt.legend()
		fig.tight_layout()
		plt.show()

	def sigmoid(self,z):
		return 1 / (1 + np.exp(-z))

data_num = 50
mx1, my1, mx2, my2 = 1, 1, 3, 3
vx1, vy1, vx2, vy2 = 2, 2, 4, 4
x = np.zeros((data_num*2,3))
y = np.zeros((data_num*2,1),dtype=np.int64)

for i in range(data_num*2):
	if i < data_num:
		x[i][0] = gaussian_data_generator(mx1,vx1)
		x[i][1] = gaussian_data_generator(my1,vy1)
		x[i][2] = 1
		y[i][0] = 0
	else:
		x[i][0] = gaussian_data_generator(mx2,vx2)
		x[i][1] = gaussian_data_generator(my2,vy2)
		x[i][2] = 1
		y[i][0] = 1

lr1 = 0.01
lr2 = 10
epoch = 3000

logistic_reg = LogisticRegression()
logistic_reg.fit(x, y, lr1=lr1, lr2=lr2, epoch=epoch)
logistic_reg.evaluate(x, y)