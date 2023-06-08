import numpy as np
import numba as nb
import gzip, warnings

class EstimationMaximization():

		def __init__(self, train_num, image_size):
				self.n_samples = train_num
				self.n_features = image_size

		def _load_data(self):
				f = gzip.open('./data/train-images-idx3-ubyte.gz','r')
				f.read(16) # 跳過
				buf = f.read(self.n_samples * self.n_features)
				training_image_data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
				training_image_data = training_image_data.reshape(self.n_samples, self.n_features)
				x_train = np.zeros(training_image_data.shape)
				x_train[training_image_data >= 128] = 1

				f = gzip.open('./data/train-labels-idx1-ubyte.gz','r')
				f.read(8) # 跳過
				buf = f.read(self.n_samples)
				y_train = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

				return x_train, y_train
		
		def _draw_imagination(self, theta, cat_cluster_mapping, labeled):
				if labeled == 0:
						for k in range(10):
								print("class {}:".format(k))
								for feature_num in range(784):
										if theta[k][feature_num] >= 0.5:
												print("1", end = " ")
										else:
												print("0", end = " ")
										if (feature_num+1)%28 == 0:
												print()
				elif labeled == 1:
						for k in range(10):
								print("labeled class {}:".format(k))
								cluster_index = cat_cluster_mapping[k]
								for feature_num in range(784):
										if theta[cluster_index][feature_num] >= 0.5:
												print("1", end = " ")
										else:
												print("0", end = " ")
										if (feature_num+1)%28 == 0:
												print()
		
		def evaluate(self):

			@nb.jit(nopython=True)
			def predict(x_train,y_train,pi,theta):
					counting_matrix = np.zeros((10, 10), dtype=np.int64)
					mapping = np.zeros((10), dtype=np.int64)
					cluster_pred = np.zeros((60000,),dtype=np.int64)
					category_pred = np.zeros((60000,),dtype=np.int64)
					for i in range(60000): # 用 train 好的 pi, theta 去算 posterior，最高的就是預測的 cluster
							p = np.zeros((10), dtype=np.float64)
							for j in range(10):
									posterior = pi[j]
									for k in range(784):
											posterior *= (theta[j, k] ** x_train[i, k])
											posterior *= ((1 - theta[j, k]) ** (1 - x_train[i, k]))
									p[j] = posterior
							# 本來的 cluster 是不知道實際類別的
							cluster_pred[i] = np.argmax(p)
							counting_matrix[y_train[i], np.argmax(p)] += 1

					for i in range(10):
							row, col = np.where(counting_matrix == counting_matrix.max())
							category = row[0]
							cluster = col[0]
							mapping[category] = cluster
							counting_matrix[category, :] = -1 # 被認領之後就不能再被認領了
							counting_matrix[:, cluster] = -1
							
					for i in range(60000):
						category_pred[i] = np.where(mapping == cluster_pred[i])[0][0]

					return mapping, category_pred
			
			mapping, prediction = predict(self.x_train, self.y_train, self.pi, self.theta)

			self._draw_imagination(self.theta, mapping, 1)
			confusion_matrix = np.zeros((10, 2, 2), dtype=np.int64)
			for i in range(60000):
					for j in range(10):
							if self.y_train[i] == j:
									if prediction[i] == j:
											confusion_matrix[j][0][0] += 1
									else:
											confusion_matrix[j][0][1] += 1
							else:
									if prediction[i] == j:
											confusion_matrix[j][1][0] += 1
									else:
											confusion_matrix[j][1][1] += 1

			for i in range(10):
					print('---------------------------------------------------------------')
					print('Confusion Matrix {}:'.format(i))
					print('\t    Predict number {} Predict not number {}'.format(i, i))
					print("Is number {}	  {} \t\t {}".format(i, confusion_matrix[i][0][0],confusion_matrix[i][0][1]))
					print("Isn\'t number {}	  {} \t\t {}".format(i, confusion_matrix[i][1][0],confusion_matrix[i][1][1]))
					sensitivity = confusion_matrix[i][0][0] / (confusion_matrix[i][0][0] + confusion_matrix[i][0][1])
					specificity = confusion_matrix[i][1][1] / (confusion_matrix[i][1][0] + confusion_matrix[i][1][1])

					print('Sensitivity (Successfully predict number {}: {}'.format(i, sensitivity))
					print('Specificity (Successfully predict not number {}: {}'.format(i, specificity))
			
			error = 60000 - np.sum(confusion_matrix[:, 0, 0])
			print('Total iteration to converge: {}'.format(self.iter))
			print('Total error rate: {}'.format(error / 60000))
				

		def fit(self):
				@nb.jit(nopython=True) # 每一個 feature 是獨立的 feature，所以要求出這個 sample 是數字 0~9 的 likelihood 必須連乘
				def e_step(x_train, pi, theta, n_samples, n_features):
						w = np.zeros((n_samples, 10), dtype=np.float64)
						for i in range(n_samples):
								for j in range(10):
										p = pi[j]
										for k in range(n_features):
												p *= (theta[j, k] ** x_train[i, k])
												p *= ((1 - theta[j, k]) ** (1 - x_train[i, k]))
										w[i, j] = p
								p_x = sum(w[i, :]) # 用 likelihood 求出 P(x) 邊際機率
								if p_x == 0:
										continue
								w[i, :] /= p_x
						return w
				
				@nb.jit(nopython=True)
				def m_step(x_train, w, pi, theta, n_samples, n_features):
						for i in range(10):
								n = np.sum(w[:, i]) # 每個 feature 是獨立的 distribution，所以要更新 P(z) 的話，就要把所有 sample 在這個 cluster 下的 posterior 相加
								pi[i] = n / n_samples # 容易 underflow，因爲一開始算 posterior 連乘 784 次會很小
								if n == 0:
										n = 1 
								for j in range(n_features): # 更新每個 cluster 在每個 feature 擲出 1 or 0 的機率
										theta[i, j] = np.dot(x_train[:, j], w[:, i]) / n # 把這個 feature 下的 sample 值乘上所有 sample 在這個 cluster 下的 Poster 
						return pi, theta

				self.x_train, self.y_train = self._load_data()
				pi = np.full((10), 0.1, dtype=np.float64)
				theta = np.random.rand(10, self.n_features)
				theta_prev = np.zeros((10, self.n_features), dtype=np.float64)
				iter = 0
				max_iter = 50
				cluster_truth_mapping = np.array([i for i in range(10)], dtype=np.int64)
				while iter < max_iter:
						iter += 1
						w = e_step(self.x_train, pi, theta, self.n_samples, self.n_features)
						pi, theta = m_step(self.x_train, w, pi, theta, self.n_samples, self.n_features)
						if 0 in pi: # 如果有 underflow 的情況，就重新賦予值，try 到定義出不會發生 underflow 的情況
								pi = np.full((10), 0.1, dtype=np.float64) 
								theta = np.random.rand(10, self.n_features).astype(np.float64)
						self._draw_imagination(theta, cluster_truth_mapping, 0)
						diff = np.sum(np.sum(np.abs(theta - theta_prev)))
						print(f'No. of Iteration: {iter}, Difference: {diff}\n')
						print('---------------------------------------------------------------')
						if diff < 1e-5 and iter > 13:
								self.theta = theta
								self.pi = pi
								self.iter = iter
								break
						theta_prev = theta

warnings.simplefilter('ignore')
model = EstimationMaximization(60000, 784)
model.fit()
model.evaluate()
