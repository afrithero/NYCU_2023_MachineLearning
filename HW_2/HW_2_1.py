import gzip, sys
import numpy as np

# Naive Bayesian 假設
# - feature 之間是獨立的
# - continous mode -> 在同一 class 下的某個特徵是成 normal distribution
class NaiveBayesianClassifier():

	def __init__(self, data_path, mode, image_size, train_num, test_num):
		self.data_path = data_path
		self.mode = mode
		self.image_size = image_size
		self.train_num = train_num
		self.test_num = test_num

	def __load_data(self):
		f = gzip.open('{}/train-images-idx3-ubyte.gz'.format(self.data_path),'r')
		f.read(16) # 跳過
		buf = f.read(self.image_size * self.train_num)
		training_image_data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
		self._training_image_data = training_image_data.reshape(self.train_num, self.image_size) # 拆成二維數組，一張圖用一個一維陣列存

		f = gzip.open('{}/t10k-images-idx3-ubyte.gz'.format(self.data_path),'r')
		f.read(16) # 跳過
		buf = f.read(self.image_size * self.test_num)
		testing_image_data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
		self._testing_image_data = testing_image_data.reshape(self.test_num, self.image_size)

		f = gzip.open('{}/train-labels-idx1-ubyte.gz'.format(self.data_path),'r')
		f.read(8) # 跳過
		buf = f.read(self.train_num)
		self._training_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

		f = gzip.open('{}/t10k-labels-idx1-ubyte.gz'.format(self.data_path),'r')
		f.read(8) # 跳過
		buf = f.read(self.test_num)
		self._testing_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

		self._category_num = len(np.unique(self._training_labels))

	def fit(self):
		self.__load_data()
		# calculate prior
		self._prior = np.zeros(10, dtype=np.float64)
		for category_index in range(self._category_num):
			self._prior[category_index] = len(np.where(self._training_labels == category_index)[0]) / self.train_num

		if self.mode == 0:
			training_category_features_bins = np.zeros((self._category_num,self.image_size,32),dtype=np.float64)
			training_category_count = np.zeros((self._category_num),dtype=np.float64)
			for category_index in range(self._category_num):
				# calculate how many images of each category among 60000 images
				training_category_count[category_index] = len(self._training_image_data[self._training_labels == category_index])
				training_image_data_current_c = self._training_image_data[self._training_labels == category_index] // 8
				for feature_num in range(self.image_size):
					# 統計 current category 在這個 feature 下各個 bin 的 num
					feature_list = training_image_data_current_c[:,[feature_num]]
					unique_bin, bin_counts = np.unique(feature_list, return_counts=True)
					bin_counts = dict(zip(unique_bin, bin_counts))
					min_count = min(bin_counts.values())
					# fill the empty bin with the smallest positive integer among 32 bins
					for bin_num in range(32):
						if bin_num not in bin_counts.keys():
							training_category_features_bins[category_index][feature_num][bin_num] = min_count
						else:
							training_category_features_bins[category_index][feature_num][bin_num] = bin_counts[bin_num]

				self._training_category_features_bins = training_category_features_bins
				self._training_category_count = training_category_count
		
		elif self.mode == 1:
			self._mean = np.zeros((self._category_num, self.image_size), dtype=np.float64)
			self._var = np.zeros((self._category_num, self.image_size), dtype=np.float64)
			for category_index in range(self._category_num):
				training_image_data_category = self._training_image_data[self._training_labels == category_index]
				# The arithmetic mean is the sum of the elements along the axis divided by the number of elements.
				self._mean[category_index, :] = training_image_data_category.mean(axis=0)
				self._var[category_index, :] = training_image_data_category.var(axis=0)
				mean = np.mean(self._var[category_index, :][self._var[category_index, :] > 0])
				self._var[category_index, :][self._var[category_index, :] == 0] = mean

	def predict(self):
		error_count = 0
		for image_num in range(self.test_num):
			posterior = np.zeros((self._category_num), dtype=np.float64)
			if self.mode == 0:
				for category_index in range(self._category_num):
					posterior[category_index] += self._prior[category_index]
					# check the likelihood of each feature from the testing image in model and sum them up wrapped by log function(i.e. cause the features are independent based on naive bayesian)  
					# 計算 testing image 所有 feature 在 current category 下的 likelihood
					for feature_number in range(self.image_size):
						bin_num = self._testing_image_data[image_num][feature_number] // 8
						posterior[category_index] += np.log(self._training_category_features_bins[category_index][feature_number][bin_num] / 
																								self._training_category_count[category_index])
						
			elif self.mode == 1:
				for category_index in range(self._category_num):
						mean = self._mean[category_index]
						var = self._var[category_index]
						test_features = self._testing_image_data[image_num]
						likelihood = np.sum(-0.5 * (np.log(2 * np.pi * var) + ((test_features - mean) ** 2) / var))
						posterior[category_index] = np.log(self._prior[category_index]) + likelihood

			posterior /= sum(posterior)
			prediction = np.argmin(posterior)
			print("Posterior (in log scale):")
			for i in range(len(posterior)):
				print("{}: {}".format(i, posterior[i]))
			print("Prediction: {}, Ans: {}".format(prediction, self._testing_labels[image_num]))
			if prediction != self._testing_labels[image_num]:
				error_count += 1 

		if self.mode == 0:
			for category_index in range(self._category_num):
				print("{}:".format(category_index))
				for feature_num in range(self.image_size):
					if np.argmax(self._training_category_features_bins[category_index][feature_num]) < (128 / 8):
						print("0", end = " ")
					elif np.argmax(self._training_category_features_bins[category_index][feature_num]) >= (128 / 8):
						print("1", end = " ")
					if (feature_num+1)%28 == 0:
						print()

		elif self.mode == 1:
			for category_index in range(self._category_num):
				print("{}:".format(category_index))
				for feature_num in range(self.image_size):
					if self._mean[category_index][feature_num] < 128:
						print("0", end = " ")
					elif self._mean[category_index][feature_num] >= 128:
						print("1", end = " ")
					if (feature_num+1)%28 == 0:
						print()
		return (error_count / 10000)


mode = int(sys.argv[1])
data_path = str(sys.argv[2])
model = NaiveBayesianClassifier(data_path, mode, 784, 60000, 10000)
model.fit()
error_rate = model.predict()
print(error_rate, end="")