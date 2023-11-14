import numba as nb
import cv2
import numpy as np
import warnings
from array2gif import write_gif

warnings.simplefilter('ignore')

@nb.jit
def kernel(img, gamma_s, gamma_c):
	n = len(img)
	pixel_coordinates = np.zeros((n,2))
	# 產生座標
	for i in range(n):
		pixel_coordinates[i] = [i//100,i%100]
	spatial_dist = np.sum(pixel_coordinates ** 2, axis=1).reshape(-1, 1) + np.sum(pixel_coordinates ** 2, axis=1) - 2 * pixel_coordinates @ pixel_coordinates.T
	# cv2 讀進來是 uint，所以要轉
	color_dist = np.sum(np.float32(img) ** 2, axis=1).reshape(-1,1) + np.sum(np.float32(img) ** 2, axis=1) - 2 * np.float32(img) @ np.float32(img).T
	return np.exp(-gamma_s * spatial_dist) * np.exp(-gamma_c * color_dist)

def load_image(path):
	img = cv2.imread(path)
	height, width, channel = img.shape
	img = img.reshape(-1,channel)
	return img, height, width

def process_gif(color_map,gif_path):
	for i in range(len(color_map)):
		color_map[i] = color_map[i].transpose(1, 0, 2)
	write_gif(color_map, gif_path, fps=2)	

class KMeans:
	def __init__(self, centroid_method, img, height, width, n_cluster):
		self.centroid_method = centroid_method
		self.img = img
		self.height = height
		self.width = width
		self.n_cluster = n_cluster

	def _init_centroids(self):
		centroids = np.zeros((self.n_cluster, self.img.shape[1]))
		n_samples = self.img.shape[0]
		n_features = self.img.shape[1]

		if self.centroid_method == "kmeans++":
			centroids[0] = self.img[np.random.randint(low=0, high=n_samples, size=1),:]
			for c_id in range(1, self.n_cluster):
				if c_id == 1:
					# nx1 can be broadcasted to nxn
					temp_dist = np.sum((self.img - centroids[0])**2, axis=1)
				else:
					# 所有 sample 取與已知 centroid 中最小距離
					temp_dist = np.minimum(temp_dist, np.sum((self.img - centroids[-1]) ** 2, axis=1))
				probs = temp_dist / np.sum(temp_dist)
				next_idx = np.random.choice(n_samples, p=probs)
				centroids[c_id] = self.img[next_idx]

		elif self.centroid_method == "random":
			# 計算每一個 channel 的平均值和標準差
			X_mean=np.mean(self.img,axis=0)
			X_std=np.std(self.img,axis=0)
			# 每個 channel 隨機採樣 k 個值
			for channel in range(n_features):
				centroids[:,channel]=np.random.normal(X_mean[channel],X_std[channel],size=self.n_cluster)

		return centroids

	def _assign_color(self, labels):
		if self.n_cluster == 2:
			cluster_color = np.array([[0, 255, 0], [0, 0, 255]]) 
		elif self.n_cluster == 3:
			cluster_color = np.array([[255,0,0],[0,255,0],[0,0,255]])
		elif self.n_cluster == 4:
			cluster_color = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0]])

		colored_region = np.zeros((self.height, self.width, 3))
		
		for h in range(self.height):
			for w in range(self.width):
				colored_region[h,w,:] = cluster_color[labels[h*self.width+w]]
		
		return colored_region.astype(np.uint8)

	def fit(self, img=None, max_iter=50):
		if img is not None:
			self.img = img
		centroids = self._init_centroids()
		color_map = []
		n_iteration = 0
		for _ in range(max_iter):
			n_iteration += 1	
			# e step
			# 因為都是都是二維且維度不一樣，所以要先拓展維度
			# img: (10000,10000) -> (10000,1,10000)
			# centroids: (k,10000)
			# apply broadcasting rule
		  # img: (10000,1,10000) -> (10000, k, 10000)
		  # centroids: (k, 10000) -> (1, k, 10000) -> (10000, k, 10000)
			dist = np.sum((self.img[:, np.newaxis] - centroids) ** 2, axis=2)
			labels = np.argmin(dist, axis=1)
			# m step
			# 把每個 cluster 的 data 取出來，然後每個 feature 取平均
			new_centroids = np.array([self.img[labels == k].mean(axis=0) for k in range(self.n_cluster)])
			diff = np.sum((new_centroids - centroids)**2)
			centroids = new_centroids

			color_map.append(self._assign_color(labels))
			print('Difference :{}'.format(diff))
		
			if diff == 0:
				break	

		return labels, color_map, n_iteration


if __name__ == '__main__':
	np.random.seed(1)
	centroid_method = 'kmeans++'
	# centroid_method = 'random'
	img_file = 'image2'
	k = 3
	img, height, width = load_image(f'./img/{img_file}.png')
	gram_matrix = kernel(img, 0.00009, 0.00009)
	k_means = KMeans(centroid_method, gram_matrix, height, width, k)
	labels, color_map, n_iteration = k_means.fit()
	print(f'Total iterations: {n_iteration}')
	save_path = f'./gif/KMeans_{img_file}_{centroid_method}_{k}.gif'
	process_gif(color_map,save_path)
	

