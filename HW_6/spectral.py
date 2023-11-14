from kmeans import *
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt 

class SpectralClustering(KMeans):
	def __init__(self, centroid_method, img, height, width, n_cluster, normalization=False):
		img = np.float32(img)
		super(SpectralClustering,self).__init__(centroid_method, img, height, width, n_cluster)
		self.normalization = normalization

	def decompose(self):
		D = cp.diag(cp.sum(cp.array(self.img), axis=1))
		L = D - cp.array(self.img)
		if self.normalization == True:
			D = cp.linalg.inv(cp.sqrt(D))
			L = D@L@D
		eigenvalue, eigenvector = cp.linalg.eigh(L)
		return cp.ndarray.get(eigenvalue), cp.ndarray.get(eigenvector)

def visuailize_eigen_space(k, normalization, eigen_vectors, labels):
	fig = plt.figure()
	if k == 2:
		ax = fig.add_subplot(111)
		markers = ['o','^']
		for marker,i in zip(markers,np.arange(2)):
			ax.scatter(U[:,0][labels==i],U[:,1][labels==i],marker=marker)
		plt.title('2D representation of Eigenspace coordinates')
		ax.set_xlabel('1st Eigenvector')
		ax.set_ylabel('2nd Eigenvector')
	
	elif k == 3:
		ax = fig.add_subplot(111,projection='3d')
		markers = ['o','^','s']
		for marker, i in zip(markers, np.arange(3)):
			ax.scatter(U[:,0][labels==i],U[:,1][labels==i],U[:,2][labels==i],marker=marker)
		plt.title('3D representation of Eigenspace coordinates')
		ax.set_xlabel('1st Eigenvector')
		ax.set_ylabel('2nd Eigenvector')
		ax.set_zlabel('3rd Eigenvector')

	if normalization == True:
		plt.savefig(f'./img/{k}D_eigen_space_norm.png')
	else:
		plt.savefig(f'./img/{k}D_eigen_space_ratio.png')

if __name__ == '__main__':
	np.random.seed(1)
	k = 3
	normalization = False
	img_file = 'image2'
	# centroid_method = 'random'
	centroid_method = 'kmeans++'
	img, height, width = load_image(f'./img/{img_file}.png')
	gram_matrix = kernel(img, 0.00009, 0.00009)
	s = SpectralClustering(centroid_method, gram_matrix, height, width, k, normalization)
	# already sorted
	eigen_value, eigen_vector = s.decompose()
	U = eigen_vector[:,0:k]
	labels, color_map, n_iteration = s.fit(img=U)
	unique_labels, counts = np.unique(labels, return_counts=True)
	print(f'unique_labels:{unique_labels}, counts:{counts}')
	print(f'Total iterations: {n_iteration}')
	if normalization == True:
		save_path = f'./gif/Spectral_{img_file}_norm_{centroid_method}_{k}.gif'
	else:
		save_path = f'./gif/Spectral_{img_file}_ratio_{centroid_method}_{k}.gif'

	process_gif(color_map,save_path)

	if k == 2 or k == 3:
		visuailize_eigen_space(k, normalization, U, labels)