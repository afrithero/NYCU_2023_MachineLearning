import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import scipy
from os.path import join
from PIL import Image


shape = (100,100)

def read_image(img_path):
    img = Image.open(img_path)
    img = img.resize(shape, Image.Resampling.LANCZOS)
    img = np.array(img)
    label_temp = img_path.split('/')
    label = label_temp[-1][7:9]
    return img.ravel().astype(np.float64), int(label)    


def read_data(dataset_path='./Yale_Face_Database/Training/'):
    all_img = []
    all_filename = []
    all_label = []
    for filename in os.listdir(dataset_path):
        img, label = read_image(join(dataset_path, filename))
        all_img.append(img)
        all_label.append(label)
    return np.array(all_img), np.array(all_label)


def PCA(X, num_components=0):
	n, d = X.shape
	mu = X.mean(axis=0)
	X = X - mu
	if n > d:
		C = X.T @ X
		eigenvalues, eigenvectors = np.linalg.eigh(C)
	else:
		C = X @ X.T
		eigenvalues, eigenvectors = np.linalg.eigh(C)
		eigenvectors = X.T @ eigenvectors

	eigenvectors_norm = np.linalg.norm(eigenvectors,axis=0)
	eigenvectors = eigenvectors / eigenvectors_norm
	eigenvectors = eigenvectors[:, np.argsort(-eigenvalues)]
	eigenvectors = eigenvectors[:, 0:num_components].real
	return eigenvectors, mu


def LDA(X, y, num_components=25):
	class_num = len(np.unique(y))
	feature_num = X.shape[1]
	feature_mean = np.mean(X, axis=0)
	C_mean = np.zeros((class_num, feature_num))
	S_w = np.zeros((feature_num, feature_num))
	S_b = np.zeros((feature_num, feature_num))

	for i in range(class_num):
		X_i = X[y==i+1]
		C_mean[i,:] = np.mean(X_i,axis=0)
		S_b += X_i.shape[0] * (C_mean[i,:] - feature_mean).T @ (C_mean[i,:] - feature_mean)
		S_w += (X_i - C_mean[i,:]).T @ (X_i - C_mean[i,:])

	eigenvalues, eigenvectors = cp.linalg.eigh(cp.array(np.linalg.inv(S_w)@S_b))
	eigenvectors = cp.ndarray.get(eigenvectors[np.argsort(-eigenvalues)])
	eigenvectors = eigenvectors[:,0:num_components].real

	return eigenvectors


def fisherfaces(X, y, num_components=0):
		# (135,2500)
		n, d = X.shape
		c = len(np.unique(y))
		# (2500,120)
		eigenvectors_pca, feature_mean = PCA(X, (n - c))
		# (135,120)
		X_pca = (X-feature_mean)@eigenvectors_pca
		# (120,25)
		eigenvectors_lda = LDA(X_pca, y, num_components)
		# (2500,25)
		eigenvectors = eigenvectors_pca @ eigenvectors_lda
		return eigenvectors, feature_mean


def show_face(X, components=25, type=0):
	l = int(np.sqrt(components))
	for i in range(components):
			plt.subplot(l,l,i+1)
			plt.imshow(X[:,i].reshape(shape),cmap='gray')
			plt.xticks(fontsize=5)
			plt.yticks(fontsize=5)
			
	plt.subplots_adjust(left=0.125,
											bottom=0.1, 
											right=1.5, 
											top=1.5, 
											wspace=0.2, 
											hspace=0.35)

	if type:
		plt.savefig('./figure/fisher_face.png',bbox_inches='tight')
	else: 
		plt.savefig('./figure/eigen_face.png',bbox_inches='tight')


def show_reconstruction(X, feature_mean, eigenvectors, type=0):
	np.random.seed(4)
	face_idx = np.random.choice(X.shape[0], 10)
	X_selected = X[face_idx,:]
	X_reduction = (X_selected - feature_mean) @ eigenvectors
	X_reconstruction = X_reduction @ eigenvectors.T + feature_mean


	for i in range(10):
			plt.subplot(2,10,i+1)
			plt.imshow(X_reconstruction[i,:].reshape(shape),cmap='gray')
			plt.xticks(fontsize=5)
			plt.yticks(fontsize=5)
			plt.subplot(2,10,10+(i+1))
			plt.imshow(X_selected[i,:].reshape(shape),cmap='gray')
			plt.xticks(fontsize=5)
			plt.yticks(fontsize=5)
			
	plt.subplots_adjust(left=0.125,
											bottom=1.0, 
											right=1.5, 
											top=1.5, 
											wspace=0.2, 
											hspace=0.2)
	
	if type:
		plt.savefig('./figure/fisher_face_reconstruction.png',bbox_inches='tight')	
	else:
		plt.savefig('./figure/eigen_face_reconstruction.png',bbox_inches='tight')


def face_recognition(X_train, y_train, X_test, y_test, method=None, kernel_type=None):
	if method == 'PCA':
		eigenvectors, feature_mean = PCA(X_train, 25)
		X_train_proj = (X_train - feature_mean) @ eigenvectors
		X_test_proj = (X_test - feature_mean) @ eigenvectors
	
	elif method == 'LDA':
		eigenvectors_pca, feature_mean = PCA(X_train, 135)
		X_train_pca = (X_train-feature_mean)@eigenvectors_pca
		eigenvectors_lda = LDA(X_train_pca, y_train, 25)
		eigenvectors = eigenvectors_pca @ eigenvectors_lda
		X_train_proj = X_train @ eigenvectors
		X_test_proj = X_test @ eigenvectors

	elif method == 'Kernel PCA':
		eigenvectors, feature_mean = kernel_pca(X_train, num_components=25, kernel_type=kernel_type)
		X_train_proj = (X_train - feature_mean) @ eigenvectors
		X_test_proj = (X_test - feature_mean) @ eigenvectors

	elif method == 'Kernel LDA':
		n, d = X_train.shape
		c = len(np.unique(y_train))
		# (2500, 135)
		eigenvectors_pca, feature_mean = PCA(X_train, 135)
		#(135, 135)
		X_train_pca = (X_train-feature_mean)@eigenvectors_pca
		#(135, 25)
		eigenvectors_lda = kernel_lda(X_train_pca, y_train, num_components=25, kernel_type=kernel_type)
		#(2500, 25)
		eigenvectors = eigenvectors_pca @ eigenvectors_lda
		X_train_proj = X_train @ eigenvectors
		X_test_proj = X_test @ eigenvectors


	dist = np.zeros((X_test.shape[0], X_train.shape[0]))	
	for i in range(X_test.shape[0]):
		dist[i,:] = np.sqrt(np.sum((X_test_proj[i] - X_train_proj)**2, axis=1))
	dist = dist.argsort()

	K = [1, 3, 5, 7, 9, 11]
	total = X_test.shape[0]

	for k in K:
		correct = 0
		for i in range(X_test.shape[0]):
			neighbor = y_train[dist[i,0:k]]
			neighbor, count = np.unique(neighbor, return_counts=True)
			predict = neighbor[np.argmax(count)]
			if predict == y_test[i]:
				correct += 1
		
		print(f'K={k:>2}, accuracy: {correct / total:>.3f} ({correct}/{total})')


def kernel_pca(X, num_components, kernel_type):
	mu = X.mean(axis=0)
	X = X - mu

	if kernel_type == 'linear':
		kernel = X @ X.T
	# k(x, y) = (alpha * <x, y> + c)^d
	elif kernel_type == 'polynomial':
		gamma = 5
		coef = 10
		degree = 2
		kernel = np.power(gamma * (X @ X.T) + coef, degree)
	# k(x, y) = exp(-gamma(|x,y|^2)
	elif kernel_type == 'RBF':
		gamma = 1e-7
		kernel = np.exp(-gamma * scipy.spatial.distance.cdist(X, X, 'sqeuclidean'))

	n = kernel.shape[0]
	one = np.ones((n, n), dtype=np.float64) / n
	kernel = kernel - one @ kernel - kernel @ one + one @ kernel @ one
	eigenvalues, eigenvectors = np.linalg.eigh(kernel)

	eigenvectors = (X_train - mu).T @ eigenvectors
	eigenvectors_norm = np.linalg.norm(eigenvectors,axis=0)
	eigenvectors = eigenvectors / eigenvectors_norm
	eigenvectors = eigenvectors[:,np.argsort(-eigenvalues)]		
	eigenvectors = eigenvectors[:, 0:num_components].real
	return eigenvectors, mu


def kernel_lda(X, y, num_components, kernel_type):
	mu = X.mean(axis=0)
	X = X - mu

	if kernel_type == 'linear':
		kernel = X @ X.T
	# k(x, y) = (alpha * <x, y> + c)^d
	elif kernel_type == 'polynomial':
		gamma = 5
		coef = 10
		degree = 2
		kernel = np.power(gamma * (X @ X.T) + coef, degree)
	# k(x, y) = exp(-gamma(|x,y|^2)
	elif kernel_type == 'RBF':
		gamma = 1e-7
		kernel = np.exp(-gamma * scipy.spatial.distance.cdist(X, X, 'sqeuclidean'))
	
	n = kernel.shape[0]
	one = np.ones((n, n), dtype=np.float64) / n
	K = kernel - one @ kernel - kernel @ one + one @ kernel @ one

	class_num = len(np.unique(y))
	K_mean = K.mean(axis=0)
	C_mean = np.zeros((class_num, n))
	S_w = np.zeros((n, n))
	S_b = np.zeros((n, n))

	for i in range(class_num):
		K_i = K[y==i+1]
		C_mean[i,:] = np.mean(K_i,axis=0)
		S_b += K_i.shape[0] * (C_mean[i,:] - K_mean).T @ (C_mean[i,:] - K_mean)
		S_w += (K_i - C_mean[i,:]).T @ (K_i - C_mean[i,:])

	eigenvalues, eigenvectors = cp.linalg.eigh(cp.array(np.linalg.pinv(S_w)@S_b))
	eigenvectors = cp.ndarray.get(eigenvectors[np.argsort(-eigenvalues)])
	eigenvectors = eigenvectors[:,0:num_components].real

	return eigenvectors

if __name__ == '__main__':

# Part 1
	X_train, y_train = read_data()
	X_test, y_test = read_data('./Yale_Face_Database/Testing')

# eigen face 
	eigenvectors_pca, feature_mean = PCA(X_train, 25)
	show_face(eigenvectors, 25, 0)
	show_reconstruction(X_train, feature_mean, eigenvectors_pca, 0)

# Fisher face
	eigenvectors_lda, feature_mean = fisherfaces(X_train, y_train, 25)
	show_face(eigenvectors_lda, 25, 1)
	show_reconstruction(X_train, feature_mean, eigenvectors_lda, 1)

# Part 2
	face_recognition(X_train, y_train, X_test, y_test, method='PCA')
	face_recognition(X_train, y_train, X_test, y_test, method='LDA')

# Part 3
	face_recognition(X_train, y_train, X_test, y_test, method='Kernel PCA', kernel_type='linear')
	face_recognition(X_train, y_train, X_test, y_test, method='Kernel PCA', kernel_type='polynomial')
	face_recognition(X_train, y_train, X_test, y_test, method='Kernel PCA', kernel_type='RBF')
	face_recognition(X_train, y_train, X_test, y_test, method='Kernel LDA', kernel_type='linear')
	face_recognition(X_train, y_train, X_test, y_test, method='Kernel LDA', kernel_type='polynomial')
	face_recognition(X_train, y_train, X_test, y_test, method='Kernel LDA', kernel_type='RBF')


