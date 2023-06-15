import numpy as np
import scipy
import os 
import matplotlib.pyplot as plt
import glob
import re
from PIL import Image


def Hbeta(D=np.array([]), beta=1.0):
    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    # 這個應該是 perplexity
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    # 這個應該是高維度下，其他點在當前點周圍的可能性
    P = P / sumP
    return H, P

# 高維度下距離轉機率的實作
def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    # 計算 pairwise 距離
    D = scipy.spatial.distance.cdist(X, X, 'sqeuclidean')
    P = np.zeros((n, n))
    # beta 應該是 Gaussian Kernel 的超參數（？，初始值設為 1
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    # 每個點都是一個高斯分佈
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        # 這是把自己和除了自己之外的點的距離取出來
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        # perplexity 和 probability
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        # 大於 0 表示超過容忍度，小於 0 表示還可以容忍度還可以再更高
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        # thisP 是一個陣列
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def sne(X, y, no_dims, initial_dims, perplexity, method, interval):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    # 先從 784 維降到 50 維
    X = pca(X, initial_dims).real
    # (2500,50)
    (n, d) = X.shape
    max_iter = 500
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    # 降維後的特徵
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    # 因為上一步是按 row 在算，可是應該要是對稱矩陣，所以要轉置後再加回去
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.	 # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        # 開始算在低維度下靠近的機率
        # 應該是會很不準，因為 Y 初始值是 random 給的
        if method == 'tsne':
            num = 1 / (1 + scipy.spatial.distance.cdist(Y, Y, 'sqeuclidean'))
        else:
            num = np.exp(-1 * scipy.spatial.distance.cdist(Y, Y, 'sqeuclidean'))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        # dY 是梯度
        PQ = P - Q
        for i in range(n):
            if method == 'tsne':
                # PQ[:, i] * num[:, i] -> (2500,)
                # np.tile(PQ[:, i] * num[:, i], (no_dims, 1)) -> (2,2500)
                # (2500,2) x (2500,2) 是對應元素相乘 
                # (Y[i, :] - Y), Y[i,:]維度是 (2,), Y是(2500,2), 相減後可知道當前點在低維度下和其他點的相對距離
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            else:
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), axis=0)


        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        # iY 決定 y 最後的更新量
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        if iter % interval == 0:
            plot_scatter(Y, labels, iter, interval, method, perplexity)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y, P, Q


def plot_similarity(P, Q, method, perplexity):
    plt.clf()
    figure = plt.figure()
    figure.add_subplot(211)
    plt.title(f'{method} high-dimension')
    plt.hist(P.flatten(),bins=50,log=True)
    figure.add_subplot(212)
    plt.title(f'{method} low-dimension')
    plt.hist(Q.flatten(),bins=50,log=True)
    plt.tight_layout()
    plt.savefig(f'./{method}_{perplexity}/similarity.png')


def plot_scatter(Y, labels, idx, interval, method, perplexity):
    plt.clf()
    scatter = plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.legend(*scatter.legend_elements(), loc='lower left', title='Digit')
    plt.title(f'{method}, perplexity: {perplexity}, iteration: {idx}')
    plt.tight_layout()
    if interval:
        plt.savefig(f'./{method}_{perplexity}/iter_{idx // interval}.png')
    else:
        plt.savefig(f'./{method}_{perplexity}/{idx}.png')


def generate_gif(method, perplexity):
	image_list = []
	file_list = os.listdir(f'{method}_{perplexity}')
	file_list.sort(key=lambda file_name: int(re.findall('\d+', file_name)[0]))
	for filename in file_list: # change ending to jpg or jpeg is using those image formats
			im=Image.open(f'{method}_{perplexity}/{filename}')
			image_list.append(im)
	image_list[0].save(f'{method}_{perplexity}/{method}_{perplexity}.gif', save_all=True, append_images=image_list[1:], optimize=False, duration=250, loop=0)


if __name__ == '__main__':
    method = 'sne'
    perplexity = 10.0
    X = np.loadtxt("./tsne_python/mnist2500_X.txt")
    labels = np.loadtxt("./tsne_python/mnist2500_labels.txt")
    if not os.path.exists(f'{method}_{perplexity}'):
            os.mkdir(f'{method}_{perplexity}')
    Y, P, Q = sne(X, labels, no_dims=2, initial_dims=50, perplexity=perplexity, method=method, interval=10)
    generate_gif(method, perplexity)
    plot_scatter(Y, labels, 'final', None, method, perplexity)
    plot_similarity(P, Q, method, perplexity)


    method = 'sne'
    perplexity = 20.0
    if not os.path.exists(f'{method}_{perplexity}'):
            os.mkdir(f'{method}_{perplexity}')
    Y, P, Q = sne(X, labels, no_dims=2, initial_dims=50, perplexity=perplexity, method=method, interval=10)
    generate_gif(method, perplexity)
    plot_scatter(Y, labels, 'final', None, method, perplexity)
    plot_similarity(P, Q, method, perplexity)


    method = 'sne'
    perplexity = 30.0
    if not os.path.exists(f'{method}_{perplexity}'):
            os.mkdir(f'{method}_{perplexity}')
    Y, P, Q = sne(X, labels, no_dims=2, initial_dims=50, perplexity=perplexity, method=method, interval=10)
    generate_gif(method, perplexity)
    plot_scatter(Y, labels, 'final', None, method, perplexity)
    plot_similarity(P, Q, method, perplexity)


    method = 'sne'
    perplexity = 40.0
    if not os.path.exists(f'{method}_{perplexity}'):
            os.mkdir(f'{method}_{perplexity}')
    Y, P, Q = sne(X, labels, no_dims=2, initial_dims=50, perplexity=perplexity, method=method, interval=10)
    generate_gif(method, perplexity)
    plot_scatter(Y, labels, 'final', None, method, perplexity)
    plot_similarity(P, Q, method, perplexity)


    method = 'sne'
    perplexity = 50.0
    if not os.path.exists(f'{method}_{perplexity}'):
            os.mkdir(f'{method}_{perplexity}')
    Y, P, Q = sne(X, labels, no_dims=2, initial_dims=50, perplexity=perplexity, method=method, interval=10)
    generate_gif(method, perplexity)
    plot_scatter(Y, labels, 'final', None, method, perplexity)
    plot_similarity(P, Q, method, perplexity)


    method = 'tsne'
    perplexity = 10.0
    if not os.path.exists(f'{method}_{perplexity}'):
            os.mkdir(f'{method}_{perplexity}')
    Y, P, Q = sne(X, labels, no_dims=2, initial_dims=50, perplexity=perplexity, method=method, interval=10)
    generate_gif(method, perplexity)
    plot_scatter(Y, labels, 'final', None, method, perplexity)
    plot_similarity(P, Q, method, perplexity)


    method = 'tsne'
    perplexity = 20.0
    if not os.path.exists(f'{method}_{perplexity}'):
            os.mkdir(f'{method}_{perplexity}')
    Y, P, Q = sne(X, labels, no_dims=2, initial_dims=50, perplexity=perplexity, method=method, interval=10)
    generate_gif(method, perplexity)
    plot_scatter(Y, labels, 'final', None, method, perplexity)
    plot_similarity(P, Q, method, perplexity)



    method = 'tsne'
    perplexity = 30.0
    if not os.path.exists(f'{method}_{perplexity}'):
            os.mkdir(f'{method}_{perplexity}')
    Y, P, Q = sne(X, labels, no_dims=2, initial_dims=50, perplexity=perplexity, method=method, interval=10)
    generate_gif(method, perplexity)
    plot_scatter(Y, labels, 'final', None, method, perplexity)
    plot_similarity(P, Q, method, perplexity)


    method = 'tsne'
    perplexity = 40.0
    if not os.path.exists(f'{method}_{perplexity}'):
            os.mkdir(f'{method}_{perplexity}')
    Y, P, Q = sne(X, labels, no_dims=2, initial_dims=50, perplexity=perplexity, method=method, interval=10)
    generate_gif(method, perplexity)
    plot_scatter(Y, labels, 'final', None, method, perplexity)
    plot_similarity(P, Q, method, perplexity)


    method = 'tsne'
    perplexity = 50.0
    if not os.path.exists(f'{method}_{perplexity}'):
            os.mkdir(f'{method}_{perplexity}')
    Y, P, Q = sne(X, labels, no_dims=2, initial_dims=50, perplexity=perplexity, method=method, interval=10)
    generate_gif(method, perplexity)
    plot_scatter(Y, labels, 'final', None, method, perplexity)
    plot_similarity(P, Q, method, perplexity)


