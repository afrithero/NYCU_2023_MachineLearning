import numpy as np
import sys

'''
每次處理一個點，用完就丟，資訊保留在 sum 變數供後續使用
'''
'''
Let n ← 0, Sum ← 0, SumSq ← 0
For each datum x:
    n ← n + 1
    Sum ← Sum + x
    SumSq ← SumSq + x × x
Var = (SumSq − (Sum × Sum) / n) / (n − 1)
'''

def gaussian_data_generator(mean,variance):
    sum = np.sum(np.random.uniform(0,1,12)) - 6
    # y = ax + b
    # Affine property
    b = mean
    a = np.sqrt(variance)
    sum = a * sum + b
    return sum

def sequential_estimator(mean, variance):
    sum_1 = 0
    sum_2 = 0
    data_size = 0
    last_variance = 0
    while True:
        data_point = gaussian_data_generator(mean, variance)
        data_size += 1
        print("Add data point: {}".format(data_point)) 
        sum_1 += data_point
        sum_2 += data_point ** 2
        current_mean = sum_1 / data_size
        if data_size == 1:
            current_variance = 0
        else:
            current_variance = (sum_2 - (sum_1**2)/data_size) / (data_size - 1)
        print("Mean = {} Variance={}".format(current_mean, current_variance))
        if last_variance != current_variance and abs(last_variance-current_variance) < 1e-6:
            break
        else:
            last_variance = current_variance
            
mean = int(sys.argv[1])
variance = int(sys.argv[2])
sequential_estimator(mean,variance)