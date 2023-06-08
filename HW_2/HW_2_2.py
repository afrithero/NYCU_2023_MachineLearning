# 每一行的 input 是 0 和 1, 可算出 p 和 1-p, 以輸入的超參數 a, b 建立出 beta prior distribution
# 根據每一行的樣本數算 Binomial maximum likelihood -> based on MLE -> 找到一個使最後機率變成最大的 m
# posterior distribution 正比於 beta prior x binomial maximum likelihood
# 從前面觀測到的 posterior 作為接下來的 prior（a, b 相當於前面資料觀測到的正反面發生次數）
import sys
import numpy as np

def factorial(n): 
		return 1 if (n==1 or n==0) else n * factorial(n - 1);

def calculate_parameter(event):
		event = np.array([int(i) for i in event],dtype=int)
		p_head, p_tail = (event == 1).mean(), (event ==0).mean()
		n = event.shape[0]
		return n, p_head, p_tail

param_a = int(sys.argv[1])
param_b = int(sys.argv[2])
file_path = str(sys.argv[3])

event_list = []
with open(file_path, 'r') as file:
		for line in file:
				event_list.append(line.rstrip('\n'))

prior_a = param_a
prior_b = param_b

for event_id in range(len(event_list)):
	if event_id != 0:
		prior_a, prior_b = posterior_a, posterior_b
	n, p_head, p_tail = calculate_parameter(event_list[event_id])
	n_fac = factorial(n)
	maximun_likelihood = 0
	m = 0
	for i in range(n,-1,-1):
			m_fac = factorial(i)
			m_n_fac = factorial(n-i)
			combination_count = (n_fac) / (m_fac * m_n_fac)
			likelihood = combination_count * (p_head ** i) * (p_tail ** (n - i))
			if likelihood > maximun_likelihood:
					maximun_likelihood = likelihood
					m = i
	posterior_a, posterior_b = prior_a + m, prior_b + n - m
	print("Case {}: {}".format(event_id+1, event_list[event_id]))
	print("Likelihood: {}".format(maximun_likelihood))
	print("Beta prior: a = {}, b = {}".format(prior_a, prior_b))
	print("Beta posterior: a = {}, b = {}".format(posterior_a, posterior_b))
	print()