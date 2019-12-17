# SCSE RecSys Group
# Minh N.
# This module takes the Yelp review.csv (which lists User - Businesses - Rating) and creates a 
# dictionary utility matrix, which can compactly store the sparse matrix.

import pandas as pd
import numpy as np
import json
import sys
import time
import scipy
from scipy import sparse as sps
from itertools import zip_longest

def sparse_dict(x, y, z):

	# Create util dict
	d = {}
	for i in z.iterrows():
		i = i[1]
		u = y[i[0]]
		b = x[i[1]]
		if u in d.keys():
			d[u][b] = i[2]
		else:
			d[u] = {b : i[2]}

	# WNMF testing
	sd = np.zeros((len(y) // 2 + 1, len(x) // 2), dtype = int)
	for k, v in d.items():
		for kk, vv in v.items():
			sd[k, kk] = vv

	t1 = time.time()
	
	matrix_u = np.random.random((len(y) // 2 + 1, 40))
	matrix_v = np.random.random((40, len(x) // 2))

	print(time.time() - t1)

	# This is the weight version of the original utility dictionary, where all real elements are 1
	sd_weight = sd.copy()
	x, y = sd.nonzero()
	for i, j in zip(x, y):
		sd_weight[i, j] = 1

	# This gives us two tuples for us to iterate over in the update process	
	ux, uy = matrix_u.nonzero()
	vx, vy = matrix_v.nonzero()

	it = 4

	t1 = time.time()
	for i in range(4):
		for ui, uj, vi, vj in zip_longest(ux, uy, vx, vy):
			u = matrix_u[ui, uj]
			vt = np.transpose(matrix_v)
			nom = np.matmul((sd[ui] * sd_weight[ui]), vt[:,uj])
			denom = np.matmul((sd_weight[ui] * (np.matmul(matrix_u[ui], matrix_v))), vt[:,uj]) 
			matrix_u[ui, uj] = u * (nom/denom)
			if vi is not None:
				v = matrix_v[vi, vj]
				ut = np.transpose(matrix_u)
				nom = np.matmul(ut[vi], sd_weight[:,vj] * sd[:,vj])
				denom =  np.matmul(ut[vi], (np.matmul(matrix_u, matrix_v[:,vj]) * sd_weight[:,vj]))
				matrix_v[vi, vj] = v * (nom/denom)
		uv = np.matmul(matrix_u, matrix_v)
	print('update time :', time.time() - t1)
	with open('ignore_test.txt', 'a+') as f:
		for i, j in zip(x, y):
			st = str(sd[i,j] + ' - ' + uv[i, j])
			f.write(st)
		f.close()

		# in order to get an num and denom, I need to at minimum, multiple the equivalent row and column.
		# row can be done by querying sparse_matrix[row index, which is i]
		# column can be done with sparse_matrix[:,column index, which is j]
		# To do
		# test each sps type to see which is fastest for this. DoK has quite access to each element but possibly not fast arithmetic
		#zip longest a pair of tuples 

	# # Writes the utility dict to file
	# with open('yelp_utility_dictionary_uc.json', 'w') as f:
	# 	dump = json.dumps(d)
	# 	f.write(dump)
	# 	f.close()

def main():
	df = pd.read_csv('yelp_review_uc.csv')
	with open('yelp_business_uc_id.json', 'r') as f:
		business_id_dict = json.load(f)
	with open('yelp_user_uc_id.json','r') as f:
		user_id_dict = json.load(f)
	sparse_dict(business_id_dict, user_id_dict, df)	

if __name__ == '__main__':
	main()