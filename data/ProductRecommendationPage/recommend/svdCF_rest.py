import numpy as np
import tqdm
import random
from urils import get_user_item_matrix












class SVD:
	def __init__(self, df, K=20):
		self.df = df
		self.mat = np.array(df)
		self.K = K
		self.bi = {}
		self.bu = {}
		self.qi = {}
		self.pu = {}
		self.avg = np.mean(self.mat[:, 2])
		for i in range(self.mat.shape[0]):
			uid = self.mat[i, 0]
			iid = self.mat[i, 1]
			self.bi.setdefault(iid, 0)
			self.bu.setdefault(uid, 0)
			self.qi.setdefault(iid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
			self.pu.setdefault(uid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))

	def predict_oneuser(self, uid, iid):  # 预测评分的函数
		# setdefault的作用是当该用户或者物品未出现过时，新建它的bi,bu,qi,pu，并设置初始值为0
		self.bi.setdefault(iid, 0)
		self.bu.setdefault(uid, 0)
		self.qi.setdefault(iid, np.zeros((self.K, 1)))
		self.pu.setdefault(uid, np.zeros((self.K, 1)))
		rating = self.avg + self.bi[iid] + self.bu[uid] + np.sum(self.qi[iid] * self.pu[uid])  # 预测评分公式

		if rating > 3:
			rating = 3
		if rating < 1:
			rating = 1
		return rating

	def predict(self, uid, n):
		datamat = get_user_item_matrix(self.df)
		columns = datamat.columns
		datamat = np.asmatrix(datamat)
		N_arr = np.nonzero(datamat[uid, :] == 0.0)[1]
		if len(N_arr) == 0:
			print('you rated everything')
			return
		item_score = {}
		for i in N_arr:
			# print(columns[i])
			s = self.predict_oneuser(uid, columns[i])
			item_score[columns[i]] = s
			# print(columns[i])
		recs = list(sorted(item_score.items(), key=lambda x: x[1], reverse=True))[:n]
		return recs

	def train(self, steps=2, gamma=0.04, Lambda=0.15):  # 训练函数，step为迭代次数。
		print('train data size', self.mat.shape)
		for step in range(steps):
			print('step', step + 1, 'is running')
			KK = np.random.permutation(self.mat.shape[0])  # 随机梯度下降算法，kk为对矩阵进行随机洗牌
			rmse = 0.0
			for i in range(self.mat.shape[0]):
				j = KK[i]
				uid = self.mat[j, 0]
				iid = self.mat[j, 1]
				rating = self.mat[j, 2]
				eui = rating - self.predict_oneuser(uid, iid)
				rmse += eui ** 2
				self.bu[uid] += gamma * (eui - Lambda * self.bu[uid])
				self.bi[iid] += gamma * (eui - Lambda * self.bi[iid])
				tmp = self.qi[iid]
				self.qi[iid] += gamma * (eui * self.pu[uid] - Lambda * self.qi[iid])
				self.pu[uid] += gamma * (eui * tmp - Lambda * self.pu[uid])
			gamma = 0.93 * gamma
			print('rmse is', np.sqrt(rmse / self.mat.shape[0]))

	def test(self, test_data):  # gamma以0.93的学习率递减

		test_data = np.array(test_data)
		print('test data size', test_data.shape)
		rmse = 0.0
		for i in range(test_data.shape[0]):
			uid = test_data[i, 0]
			iid = test_data[i, 1]
			rating = test_data[i, 2]
			eui = rating - self.predict(uid, iid)
			rmse += eui ** 2
		print('rmse of test data is', np.sqrt(rmse / test_data.shape[0]))






def getData():  # 获取训练集和测试集的函数
	import re
	f = open('C:/Users/xuwei/Desktop/data.txt', 'r')
	lines = f.readlines()
	f.close()
	data = []
	for line in lines:
		list = re.split('\t|\n', line)
		if int(list[2]) != 0:  # 提出评分0的数据，这部分是用户评论了但是没有评分的
			data.append([int(i) for i in list[:3]])
	random.shuffle(data)
	train_data = data[:int(len(data) * 7 / 10)]
	test_data = data[int(len(data) * 7 / 10):]
	print('load data finished')
	print('total data ', len(data))
	return train_data, test_data





# item_score = predict_recommend(np.mat(data), 0, arr, cosSim)
# item_score = [(i-1,j) for i, j in item_score.items()]
# item_score

# data1 = predict(data, cosSim)