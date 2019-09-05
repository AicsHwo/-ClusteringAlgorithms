from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

class DataPointGeneration(ABC):
	@abstractmethod
	def __call__(self, N = 100):
		return NotImplemented
	def display(self):
		pass


class Gaussian(DataPointGeneration):
	def __init__(self, mean = np.array([0.0, 0.0]), cov = np.identity(2)):
		self.mu = mean
		self.cov = cov
		assert(len(self.mu) == self.cov.shape[0] == self.cov.shape[1])
		self.data = None
	def __call__(self, N = 100):
		'''
		return in shape of N x D
		'''
		self.data = np.random.multivariate_normal(self.mu, self.cov, N)
		return self.data
	def display(self, alpha = 0.6):
		if self.data is not None:
			plt.ion()
			fig, ax = plt.subplots()
			colors = np.random.rand(3)
			ax.scatter(self.data[:,0], self.data[:,1], c = colors, alpha = alpha)
			fig.canvas.draw()
			fig.show()
		else:
			print("No data point present")


class MixtureOfGaussian(DataPointGeneration):
	@staticmethod
	def Random_ND_Clusters(N = 3, D = 2, mu_range = 52, cov_range = 32):
		import time
		np.random.seed(int(time.time()%1000))
		means = np.random.rand(N, D) * mu_range - (mu_range/2)
		covs = [np.diag(np.random.rand(D) * cov_range) for c in range(N)]
		return MixtureOfGaussian(means, covs)

	def __init__(self, means = [np.array([0, 0])], covs = [np.identity(2)]):
		self.Mus = means
		self.Covs = covs
		# print(self.Mus, self.Covs)
		assert(len(self.Mus) == len(self.Covs))
		for c in range(len(self.Mus)):
			assert( len(self.Mus[c]) == self.Covs[c].shape[0] \
									 == self.Covs[c].shape[0] )
		self.data = None
	def __call__(self, N = [100]):
		if len(N) == 1:
			N = N * len(self.Mus)
		assert( len(N) == len(self.Mus) )
		zipped_parameters = zip(self.Mus, self.Covs, N)
		self.data = np.array([np.random.multivariate_normal(mu, cov, n) for mu, cov, n in zipped_parameters])
		return self.data
	def display(self, alpha = 0.6):
		'''
		Currently only 2d data is supported
		'''
		if self.data is not None:
			plt.ion()
			fig, ax = plt.subplots()
			for cluster in self.data:
				colors = np.random.rand(3)
				ax.scatter(cluster[:,0], cluster[:,1], c = colors, alpha = alpha)
			fig.canvas.draw()
			fig.show()
		else:
			print('No data point present')


def shuffle(datapoints):
	'''
	assume datapoints is of shape k x N x D
	'''
	assert(len(datapoints.shape) == 3)
	k, N, D = datapoints.shape
	new_datapoints = np.copy(datapoints)
	new_datapoints = new_datapoints.reshape((k*N, D))
	# print(datapoints)
	np.random.shuffle(new_datapoints)
	return new_datapoints