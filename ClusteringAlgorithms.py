from abc import ABC, abstractmethod
class Clustering(ABC):
	@abstractmethod
	def __call__(self, datapoints):
		return NotImplemented

import matplotlib.pyplot as plt
import numpy as np

class KMeans(Clustering):
	'''
	K Means algorithm has to assign "k" classes as parameter
	'''
	def __init__(self, k = 3, maxiter = 100, stop_threshold = 1., order = 2):
		self.k = k
		self.maxiter = maxiter
		self.stop_threshold = stop_threshold
		self.order = order
		self.centers = None
		self.datapoints = None
		self.k_clusters = None
	def __call__(self, datapoints):
		self.datapoints = datapoints
		self.N, self.D = datapoints.shape
		self.k_clusters = [None] * self.N
		# randomly choose k points as initial centers
		init_centers_idx = np.random.choice(self.N, self.k, replace=False)
		self.centers = [self.datapoints[ck] for ck in init_centers_idx]
		# loop over data points over maximum iteration times
		for i in range(self.maxiter):
			changes = self.update()
			# if changes are minor, stop iterations
			if(changes <= self.stop_threshold):
				return self.k_clusters
		return self.k_clusters
	def update(self):
		'''
		Update both centers and point classes
		'''
		# 1. Classify point by distance to center k
		distance = lambda x, y : np.linalg.norm(x-y, self.order)
		# print(self.centers)
		# input()
		for idx, point in enumerate(self.datapoints):
			distances = [distance(point, center_k) for center_k in self.centers]
			# print(distances)
			self.k_clusters[idx] = np.argmax( distances )
		# 2. Update K centers
		last_centers = self.centers
		counts = np.zeros((self.k))
		self.centers = np.zeros((self.k, self.D))
		for idx, point in enumerate(self.datapoints):
			ith_class = self.k_clusters[idx]
			self.centers[ith_class] += point 
			counts[ith_class] += 1
		for kth in range(self.k):
			if counts[kth] > 0:
				self.centers[kth] /= counts[kth]
			else:
				self.centers[kth] = last_centers[kth]
		return distance( last_centers, self.centers )
		# inter change centers not resolved yet
	def display(self, alpha = 0.52):
		plt.ion()
		fig, ax = plt.subplots()
		if self.k_clusters is not None:
			colors = np.random.rand(self.k, 3)
			colors = [colors[ck] for ck in self.k_clusters]
			ax.scatter( self.datapoints[:,0], self.datapoints[:,1], c = colors, alpha = alpha )
		else:
			print('Not performed yet')


