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
			self.k_clusters[idx] = np.argmin( distances )
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

	def in_animation(self, datapoints, anime_start_delay = 3, alpha = 0.52):
		import matplotlib.animation as animation
		self.datapoints = datapoints
		self.N, self.D = datapoints.shape
		self.k_clusters = [None] * self.N
		self.anime_start_delay = anime_start_delay
		plt.ion()
		self.anime_fig, self.anime_ax = plt.subplots()
		self.anime_scatter = None
		self.anime_alpha = alpha
		self.anime = animation.FuncAnimation(self.anime_fig, self.animation_update, \
											 frames=np.arange(self.maxiter + anime_start_delay), interval = 532, \
											 init_func = self.animation_init, blit = True, repeat = True)
		plt.show()

	def animation_init(self):
		# randomly choose k points as initial centers
		init_centers_idx = np.random.choice(self.N, self.k, replace=False)
		self.centers = [self.datapoints[ck] for ck in init_centers_idx]
		class_colors = np.random.rand(self.k, 3)
		init_solid_colors = np.hstack( (class_colors, np.ones((self.k,1))) )
		self.cluster_colors = np.hstack( (class_colors, np.ones((self.k,1)) * self.anime_alpha) )
		unassigned = np.array([0.5, 0.5, 0.5, self.anime_alpha])

		colors = np.ones((self.N, 4)) * unassigned
		for color_idx, center_idx in enumerate(init_centers_idx):
			colors[center_idx] = init_solid_colors[color_idx]
		if self.anime_scatter is not None:
			self.anime_scatter.remove()
		self.anime_scatter = self.anime_ax.scatter(self.datapoints[:, 0], self.datapoints[:, 1], c = colors)
		return self.anime_scatter,

	def animation_update(self, ith_frame):
		if ith_frame > self.anime_start_delay:
			self.update()
			new_color = np.array([self.cluster_colors[k] for k in self.k_clusters])
			# print(new_color)
			self.anime_scatter.set_facecolor(new_color)
		return self.anime_scatter,

