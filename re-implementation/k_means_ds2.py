from __future__ import division
import numpy as np
import sys
import utilities
import math

(train_set, train_label, count_label) = utilities.load_dataset(utilities.DS2_TRAIN_PATH, utilities.DS2_LABEL_SIZE)

train_centroids = []

for i in range(0, len(count_label)):
	centroid = []
	for j in range(0, len(train_set[0])):
		centroid.append(0)
	train_centroids.append(centroid)

for i in range(0, len(train_set)):
	for j in range(0, utilities.DS2_IMG_HEIGHT * utilities.DS2_IMG_WIDTH):
		train_centroids[train_label[i]][j] = train_centroids[train_label[i]][j] + train_set[i][j]

for i in range(0, len(train_centroids)):
	for j in range(0, utilities.DS2_IMG_HEIGHT * utilities.DS2_IMG_WIDTH):
		train_centroids[i][j] = train_centroids[i][j] / count_label[i]

# print(train_centroids[0][0])

# 	train_centroids[train_label[i]] = train_centroids[train_label[i]] + train_set[i]
# print(len(count_label))

(val_set, val_label, val_count_label) = utilities.load_dataset(utilities.DS2_VAL_PATH, utilities.DS2_LABEL_SIZE)

correct_count = 0
for i in range(0, len(val_set)):
	min_distance = sys.maxint
	min_k = -1
	for k in range(0, len(train_centroids)):
		cur_distance = 0
		for j in range(0, utilities.DS2_IMG_HEIGHT * utilities.DS2_IMG_WIDTH):
			cur_distance = cur_distance + math.pow(val_set[i][j] - train_centroids[k][j], 2)

		if cur_distance < min_distance:
			min_distance = cur_distance
			min_k = k

	print("predict: " + str(min_k))
	print("actual: " + str(val_label[i]))
	if min_k == val_label[i]:
		correct_count = correct_count + 1

print(correct_count / utilities.DS2_VAL_SIZE)