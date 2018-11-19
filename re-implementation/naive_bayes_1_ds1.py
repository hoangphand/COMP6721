from __future__ import division
import csv
from PIL import Image
import sys
import utilities

train_label = []
train_set = []
train_count_label = []
(train_set, train_label, count_label) = utilities.load_dataset(utilities.DS1_TRAIN_PATH, utilities.DS1_LABEL_SIZE)

prior_els = []

for i in range(utilities.DS1_LABEL_SIZE):
	el = []
	for j in range(len(train_set[0])):
		el.append(0)
	prior_els.append(el)

for i in range(utilities.DS1_TRAIN_SIZE):
	for j in range(len(train_set[0])):
		prior_els[train_label[i]][j] = prior_els[train_label[i]][j] + train_set[i][j]

val_label = []
val_set = []
val_count_label = []
(val_set, val_label, val_count_label) = utilities.load_dataset(utilities.DS1_VAL_PATH, utilities.DS1_LABEL_SIZE)

correct_count = 0

for row in range(utilities.DS1_VAL_SIZE):
	predictions = []
	for i in range(utilities.DS1_LABEL_SIZE):
		tmp = 1
		for j in range(len(train_set[0])):
			if val_set[row][j] == 1:
				tmp = tmp * (prior_els[i][j] + 1) / (count_label[i])
			else:
				tmp = tmp * (count_label[i] - prior_els[i][j] + 1) / (count_label[i]) 
		predictions.append(tmp * count_label[i] / utilities.DS1_TRAIN_SIZE)

	prediction = -1

	for i in range(utilities.DS1_LABEL_SIZE):
		if predictions[i] > prediction:
			prediction = predictions[i]
			ans = i

	print("prediction: " + str(ans))
	print("actual: " + str(val_label[row]))

	if ans == val_label[row]:
		correct_count = correct_count + 1

print(correct_count / utilities.DS1_VAL_SIZE)
