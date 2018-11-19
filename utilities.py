from __future__ import division
import csv
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel

# DS1
DS1_IMG_HEIGHT = 32
DS1_IMG_WIDTH = 32
DS1_TRAIN_SIZE = 1960
DS1_VAL_SIZE = 514
DS1_TEST_SIZE = 514
DS1_LABEL_SIZE = 51
# INPUT FILES
DS1_TRAIN_PATH = 'ds/ds1/ds1Train.csv'
DS1_VAL_PATH = 'ds/ds1/ds1Val.csv'
DS1_TEST_PATH = 'ds/ds1/ds1Test.csv'
# OUTPUT FILES
DS1_VAL_DT_OUT = 'output/ds1Val-dt.csv'
DS1_VAL_NB_OUT = 'output/ds1Val-nb.csv'
DS1_VAL_3_OUT = 'output/ds1Val-3.csv'
DS1_TEST_DT_OUT = 'output/ds1Test-dt.csv'
DS1_TEST_NB_OUT = 'output/ds1Test-nb.csv'
DS1_TEST_3_OUT = 'output/ds1Test-3.csv'
DS1_DT_MODEL = 'models/ds1-dt.joblib'
DS1_NB_MODEL = 'models/ds1-nb.joblib'
DS1_3_MODEL = 'models/ds1-3.joblib'

# DS2
DS2_IMG_HEIGHT = 32
DS2_IMG_WIDTH = 32
DS2_TRAIN_SIZE = 6400
DS2_VAL_SIZE = 2000
DS2_TEST_SIZE = 2000
DS2_LABEL_SIZE = 10
# INPUT FILES
DS2_TRAIN_PATH = 'ds/ds2/ds2Train.csv'
DS2_VAL_PATH = 'ds/ds2/ds2Val.csv'
DS2_TEST_PATH = 'ds/ds2/ds2Test.csv'
# OUTPUT FILES
DS2_VAL_DT_OUT = 'output/ds2Val-dt.csv'
DS2_VAL_NB_OUT = 'output/ds2Val-nb.csv'
DS2_VAL_3_OUT = 'output/ds2Val-3.csv'
DS2_TEST_DT_OUT = 'output/ds2Test-dt.csv'
DS2_TEST_NB_OUT = 'output/ds2Test-nb.csv'
DS2_TEST_3_OUT = 'output/ds2Test-3.csv'
DS2_DT_MODEL = 'models/ds2-dt.joblib'
DS2_NB_MODEL = 'models/ds2-nb.joblib'
DS2_3_MODEL = 'models/ds2-3.joblib'

def load_dataset(dataset_path, label_size):
	label = []
	dataset = []
	with open(dataset_path, 'rb') as input_file:
		reader = csv.reader(input_file)

		for row in reader:
			label.append(int(row[len(row) - 1]))
			dataset.append([int(row[i]) for i in range(0, len(row) - 1)])

	count_label = []
	for i in range(label_size):
		count_label.append(0)

	for row in label:
		count_label[row] = count_label[row] + 1

	return (dataset, label, count_label)

def load_testset(dataset_path):
	dataset = []
	with open(dataset_path, 'rb') as input_file:
		reader = csv.reader(input_file)

		for row in reader:
			dataset.append([int(row[i]) for i in range(0, len(row))])

	return dataset

def cal_accuracy(predictions, actual):
	correct_count = 0
	for row in range(len(predictions)):
		if predictions[row] == actual[row]:
			correct_count = correct_count + 1
	return round(correct_count / len(predictions) * 100, 2)

def save_output(predictions, path):
	with open(path, 'w') as output_file:
		writer = csv.writer(output_file)
		for index in range(0, len(predictions)):
			writer.writerow([index + 1, predictions[index]])

def train_dim_deduction_variance(train_path, label_size, threshold):
	(train_set, train_label, count_label) = load_dataset(train_path, label_size)

	sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))

	new_train_set = sel.fit_transform(train_set)
	indices = sel.get_support(True)

	return (new_train_set, train_label, count_label, indices)

def val_dim_deduction_variance(val_path, label_size, indices):
	(val_set, val_label, val_count_label) = load_dataset(val_path, label_size)

	new_val_set = []

	for i in range(len(val_set)):
		new_val_set.append([val_set[i][j] for j in indices])

	return (new_val_set, val_label, val_count_label)