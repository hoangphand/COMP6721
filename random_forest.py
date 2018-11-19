from __future__ import division
import utilities
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


ds = 1
is_original_data = True
# is_original_data = False
is_on_test_set = False
# is_on_test_set = True

if ds == 1:
	# DS1
	ds_train_path = utilities.DS1_TRAIN_PATH
	ds_label_size = utilities.DS1_LABEL_SIZE
	if is_on_test_set:
		ds_val_path = utilities.DS1_TEST_PATH
	else:
		ds_val_path = utilities.DS1_VAL_PATH
else:
	# DS2
	ds_train_path = utilities.DS2_TRAIN_PATH
	ds_label_size = utilities.DS2_LABEL_SIZE
	if is_on_test_set:
		ds_val_path = utilities.DS2_TEST_PATH
	else:
		ds_val_path = utilities.DS2_VAL_PATH

if is_original_data:
	# ORIGINAL DATA
	(train_set, train_label, count_label) = utilities.load_dataset(ds_train_path, ds_label_size)
	if is_on_test_set:
		val_set = utilities.load_testset(ds_val_path)
	else:
		(val_set, val_label, val_count_label) = utilities.load_dataset(ds_val_path, ds_label_size)
else:
	# DATA DIM-DEDUCTION
	threshold = 0.95
	(train_set, train_label, count_label, indices) = utilities.train_dim_deduction_variance(ds_train_path, ds_label_size, threshold)
	if is_on_test_set:
		val_set = utilities.load_testset(ds_val_path)
	else:
		(val_set, val_label, val_count_label) = utilities.val_dim_deduction_variance(ds_val_path, ds_label_size, indices)

# FIND BEST NUMBER OF TREES
# no_of_trees = 50
# while no_of_trees <= 500:
# 	# GINI
# 	clf = RandomForestClassifier(n_estimators=no_of_trees, random_state=0)
# 	# ENTROPY
# 	# clf = RandomForestClassifier(n_estimators=no_of_trees, criterion="entropy", random_state=0)

# 	clf = clf.fit(train_set, train_label)

# 	predictions = clf.predict(val_set)
# 	print(str(no_of_trees) + ": " + str(round(utilities.cal_accuracy(predictions, val_label), 2)))

# 	no_of_trees = no_of_trees + 50

best_no_of_tree = 275
clf = RandomForestClassifier(n_estimators=best_no_of_tree, random_state=0)
clf = clf.fit(train_set, train_label)
predictions = clf.predict(val_set)

if is_on_test_set is False:
	print(utilities.cal_accuracy(predictions, val_label))

# FIND BEST MAX DEPTH VALUE
# for depth in range(1, 40):
# 	best_no_of_tree = 250
# 	clf = RandomForestClassifier(n_estimators=best_no_of_tree, random_state=0, max_depth=depth)
# 	clf = clf.fit(train_set, train_label)
# 	predictions = clf.predict(val_set)
# 	print("depth: " + str(depth) + ", accu: " + str(utilities.cal_accuracy(predictions, val_label)))

if ds == 1:
	joblib.dump(clf, utilities.DS1_3_MODEL)
	if is_on_test_set:
		utilities.save_output(predictions, utilities.DS1_TEST_3_OUT)
	else:
		utilities.save_output(predictions, utilities.DS1_VAL_3_OUT)
else:
	joblib.dump(clf, utilities.DS2_3_MODEL)
	if is_on_test_set:
		utilities.save_output(predictions, utilities.DS2_TEST_3_OUT)
	else:
		utilities.save_output(predictions, utilities.DS2_VAL_3_OUT)