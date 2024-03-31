import os,sys
import numpy as np
from prepare_adult_data import *
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints

NUM_FOLDS = 10 # we will show 10-fold cross validation accuracy as a performance measure

def test_synthetic_data():
	
	
	X, y, x_control = load_adult_data() 
	ut.compute_p_rule(x_control["sex"], y) # compute the p-rule in the original data

	""" Classify the data without any constraints """
	apply_fairness_constraints = 0
	apply_accuracy_constraint = 0
	sep_constraint = 0
	gamma=0

	loss_function = lf._logistic_loss
	X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
	test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr = ut.compute_cross_validation_error(X, y, x_control, NUM_FOLDS, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, ['sex'], [{} for i in range(0,NUM_FOLDS)],gamma)
	print()
	print ("== Unconstrained (original) classifier ==")
	ut.print_classifier_fairness_stats(test_acc_arr, correlation_dict_test_arr, cov_dict_test_arr, "sex")


	""" Now classify such that we achieve perfect fairness """
	apply_fairness_constraints = 1
	cov_factor = 0
	test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr = ut.compute_cross_validation_error(X, y, x_control, NUM_FOLDS, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, ['sex'], [{'sex':cov_factor} for i in range(0,NUM_FOLDS)],gamma)		
	print()
	print ("== Constrained (fair) classifier ==")
	ut.print_classifier_fairness_stats(test_acc_arr, correlation_dict_test_arr, cov_dict_test_arr, "sex")

	""" Now plot a tradeoff between the fairness and accuracy """
	ut.plot_cov_thresh_vs_acc_pos_ratio(X, y, x_control, NUM_FOLDS, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, ['sex'])

	

def main():
	test_synthetic_data()


if __name__ == '__main__':
	main()