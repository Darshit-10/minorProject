import numpy as np
import pandas as pd
from prepare_adult_data import *
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf
import Uniform_sampling as us
NUM_FOLDS = 10 # we will show 10-fold cross validation accuracy as a performance measure

def test_synthetic_data():
	
	# Load the adult data
    X, y, x_control = load_adult_data() 
    ut.compute_p_rule(x_control["sex"], y) 

    # Split the data into train and test
    X = ut.add_intercept(X) 
    train_fold_size = 0.9
    x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)
	
     # Define the list of num_points values
    num_points_list = [300, 500, 1500, 2500, 5000]

    # Lists to store number of points and corresponding accuracies
    accuracy_list = []
    
    accuracy_list2 = []
    

    # Loop through each num_points value
    for num_points in num_points_list:

          # Run each configuration five times
        random_x_train, random_y_train, random_x_control_train = us.generate_uniform_sampling(x_train, y_train, x_control_train, num_points)

        ut.compute_p_rule(random_x_control_train["sex"], random_y_train)
            
        """ Now classify such that we achieve perfect fairness """
        apply_fairness_constraints = 0
        apply_accuracy_constraint = 0
        sensitive_attrs = ["sex"]
        sep_constraint =0
        gamma = 0

        loss_function = lf._logistic_loss
        X = ut.add_intercept(X) 
        test_acc_arr_f, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr = ut.compute_cross_validation_error(random_x_train, random_y_train, random_x_control_train, NUM_FOLDS, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint,sensitive_attrs, [{} for i in range(0,NUM_FOLDS)],gamma)		
        print()
        print ("== Constrained (fair) classifier ==")
        ut.print_classifier_fairness_stats(test_acc_arr_f, correlation_dict_test_arr, cov_dict_test_arr, "sex")

            
        """ Classify such that we optimize for fairness subject to a certain loss in accuracy """ 
        apply_accuracy_constraint = 1 
        cov_factor = 0
        test_acc_arr_a, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr = ut.compute_cross_validation_error(random_x_train, random_y_train, random_x_control_train, NUM_FOLDS, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, [{'sex':cov_factor} for i in range(0,NUM_FOLDS)],gamma)		
        print()
        print ("== Constrained (accuracy) classifier ==")
        ut.print_classifier_fairness_stats(test_acc_arr_a, correlation_dict_test_arr, cov_dict_test_arr, "sex")

        # Append average accuracy to the lists
        accuracy_list.append(test_acc_arr_f)
        #p_rule_list.append(prule_f)
        
        accuracy_list2.append(test_acc_arr_a)
        #p_rule_list2.append(prule_a)

        print(f"Average Accuracy (Fairness Constraint): {test_acc_arr_f}")
        print(f"Average Accuracy (Accuracy Constraint): {test_acc_arr_a}")


        """ Now plot a tradeoff between the fairness and accuracy """
        ut.plot_cov_thresh_vs_acc_pos_ratio(random_x_train, random_y_train, random_x_control_train, NUM_FOLDS, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, ['sex'])

	

def main():
	test_synthetic_data()


if __name__ == '__main__':
	main()