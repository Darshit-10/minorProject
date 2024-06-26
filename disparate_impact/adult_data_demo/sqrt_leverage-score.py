import numpy as np
import pandas as pd
from prepare_adult_data import *
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf
import Uniform_sampling as us
import time

def print_classifier_fairness_stats(acc_arr, correlation_dict_arr, cov_dict_arr, s_attr_name):
    
    correlation_dict = ut.get_avg_correlation_dict(correlation_dict_arr)
    non_prot_pos = correlation_dict[s_attr_name][1][1]
    prot_pos = correlation_dict[s_attr_name][0][1]
    p_rule = (prot_pos / non_prot_pos) * 100.0
    
    print ("Accuracy: %0.2f" % (np.mean(acc_arr)))
    print ("Protected/non-protected in +ve class: %0.0f%% / %0.0f%%" % (prot_pos, non_prot_pos))
    print ("P-rule achieved: %0.0f%%" % (p_rule))
    print()
    return p_rule


def sqrt_leverage_score_sampling_svd(x_train, y_train, x_control_train, num_points):
    # Ensure y_train is a column vector
    y_train_col = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train

    # Check if x_control_train is already a DataFrame, if not convert it
    x_control_train_df = x_control_train if isinstance(x_control_train, pd.DataFrame) else pd.DataFrame(x_control_train)

    # Concatenate x_train, y_train_col, and x_control_train_df to form the dataset matrix
    dataset_matrix = np.concatenate((x_train, y_train_col, x_control_train_df.values), axis=1)

    # Perform Singular Value Decomposition
    U, Sigma, Vt = np.linalg.svd(dataset_matrix, full_matrices=False)

    # Calculate the leverage scores (squared norms of the rows of U)
    leverage_scores = np.linalg.norm(U, axis=1)**2

    # Calculate square root of leverage scores
    sqrt_leverage_scores = np.sqrt(leverage_scores)

    # Normalize the square root leverage scores to get a probability distribution
    sqrt_leverage_scores_normalized = sqrt_leverage_scores / np.sum(sqrt_leverage_scores)

    # Choose num_points random indices based on the normalized square root leverage scores
    random_indices = np.random.choice(len(dataset_matrix), num_points, replace=False, p=sqrt_leverage_scores_normalized)

    # Extract the corresponding rows from the datasets
    random_x_train = x_train[random_indices]
    random_y_train = y_train[random_indices]
    random_x_control_train = x_control_train_df.iloc[random_indices]

    return random_x_train, random_y_train, random_x_control_train


def test_leverage_score_sampling():

    """ Load the adult data """
    X, y, x_control = load_adult_data() # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
    ut.compute_p_rule(x_control['sex'], y) # compute the p-rule in the original data



    """ Split the data into train and test """
    X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
    train_fold_size = 0.9
    x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

    loss_function = lf._logistic_loss
    sensitive_attrs = ["sex"]
    sensitive_attrs_to_cov_thresh = {}
    gamma = None

    def train_test_classifier(random_x_train, random_y_train, random_x_control_train):
        w1 = ut.train_model(random_x_train, random_y_train, random_x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
        train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w1, random_x_train,random_y_train, X, y, None, None)
        distances_boundary_test = (np.dot(x_test, w1)).tolist()
        all_class_labels_assigned_test = np.sign(distances_boundary_test)
        correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
        cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
        p_rule = print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])   
        return w1, p_rule, test_score

    # Define the list of num_points values
    num_points_list = [500, 1000, 1500, 2500, 3000]

    # Lists to store number of points and corresponding accuracies
    accuracy_list = []
    p_rule_list = []

    accuracy_list2 = []
    p_rule_list2 = []

    # Loop through each num_points value
    for num_points in num_points_list:
        total_acc_1 = 0
        total_acc_2 = 0
        total_prule_1 = 0
        total_prule_2 = 0
        exe_sample =0
        exe_fairness = 0
        exe_accuracy = 0
        for _ in range(5):  # Run each configuration five times
            start_time = time.time()
            random_x_train, random_y_train, random_x_control_train = sqrt_leverage_score_sampling_svd(x_train, y_train, x_control_train, num_points)
            end_time = time.time()

            # Calculate the execution time
            e = end_time - start_time
            exe_sample+=e
            ut.compute_p_rule(random_x_control_train["sex"], random_y_train)

            
            apply_fairness_constraints = 1
            apply_accuracy_constraint = 0
            sep_constraint = 0
            sensitive_attrs_to_cov_thresh = {"sex": 0}
            
            print("== Classifier with fairness constraint ==")
            start_time1 = time.time()
            w_f_, prule_f, acc_f = train_test_classifier(random_x_train, random_y_train, random_x_control_train)
            end_time1 = time.time()
            

            # Calculate the execution time
            e1 = end_time1 - start_time1
            exe_fairness+=e1

            total_acc_1 += acc_f
            total_prule_1 += prule_f

            """ Classify such that we optimize for fairness subject to a certain loss in accuracy """ 
            apply_fairness_constraints = 0 
            apply_accuracy_constraint = 1 
            sep_constraint = 0
            gamma = 0.5 
            print("== Classifier with accuracy constraint ==")
            start_time2 = time.time()
            w_a_, prule_a, acc_a = train_test_classifier(random_x_train, random_y_train, random_x_control_train)
            end_time2 = time.time()
            
            # Calculate the execution time
            e2 = end_time2 - start_time2
            exe_accuracy+=e2

            total_acc_2 += acc_a
            total_prule_2 += prule_a

        # Calculate average accuracy
        avg_acc_f = total_acc_1 / 5
        avg_acc_a = total_acc_2 / 5
        avg_prule_f = total_prule_1 / 5
        avg_prule_a = total_prule_2 / 5
        avg_exe_sample = exe_sample/5
        avg_exe_fairness = exe_fairness/5
        avg_exe_accuracy = exe_accuracy/5
 
        # Append average accuracy to the lists
        accuracy_list.append(avg_acc_f)
        p_rule_list.append(avg_prule_f)
        
        accuracy_list2.append(avg_acc_a)
        p_rule_list2.append(avg_prule_a)

        print(f"Average Accuracy (Fairness Constraint): {avg_acc_f}")
        print(f"Average p%-rule (Fairness Constraint): {avg_prule_f}")
        print(f"Average Accuracy (Accuracy Constraint): {avg_acc_a}")
        print(f"Average p%-rule (Accuracy Constraint): {avg_prule_a}")
        print("execution time of sample = ",avg_exe_sample)
        print("execution time of fairness = ",avg_exe_fairness)
        print("execution time of accuracy = ",avg_exe_accuracy)
        

    # Plotting
    us.plot_num_points_vs_accuracy_and_P_rule(num_points_list, accuracy_list, p_rule_list, "Fairness","sqrt_levarage_score")
    us.plot_num_points_vs_accuracy_and_P_rule(num_points_list, accuracy_list2, p_rule_list2, "Accuracy","sqrt_levarage_score")


def main():
	test_leverage_score_sampling()


if __name__ == '__main__':
	main()