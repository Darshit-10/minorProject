import numpy as np
import pandas as pd
from prepare_adult_data import *
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf
import Uniform_sampling as us


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

def leverage_score_sampling_svd(x_train, y_train, x_control_train, num_points):
    x_control_train_df = pd.DataFrame(x_control_train)

    # Concatenate x_train, y_train, and x_control_train_df to form the dataset matrix
    dataset_matrix = np.concatenate((x_train, y_train.reshape(-1, 1), x_control_train_df), axis=1)

    # Perform Singular Value Decomposition
    U, Sigma, Vt = np.linalg.svd(dataset_matrix, full_matrices=False)

    # Calculate the norms of the rows of U
    norms = np.linalg.norm(U, axis=1)

    # Find the sum of values in the norms array
    sum_norms = np.sum(norms)

    # Divide each value in norms by the sum to obtain normalized norms (leverage scores)
    normalized_leverage_scores = norms / sum_norms

    # Choose num_points random indices based on the normalized leverage scores
    random_indices = np.random.choice(len(dataset_matrix), num_points, replace=True, p=normalized_leverage_scores)

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
    train_fold_size = 0.7
    x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

    loss_function = lf._logistic_loss
    sensitive_attrs = ["sex"]
    sensitive_attrs_to_cov_thresh = {}
    gamma = None

    def train_test_classifier(random_x_train, random_y_train, random_x_control_train):
        w1 = ut.train_model(random_x_train, random_y_train, random_x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
        train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w1, random_x_train,random_y_train, x_test, y_test, None, None)
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
        avg_acc_f = 0
        avg_acc_a = 0
        
        for _ in range(15):  # Run each configuration five times
            random_x_train, random_y_train, random_x_control_train = leverage_score_sampling_svd(x_train, y_train, x_control_train, num_points)
         
            ut.compute_p_rule(random_x_control_train["sex"], random_y_train)

            
            apply_fairness_constraints = 1
            apply_accuracy_constraint = 0
            sep_constraint = 0
            sensitive_attrs_to_cov_thresh = {"sex": 0}
            
            print("== Classifier with fairness constraint ==")
            w_f_, prule_f, acc_f = train_test_classifier(random_x_train, random_y_train, random_x_control_train)
            
            avg_acc_f += acc_f

            """ Classify such that we optimize for fairness subject to a certain loss in accuracy """ 
            apply_fairness_constraints = 0 
            apply_accuracy_constraint = 1 
            sep_constraint = 0
            gamma = 0.5 
            print("== Classifier with accuracy constraint ==")
            w_a_, prule_a, acc_a = train_test_classifier(random_x_train, random_y_train, random_x_control_train)
            
            avg_acc_a += acc_a

        # Calculate average accuracy
        avg_acc_f /= 15
        avg_acc_a /= 15
 
        # Append average accuracy to the lists
        accuracy_list.append(avg_acc_f)
        p_rule_list.append(prule_f)
        
        accuracy_list2.append(avg_acc_a)
        p_rule_list2.append(prule_a)

        print(f"Average Accuracy (Fairness Constraint): {avg_acc_f}")
        print(f"Average Accuracy (Accuracy Constraint): {avg_acc_a}")

        

    # Plotting
    us.plot_num_points_vs_accuracy(num_points_list, accuracy_list)
    us.plot_num_points_vs_accuracy(num_points_list, accuracy_list2)

    us.plot_num_points_vs_P_rule(num_points_list,p_rule_list)
    us.plot_num_points_vs_P_rule(num_points_list,p_rule_list2)

def main():
	test_leverage_score_sampling()


if __name__ == '__main__':
	main()