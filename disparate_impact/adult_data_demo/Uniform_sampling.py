from prepare_adult_data import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import multiprocessing as mp

sys.path.insert(0, '../../fair_classification/')
import utils as ut
import loss_funcs as lf

NUM_FOLDS=10

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

def Train_test_classifier(random_x_train, random_y_train, random_x_control_train):
    w1 = ut.train_model(random_x_train, random_y_train, random_x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
    train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w1, random_x_train,random_y_train, x_test, y_test, None, None)
    distances_boundary_test = (np.dot(x_test, w1)).tolist()
    all_class_labels_assigned_test = np.sign(distances_boundary_test)
    correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
    cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
    p_rule = print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])   
    return w1, p_rule, test_score

def generate_uniform_sampling(x_train, y_train, x_control_train, num_points):
    num_rows = len(x_train)
    # Generate array P containing parts of uniform samples
    P = np.linspace(0, 1, num_rows + 1)[1:]

    # Choose num_points random indices
    random_indices = np.random.choice(num_rows, num_points, replace=True)

    x_control_train_df = pd.DataFrame(x_control_train)

    # Use iloc to index x_control_train based on random_indices
    random_x_train = x_train[random_indices]
    random_y_train = y_train[random_indices]
    random_x_control_train = x_control_train_df.iloc[random_indices]

    return random_x_train, random_y_train, random_x_control_train

def plot_num_points_vs_accuracy(num_points_list, accuracy_list):
    plt.plot(num_points_list, accuracy_list, marker='o')
    plt.xlabel('Number of Points')
    plt.ylabel('Accuracy')
    plt.title('Number of Points vs. Accuracy')
    plt.grid(True)
    plt.xticks(np.arange(300, max(num_points_list)+1,300))
    plt.show()

def plot_num_points_vs_P_rule(num_points,p_rule_list):
    plt.plot(num_points,p_rule_list, marker='o', color='orange', label='Number of points vs. P-rule')
    plt.xlabel('Number of Points')
    plt.ylabel('P-rule')
    plt.title('Number of points vs. P-rule')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Load the adult data
    X, y, x_control = load_adult_data() 
    ut.compute_p_rule(x_control["sex"], y) 

    # Split the data into train and test
    X = ut.add_intercept(X) 
    train_fold_size = 0.9
    x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

    loss_function = lf._logistic_loss
    sensitive_attrs = ["sex"]
    sensitive_attrs_to_cov_thresh = {}
    gamma = None

    # Define the list of num_points values
    num_points_list = [300, 500, 1500, 2500, 5000]

    # Lists to store number of points and corresponding accuracies
    accuracy_list = []
    p_rule_list = []

    accuracy_list2 = []
    p_rule_list2 = []

    # Loop through each num_points value
    for num_points in num_points_list:
        avg_acc_f = 0
        avg_acc_a = 0
        
        for _ in range(5):  # Run each configuration five times
            random_x_train, random_y_train, random_x_control_train = generate_uniform_sampling(x_train, y_train, x_control_train, num_points)

            ut.compute_p_rule(random_x_control_train["sex"], random_y_train)
            
            apply_fairness_constraints = 1
            apply_accuracy_constraint = 0
            sep_constraint = 0
            cov_factor = 0
            sensitive_attrs_to_cov_thresh = {"sex": 0}
            
            print("== Classifier with fairness constraint ==")
            w_f_, prule_f, acc_f = Train_test_classifier(random_x_train, random_y_train, random_x_control_train)
            
            avg_acc_f += acc_f

            """ Classify such that we optimize for fairness subject to a certain loss in accuracy """ 
            apply_fairness_constraints = 0 
            apply_accuracy_constraint = 1 
            sep_constraint = 0
            gamma = 0.5 
            print("== Classifier with accuracy constraint ==")
            w_a_, prule_a, acc_a = Train_test_classifier(random_x_train, random_y_train, random_x_control_train)
            
            avg_acc_a += acc_a

        # Calculate average accuracy
        avg_acc_f /= 5
        avg_acc_a /= 5

        # Append average accuracy to the lists
        accuracy_list.append(avg_acc_f)
        p_rule_list.append(prule_f)
        
        accuracy_list2.append(avg_acc_a)
        p_rule_list2.append(prule_a)

        print(f"Average Accuracy (Fairness Constraint): {avg_acc_f}")
        print(f"Average Accuracy (Accuracy Constraint): {avg_acc_a}")

        

    # Plotting
    plot_num_points_vs_accuracy(num_points_list, accuracy_list)
    plot_num_points_vs_accuracy(num_points_list, accuracy_list2)

    plot_num_points_vs_P_rule(num_points_list,p_rule_list)
    plot_num_points_vs_P_rule(num_points_list,p_rule_list2)
