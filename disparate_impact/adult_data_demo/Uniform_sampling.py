from prepare_adult_data import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import multiprocessing as mp
import time
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

def Train_test_classifier(random_x_train, random_y_train, random_x_control_train,apply_fairness_constraints, apply_accuracy_constraint):
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
    random_indices = np.random.choice(num_rows, num_points, replace=False)
    random_x_train = x_train[random_indices]
    random_y_train = y_train[random_indices]
    random_x_control_train = {attr: x_control_train[attr][random_indices] for attr in x_control_train}

    return random_x_train, random_y_train, random_x_control_train
 

def plot_num_points_vs_accuracy_and_P_rule(num_points_list, accuracy_list, p_rule_list, constraint_type):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(num_points_list, accuracy_list, marker='o', color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('P-rule', color=color)  
    ax2.plot(num_points_list, p_rule_list, marker='o', color=color, label='P-rule')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title(f'Number of Points vs. Accuracy and P-rule ({constraint_type} Constraint)')
    plt.grid(True)
    plt.xticks(np.arange(0, max(num_points_list)+1, 100))  # Adjusting x-axis scale
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
    # Lists to store number of points and corresponding accuracies and p-rules
    accuracy_list_f = []
    p_rule_list_f = []
    accuracy_list_a = []
    p_rule_list_a = []
    # Loop through each num_points value
    for num_points in num_points_list:
        total_acc_f = 0
        total_acc_a = 0
        total_prule_f = 0
        total_prule_a = 0
        exe_sample =0
        exe_fairness = 0
        exe_accuracy = 0
        print(" For sample size",num_points)
        for _ in range(5):  # Run each configuration five times
            start_time = time.time()
            random_x_train, random_y_train, random_x_control_train = generate_uniform_sampling(x_train, y_train, x_control_train, num_points)
            end_time = time.time()

            # Calculate the execution time
            e = end_time - start_time
            exe_sample+=e

            # Classifier with fairness constraint

            print("== Classifier with fairness constraint ==")
            apply_fairness_constraints = 1
            apply_accuracy_constraint = 0
            sep_constraint = 0
            
            sensitive_attrs_to_cov_thresh = {"sex": 0}
            start_time1 = time.time()
            w_f, prule_f, acc_f = Train_test_classifier(random_x_train, random_y_train, random_x_control_train, apply_fairness_constraints, apply_accuracy_constraint)
            end_time1 = time.time()
            

            # Calculate the execution time
            e1 = end_time1 - start_time1
            exe_fairness+=e1

            total_acc_f += acc_f
            total_prule_f += prule_f

            # Classifier with accuracy constraint
            print("== Classifier with accuracy constraint ==")
            apply_fairness_constraints = 0
            apply_accuracy_constraint = 1
            sep_constraint=0
            gamma = 0.5
            start_time2 = time.time()
            w_a, prule_a, acc_a = Train_test_classifier(random_x_train, random_y_train, random_x_control_train, apply_fairness_constraints, apply_accuracy_constraint)
            end_time2 = time.time()
            
            # Calculate the execution time
            e2 = end_time2 - start_time2
            exe_accuracy+=e2
            total_acc_a += acc_a
            total_prule_a += prule_a

        # Calculate average accuracy and p-rule
        avg_acc_f = total_acc_f / 5
        avg_acc_a = total_acc_a / 5
        avg_prule_f = total_prule_f / 5
        avg_prule_a = total_prule_a / 5
        avg_exe_sample = exe_sample/5
        avg_exe_fairness = exe_fairness/5
        avg_exe_accuracy = exe_accuracy/5

        # Append average accuracy and p-rule to the lists
        accuracy_list_f.append(avg_acc_f)
        p_rule_list_f.append(avg_prule_f)
        
        accuracy_list_a.append(avg_acc_a)
        p_rule_list_a.append(avg_prule_a)

   

        print(f"Average Accuracy (Fairness Constraint): {avg_acc_f}")
        print(f"Average p%-rule (Fairness Constraint): {avg_prule_f}")
        print(f"Average Accuracy (Accuracy Constraint): {avg_acc_a}")
        print(f"Average p%-rule (Accuracy Constraint): {avg_prule_a}")
        print("execution time of sample = ",avg_exe_sample)
        print("execution time of fairness = ",avg_exe_fairness)
        print("execution time of accuracy = ",avg_exe_accuracy)

    # Plotting
    plot_num_points_vs_accuracy_and_P_rule(num_points_list, accuracy_list_f, p_rule_list_f, "Fairness")
    plot_num_points_vs_accuracy_and_P_rule(num_points_list, accuracy_list_a, p_rule_list_a, "Accuracy")
