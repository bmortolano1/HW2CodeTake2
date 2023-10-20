import numpy as np
from DecisionTree import decision_tree as dt
import matplotlib.pyplot as plt

def randomly_sample(table, sample_size):
    return table[np.random.randint(0, np.size(table,0), sample_size), :]

def bag_single_tree(type, table, max_tree_depth, depth, attributes, attribute_vals, sample_size):
    subsample = randomly_sample(table, sample_size)
    weights = np.ones(np.size(subsample, 0))
    mcv = dt.most_common_value(subsample[:, -1], weights)
    tree = dt.id3(type, subsample, max_tree_depth, depth, attributes, attribute_vals, mcv, weights)
    return tree

def combine(trees, attributes, test_table):
    n_cor = 0
    n_incor = 0

    for i in range(np.size(test_table, 0)):
        row = test_table[i]
        x = row[0:-1]
        y = row[-1]

        result_votes = {}

        # Take votes from all trees
        for j in range(np.size(trees)):
            tree = trees[j]

            result = dt.predict_value(tree, attributes, x)

            if result in result_votes.keys():
                result_votes[result] = result_votes[result] + 1
            else:
                result_votes[result] = 1

        final_result = 0
        best_vote = 0

        for key in result_votes.keys():
            if result_votes[key] > best_vote:
                best_vote = result_votes[key]
                final_result = key

        if final_result == y: # Correct
            n_cor = n_cor + 1
        else: # Incorrect
            n_incor = n_incor + 1

    error = n_incor / (n_incor + n_cor)

    return error

def perform_bagging(type, table_test, table_train, max_tree_depth, attributes, attribute_vals, sample_size, num_trees, plot, write, fileName):
    trees = []

    global_test_errors = []
    global_train_errors = []

    if write:
        file1 = open(fileName, "w")

    for i in range(num_trees):
        print(i)

        trees.append(bag_single_tree(type, table_train, max_tree_depth,0, attributes, attribute_vals, sample_size))

        if write or plot:
            global_test_error = combine(trees, attributes, table_test)
            global_train_error = combine(trees, attributes, table_train)

        if write:
            global_test_errors.append(global_test_error)
            global_train_errors.append(global_train_error)

            line = [i, global_test_error, global_train_error]

            file1.write(str(line) + "\n")

        if plot and i%10 == 1:
            plt.figure(1)
            plt.plot(global_test_errors)
            plt.title("Global Test Errors")

            plt.figure(2)
            plt.plot(global_train_errors)
            plt.title("Global Train Errors")


    if write:
        file1.close()

    return trees

def multiple_bagged_predictors(type, table_test, table_train, max_tree_depth, attributes, attribute_vals, sample_size, sample_size_outer, num_trees, num_bag_preds, fileName):

    bagged_predictors = []
    file1 = open(fileName, "w")

    for i in range(num_bag_preds):

        # Train with small subtable
        subtable1 = randomly_sample(table_test, sample_size_outer)

        predictor = perform_bagging(3, False, subtable1, max_tree_depth, attributes, attribute_vals, sample_size, num_trees, False, False, "")
        bagged_predictors.append(predictor)

        single_tree = predictor[0]

        results_single = []
        results_aggregate = []

        for row in table_test:
            x = row[0:-1]
            y = row[-1]

            if y == "yes":
                y = 1
            elif y == "no":
                y = 0

            result = dt.predict_value(single_tree, attributes, x)
            result_aggregate = combine(predictor, attributes, [row])

            if result == "yes":
                result = 1
            elif result == "no":
                result = 0

            if result_aggregate == "yes":
                result_aggregate = 1
            elif result_aggregate == "no":
                result_aggregate = 0

            results_single.append(result)
            results_aggregate.append(result_aggregate)

        bias_single = (np.mean(results_single) - y) ** 2  # Bias of one tree
        var_single = np.var(results_single)  # Variance of one tree

        bias_aggregate = (np.mean(results_aggregate) - y) ** 2  # Bias of one tree
        var_aggregate = np.var(results_aggregate)  # Variance of one tree

        line = [i, bias_single, var_single, bias_aggregate, var_aggregate]
        file1.write(str(line) + "\n")

        print(i)