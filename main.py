from DecisionTree import decision_tree as dt
from EnsembleLearning import boosting as boosting
from EnsembleLearning import bagging as bag
import numpy as np
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Parse files
    bank_table_train = dt.parse_file("./bank-4/train.csv")
    print(bank_table_train)
    bank_table_test = dt.parse_file("./bank-4/test.csv")

    bank_attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                       'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    bank_attribute_values = [['under_med', 'over_med'],
                             ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
                              "blue-collar", "self-employed", "retired", "technician", "services"],
                             ["married", "divorced", "single"],
                             ["unknown", "secondary", "primary", "tertiary"],
                             ["yes", "no"],
                             ['under_med', 'over_med'],
                             ["yes", "no"],
                             ["yes", "no"],
                             ["unknown", "telephone", "cellular"],
                             ['under_med', 'over_med'],
                             ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                             ['under_med', 'over_med'],
                             ['under_med', 'over_med'],
                             ['under_med', 'over_med'],
                             ['under_med', 'over_med'],
                             ["unknown", "other", "failure", "success"]]

    # Transform numerical data into binary data
    for i in [0, 5, 9, 11, 12, 13, 14]:
        bank_table_train = dt.trans_numer_to_bin(bank_table_train, i)
        bank_table_test = dt.trans_numer_to_bin(bank_table_test, i)

    ######################## BEGIN BOOSTING ##################################

    # # Add column for weights. Initialize all weights to one.
    # weights = np.ones((bank_table_test.shape[0], 1), dtype=np.float32)
    #
    # # Run all three with different tree heights:
    # # Create "stump trees" with Gini Index for Information Gain. These will serve as our weak learners:
    #
    # iters = 500
    # votes = np.zeros(iters)
    # trees = []
    # weighted_test_errors = []
    # weighted_train_errors = []
    # unweighted_test_errors = []
    # unweighted_train_errors = []
    # votes = []
    #
    # global_test_errors = []
    # global_train_errors = []
    #
    # file1 = open("Part1DataTake2.txt", "w")
    #
    # for t in range(500):
    #
    #     tree = dt.id3(3, bank_table_train, 1, 0, bank_attributes, bank_attribute_values,
    #                          dt.most_common_value(bank_table_train[:, -1], weights), weights)
    #     [vote, train_error, _, new_weights, train_error_unweighted] = boosting.adaboost(tree, bank_attributes, bank_table_train, weights)
    #     [_, test_error, _, _, test_error_unweighted] = boosting.adaboost(tree, bank_attributes, bank_table_test, weights)
    #
    #     trees.append(tree)
    #     weighted_test_errors.append(test_error)
    #     weighted_train_errors.append(train_error)
    #     unweighted_test_errors.append(test_error_unweighted)
    #     unweighted_train_errors.append(train_error_unweighted)
    #     votes.append(vote)
    #     weights = new_weights
    #
    #     global_test_error = boosting.combine(trees, bank_attributes, bank_table_test, votes)
    #     global_train_error = boosting.combine(trees, bank_attributes, bank_table_train, votes)
    #
    #     global_test_errors.append(global_test_error)
    #     global_train_errors.append(global_train_error)
    #
    #     line = [t, test_error, train_error, test_error_unweighted, train_error_unweighted, global_test_error, global_train_error]
    #     file1.write(str(line) + "\n")
    #
    #     if t%10 == 1: # Plot every ten iterations
    #         plt.figure(1)
    #         plt.plot(weighted_test_errors)
    #         plt.title("Weighted Test Errors")
    #
    #         plt.figure(2)
    #         plt.plot(weighted_train_errors)
    #         plt.title("Weighted Train Errors")
    #
    #         plt.figure(3)
    #         plt.plot(unweighted_test_errors)
    #         plt.title("Unweighted Test Errors")
    #
    #         plt.figure(4)
    #         plt.plot(unweighted_train_errors)
    #         plt.title("Unweighted Train Errors")
    #
    #         plt.figure(5)
    #         plt.plot(global_test_errors)
    #         plt.title("Global Test Errors")
    #
    #         plt.figure(6)
    #         plt.plot(global_train_errors)
    #         plt.title("Global Train Errors")
    #
    #         plt.show(block=False)
    #         plt.pause(3)
    #
    #     print(t)
    #
    # file1.close()

    ######################## BEGIN BAGGING ##################################
    # I put all of this in one function because I decided to be more organized now

    # Part B
    # bag.perform_bagging(3, bank_table_test, bank_table_train, 100, bank_attributes, bank_attribute_values, 500, 500, False, True, "Part2Data.txt")

    # Part C
    # bag.multiple_bagged_predictors(type, bank_table_test, bank_table_train, 100, bank_attributes, bank_attribute_values, 250, 1000, 500, 100, "Part3Data.txt")

    # Part D - Size 2
    bag.perform_random_forests(3, bank_table_test, bank_table_train, 100, bank_attributes, bank_attribute_values, 2000, 1500,
                        False, True, "Part4Data2.txt", 2)

    # Part D - Size 4
    bag.perform_random_forests(3, bank_table_test, bank_table_train, 100, bank_attributes, bank_attribute_values, 2000,
                               1500,
                               False, True, "Part4Data4.txt", 4)

    # Part D - Size 6
    bag.perform_random_forests(3, bank_table_test, bank_table_train, 100, bank_attributes, bank_attribute_values, 2000,
                               1500,
                               False, True, "Part4Data6.txt", 6)
