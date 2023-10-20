import numpy as np
from DecisionTree import decision_tree as dt

def adaboost(tree, attributes, test_table, weights):
    correct = np.zeros(np.size(test_table,0))
    n_cor = 0
    n_incor = 0
    n_cor_unweighted = 0
    n_incor_unweighted = 0
    pred = np.zeros(np.size(test_table,0), dtype=str)

    # Skip first row for value testing bc thats the weight
    for i in range(np.size(test_table, 0)):
        row = test_table[i]
        x = row[0:-1]
        y = row[-1]

        result = dt.predict_value(tree, attributes, x)
        pred[i] = result

        if result == y: # Correct
            correct[i] = 1
            n_cor = n_cor + weights[i]
            n_cor_unweighted = n_cor_unweighted + 1
        else: # Incorrect
            correct[i] = 0
            n_incor = n_incor + weights[i]
            n_incor_unweighted = n_incor_unweighted + 1

    error = n_incor / (n_incor + n_cor)
    error_unweighted = n_incor_unweighted / (n_incor_unweighted + n_cor_unweighted)
    vote = 0.5 * np.log((1-error)/error)

    multiplier = 2 * correct - 1
    new_weights = [np.exp(-vote * multiplier[i]) * weights[i] for i in range(np.size(test_table, 0))]

    # Normalize weights vector
    size = np.size(new_weights, 0)
    sum = np.sum(new_weights)

    new_weights = np.array([float(x) * size/sum for x in new_weights])

    return [vote, error, pred, new_weights, error_unweighted]

def combine(trees, attributes, test_table, votes):
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
            vote = votes[j]

            result = dt.predict_value(tree, attributes, x)

            if result in result_votes.keys():
                result_votes[result] = result_votes[result] + vote
            else:
                result_votes[result] = vote

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
