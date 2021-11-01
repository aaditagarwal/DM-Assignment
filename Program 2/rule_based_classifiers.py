import time
import numpy as np
import pandas as pd
from itertools import combinations, product
from sklearn.metrics import accuracy_score
from collections import defaultdict


class RULES:
    def __init__(self):
        self.contains_header = True
        self.discretize_mode = 'equal'
        self.number_bins = 7
        self.discretize_ints = False
        self.bins = []

        self.attribute_names = None
        self.preproc_dict = None
        self.labels_dict = None

        self.rules = []
        self.most_probable_class = None
        self.n_attributes = 0

    # ######################### F I T #########################
    def fit(self, x, y, method='RRULES', show_metrics=True):
        since = time.time()
        x, y = self.__preproc_train_data(x, y)
        since = time.time()
        if method == 'RRULES':
            print('Training with RRULES...')
            self.__fit_RRULES(x, y)

    def __fit_RRULES(self, x, y):
        # We calculate the most probable class to create a default rule for unseen combinations of attributes
        classes, counts = np.unique(y, return_counts=True)
        self.most_probable_class = classes[np.argmax(counts)]

        # ##### RRULES #####
        n_examples, n_attributes = x.shape
        self.n_attributes = n_attributes
        # Track non-classified by index
        indices_not_classified = np.arange(n_examples)

        # For each n_conditions = 1, ..., n_attributes
        for n_conditions in range(1, n_attributes + 1):
            # Generate all possible combinations of attributes (without repetition and without order)
            # of length n_conditions
            attribute_combinations_n = combinations(range(n_attributes), n_conditions)
            # For each combination of attributes (columns)
            for attribute_group in attribute_combinations_n:
                lists_of_values = []
                # Calculate the unique values of the chosen attributes from the non-classified instances,
                # and generate all combinations of selectors <attribute, value> given the chosen attributes
                # These combination of selectors form conditions
                for attribute in attribute_group:
                    lists_of_values.append(np.unique(x[indices_not_classified, attribute]))
                value_combinations = product(*lists_of_values)
                # For each condition <att1, val1>, <att2, val2>, ...
                for value_group in value_combinations:
                    # Find indices of ALL INSTANCES that match the condition
                    indices_match = np.where((x[:, list(attribute_group)] == value_group).all(axis=1))[0]
                    # Find indices of NON-CLASSIFIED INSTANCES that match the condition
                    indices_match_not_classified = \
                        np.where((x[np.ix_(indices_not_classified, attribute_group)] == value_group).all(axis=1))[0]
                    if len(indices_match) == 0:
                        # This condition is not present in the training set of examples
                        continue
                    if len(indices_match_not_classified) == 0:
                        # Although this condition is present in the set of examples,
                        # it does not match any non-classified instance
                        # It is the case of a condition that could end generating an IRRELEVANT RULE
                        continue
                    # Take the ground truth of the matched instances and look if they belong to a single class
                    classes = y[indices_match]
                    unique_classes = np.unique(classes)
                    if len(unique_classes) == 1:
                        # Generate the rule and add it to the set of rules
                        # The rule is encoded as the set of attributes to match, their values and the class
                        self.rules.append((attribute_group, value_group, unique_classes[0]))
                        # Remove the classified instances from the set of non-classified ones
                        indices_not_classified = np.setdiff1d(indices_not_classified, indices_match, assume_unique=True)
                        # If there aren't any more instances to classify, return
                        if len(indices_not_classified) == 0:
                            return
                    # If we are in the last iteration, we are checking for the full antecedent
                    # If there is more than a single class, we have a contradiction in the data
                    # Let's choose the most probable class (or random if tie)
                    elif n_conditions == n_attributes:
                        print("WARNING: There are contradictions in the training set")
                        unique_classes, counts = np.unique(classes, return_counts=True)
                        self.rules.append((attribute_group, value_group, unique_classes[np.argmax(counts)]))
                        # Remove the classified instances from the set of non-classified ones
                        indices_not_classified = np.setdiff1d(indices_not_classified, indices_match, assume_unique=True)
                        # If there aren't any more instances to classify, return
                        if len(indices_not_classified) == 0:
                            return

    # ######################### P R E D I C T #########################
    def predict(self, x):
        # Predict
        y_pred = self.__predict(x)
        # Predictions are integers, convert back to original values
        y_pred = np.vectorize(self.labels_dict[-1].get)(y_pred)
        return y_pred

    def score(self, x, y):
        y_pred = self.__predict(x)
        # Convert true class to integers (predictions already are)
        y = np.vectorize(self.preproc_dict[-1].get)(y)
        return accuracy_score(y, y_pred)

    def __predict(self, x):
        print('Predicting...')
        x = self.__preproc_test_data(x)
        y_pred = []
        # For each instance
        for instance in x:
            classified = False
            # For each rule
            for attributes, values, tag in self.rules:
                # Check if antecedent matches
                if np.array_equal(instance[list(attributes)], values):
                    y_pred.append(tag)
                    classified = True
                    break
            # No rule matched, we apply the default rule --> Most probable class
            if not classified:
                y_pred.append(self.most_probable_class)
        return np.array(y_pred)

    # ################ M E T R I C S   A N D   R U L E S ################
    def compute_metrics(self, x, y):
        # Use the already computed (training) bins and integer conversions
        x = self.__preproc_test_data(x)
        y = np.vectorize(self.preproc_dict[-1].get)(y)
        return self.__compute_metrics(x, y)

    def __compute_metrics(self, x, y):
        n_examples = x.shape[0]

        # Store (Coverage, Precision) for each rule
        metrics = []
        overall_coverage = []
        overall_precision = []
        for attributes, values, tag in self.rules:
            indices_match_condition = np.where((x[:, list(attributes)] == values).all(axis=1))[0]
            coverage = len(indices_match_condition) / n_examples
            indices_match_rule = np.where(y[indices_match_condition] == tag)[0]
            precision = len(indices_match_rule) / len(indices_match_condition)
            overall_precision.append(precision)
            overall_coverage.append(coverage)
            metrics.append((coverage, precision))
        # Add overall metrics
        metrics.append((sum(overall_coverage), sum(overall_precision) / len(overall_precision)))

        return metrics

    # ############### P R E P R O C E S S I N G ###############
    def __preproc_train_data(self, x, y):
        # Set attribute names for pretty printing
        if self.contains_header:
            names_x = x.columns.values.tolist()
            name_y = y.columns.values.tolist()
            self.attribute_names = names_x + name_y

        column_types = x.dtypes
        # Missing Values
        for attribute, dtype in zip(x, column_types):
            # We take the mean for floats
            if np.issubdtype(dtype, np.floating):
                x.loc[:, attribute].fillna(x[attribute].mean(), inplace=True)
            # We take the mode for categoricals (including integers)
            else:
                # Intermediate conversion into '?' (to join different representations for mv)
                x.loc[:, attribute].fillna('?', inplace=True)
                uniques, counts = np.unique(x[attribute], return_counts=True)
                mode = uniques[np.argmax(counts)]
                if mode == '?':
                    mode = uniques[np.argsort(counts)[-2]]
                x.loc[x[attribute] == '?', attribute] = mode

        # Discretize Numeric attributes
        # Store exact discretization bins for test data
        if self.number_bins != 0:
            to_discretize = np.number if self.discretize_ints else np.floating
            for attribute, dtype in zip(x, column_types):
                if np.issubdtype(dtype, to_discretize):
                    if self.discretize_mode == 'equal':
                        x[attribute], bins = pd.cut(x[attribute], bins=self.number_bins, retbins=True)
                        self.bins.append((attribute, bins, np.unique(x[attribute])))
                    elif self.discretize_mode == 'freq':
                        x[attribute], bins = pd.qcut(x[attribute], q=self.number_bins, retbins=True)
                        self.bins.append((attribute, bins, np.unique(x[attribute])))
                    else:
                        raise ValueError("Wrong discretize_mode")

        # Move everything to integer, so numpy works faster
        x = x.to_numpy()
        y = y.to_numpy()
        data = np.concatenate((x, y), axis=1)
        _, n_cols = data.shape

        # Store conversion from original values to integers for pretty printing
        inv_conversions = []
        conversions = []
        for i in range(n_cols):
            col = data[:, i]
            uniques = np.unique(col).tolist()
            d = defaultdict(lambda: -1, zip(uniques, range(len(uniques))))
            d_inv = dict(zip(range(len(uniques)), uniques))
            data[:, i] = np.vectorize(d.get)(col)
            conversions.append(d)
            inv_conversions.append(d_inv)

        # Preprocessing Dictionary
        # Contains all the conversions from original values to integers --> To be used when preprocessing test data
        self.preproc_dict = conversions
        # Labels Dictionary
        # Contains all the conversions from integers to original values --> To be used when pretty printing
        self.labels_dict = inv_conversions

        return data[:, :-1].astype(np.uint8), data[:, -1].astype(np.uint8)

    def __preproc_test_data(self, x):
        # Preprocess only attributes (not class)
        # Use the same exact steps than in training
        #   MV
        #   Discretization using same bins
        #   Conversion to integers using same mapping

        column_types = x.dtypes
        # Missing Values
        for attribute, dtype in zip(x, column_types):
            # We take the mean for floats
            if np.issubdtype(dtype, np.floating):
                x.loc[:, attribute].fillna(x[attribute].mean(), inplace=True)
            # We take the mode for categoricals (including integers)
            else:
                # Intermediate conversion into '?' (to join different representations for mv)
                x.loc[:, attribute].fillna('?', inplace=True)
                uniques, counts = np.unique(x[attribute], return_counts=True)
                mode = uniques[np.argmax(counts)]
                if mode == '?':
                    mode = uniques[np.argsort(counts)[-2]]
                x.loc[x[attribute] == '?', attribute] = mode

        # Discretize Numeric attributes using training bins
        for attribute, bins, labels in self.bins:
            if len(labels) + 1 == len(bins):
                x[attribute] = pd.cut(x[attribute], bins=bins, labels=labels)
            else:
                x[attribute] = pd.cut(x[attribute], bins=bins)

        # Move everything to integer, so numpy works faster
        data = x.to_numpy()
        _, n_cols = data.shape

        # Use original-integer training mapping
        for i in range(n_cols):
            col = data[:, i]
            data[:, i] = np.vectorize(self.__my_vectorized_mapping)(i, col)

        return data.astype(np.uint8)

    def __my_vectorized_mapping(self, i, x):
        return self.preproc_dict[i][x]