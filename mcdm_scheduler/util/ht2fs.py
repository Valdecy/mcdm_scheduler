###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: HT2FS
def ht2fs_weight_calculation(crisp_inputs, uncertainty_ranges, criteria_importance):
    
    ################################################
    def fuzzify_input(crisp_value, uncertainty_range):
        a = crisp_value - 2 * uncertainty_range
        b = crisp_value - uncertainty_range
        c = crisp_value
        d = crisp_value
        e = crisp_value + uncertainty_range
        f = crisp_value + 2 * uncertainty_range
        return (a, b, c, d, e, f)

    def construct_fuzzy_decision_matrix(fuzzy_inputs, criteria_importance):
        n            = len(criteria_importance)
        fuzzy_matrix = np.zeros((n, n), dtype = object)
        for i in range(n):
            for j in range(n):
                if (i == j):
                    fuzzy_matrix[i][j] = fuzzify_input(1.0, 0.0)
                else:
                    fuzzy_matrix[i][j] = fuzzy_inputs[j]
        return fuzzy_matrix

    def apply_fuzzy_arithmetic(fuzzy_matrix):
        n                 = len(fuzzy_matrix)
        composite_weights = np.zeros(n, dtype = object)
        for i in range(n):
            composite_weight = fuzzy_matrix[i][0]
            for j in range(1, n):
                composite_weight = fuzzy_arithmetic_addition(composite_weight, fuzzy_matrix[i][j])
            composite_weights[i] = composite_weight
        
        return composite_weights

    def fuzzy_arithmetic_addition(fuzzy1, fuzzy2):
        return tuple(a + b for a, b in zip(fuzzy1, fuzzy2))

    def defuzzify_value(fuzzy_value):
        a, b, c, d, e, f = fuzzy_value
        numerator        = (a + 2*b + 3*c + 3*d + 2*e + f) / 12
        return numerator
    ################################################
    
    fuzzy_inputs          = [fuzzify_input(crisp, uncertainty) for crisp, uncertainty in zip(crisp_inputs, uncertainty_ranges)]
    fuzzy_decision_matrix = construct_fuzzy_decision_matrix(fuzzy_inputs, criteria_importance)
    composite_weights     = apply_fuzzy_arithmetic(fuzzy_decision_matrix)
    crisp_weights         = [defuzzify_value(fuzzy) for fuzzy in composite_weights]
    return crisp_weights

###############################################################################
