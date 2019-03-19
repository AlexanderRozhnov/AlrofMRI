import numpy as np


def compute_kendall_tau_corr_coeff(control_subjects, treated_subjects):
    """
    This function computes Kendall-Tau correlation coefficient between features and target.
    Description of Kendall-Tau correlation coefficient can be found in Wikipedia: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

    Complexity of the following implementation is O(n^2).
    Parameters

    ---------------

    :param control_subjects: numpy.ndarray of shape (n_control_subjects, n_features)
        Matrix containing features for control subjects, i.e. for label 1
    :param treated_subjects: numpy.ndarray of shape (n_treated_subjects, n_features)
        Matrix containing features for control subjects, i.e. for label 1
    :return: numpy.ndarray of shape (n_features,)
        Array of correlation coefficients
    """
    assert control_subjects.shape[1] == treated_subjects.shape[1], \
        "Control and treated group should have the same dimension 1 size. You provided {} and {}".format(control_subjects.shape[1],
                                                                                                         treated_subjects.shape[1])

    kendall_tau_corr_coefficients = np.zeros(control_subjects.shape[1], )
    for i in range(control_subjects.shape[1]):
        control_features = control_subjects[:, i]
        ill_features = treated_subjects[:, i]

        control_matrix = np.repeat(control_features[:, None], ill_features.shape[0], axis=1)
        ill_matrix = np.repeat(ill_features[None, :], control_features.shape[0], axis=0)

        pairs_diff = ill_matrix - control_matrix
        kendall_tau_corr_coefficients[i] = (np.argwhere(pairs_diff > 0).shape[0] - np.argwhere(pairs_diff < 0).shape[0]) / pairs_diff.size

    return kendall_tau_corr_coefficients
