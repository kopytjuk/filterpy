import numpy as np


def constvel2constacc(x: np.ndarray, dim: int) -> np.ndarray:

    num_vars_per_dim_constvel = 2
    num_vars_per_dim_constacc = 3
    
    num_state_attributes_constvel = num_vars_per_dim_constvel*dim
    num_state_attributes_constacc = num_vars_per_dim_constacc*dim

    is_cov_matrix = x.shape == (num_state_attributes_constvel, num_state_attributes_constvel)
    if is_cov_matrix:
        P = np.zeros((num_state_attributes_constacc, num_state_attributes_constacc))

        default_covariance = 100.0

        for i in range(dim):
            idx_input = i*num_vars_per_dim_constvel
            idx_output = i*num_vars_per_dim_constacc
            P[idx_output:(idx_output+num_vars_per_dim_constvel), idx_output:(idx_output+num_vars_per_dim_constvel)] = \
                x[idx_input:(idx_input+num_vars_per_dim_constvel), idx_input:(idx_input+num_vars_per_dim_constvel)]
            P[idx_output + num_vars_per_dim_constacc - 1, idx_output + num_vars_per_dim_constacc - 1] = default_covariance
        return P
    else:
        x = x.flatten()
        s = np.zeros(num_state_attributes_constacc, dtype=float)
        for i in range(dim):
            idx_input = i*num_vars_per_dim_constvel
            idx_output = i*num_vars_per_dim_constacc
            s[idx_output:(idx_output+num_vars_per_dim_constvel)] = x[idx_input:(idx_input+num_vars_per_dim_constvel)]
            s[idx_output + num_vars_per_dim_constacc - 1] = 0.0
        return s


def constacc2constvel(x: np.ndarray, dim: int) -> np.ndarray:

    num_vars_per_dim_constvel = 2
    num_vars_per_dim_constacc = 3
    
    num_state_attributes_constvel = num_vars_per_dim_constvel*dim
    num_state_attributes_constacc = num_vars_per_dim_constacc*dim

    is_cov_matrix = x.shape == (num_state_attributes_constacc, num_state_attributes_constacc)
    if is_cov_matrix:
        P = np.zeros((num_state_attributes_constvel, num_state_attributes_constvel))

        for i in range(dim):
            idx_input = i*num_vars_per_dim_constacc
            idx_output = i*num_vars_per_dim_constvel
            P[idx_output:(idx_output+num_vars_per_dim_constvel), idx_output:(idx_output+num_vars_per_dim_constvel)] = \
                x[idx_input:(idx_input+num_vars_per_dim_constvel), idx_input:(idx_input+num_vars_per_dim_constvel)]
        return P
    else:
        x = x.flatten()
        s = np.zeros(num_state_attributes_constvel, dtype=float)
        for i in range(dim):
            idx_input = i*num_vars_per_dim_constacc
            idx_output = i*num_vars_per_dim_constvel
            s[idx_output:(idx_output+num_vars_per_dim_constvel)] = x[idx_input:(idx_input+num_vars_per_dim_constvel)]
        return s

