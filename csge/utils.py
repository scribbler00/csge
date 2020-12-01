import numpy as np


def soft_gating_formular(omega, eta):
    eps = 0.00000001
    denom_1 = np.add(np.power(omega, eta), eps)
    denom_2 = np.sum(1.0 / (np.power(omega, eta) + eps), axis=1)
    return 1.0 / (denom_1 * denom_2.reshape((-1, 1)))
