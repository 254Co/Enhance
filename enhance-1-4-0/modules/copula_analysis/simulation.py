# copula_analysis/simulation.py
import numpy as np
from scipy.stats import norm, t

class JointReturnSimulation:
    def __init__(self, copula):
        self.copula = copula
    
    def simulate(self, num_samples, marginals):
        u = self.copula.random(num_samples)
        simulated_data = []
        for i in range(u.shape[1]):
            if marginals[i]['type'] == 'normal':
                simulated_data.append(norm.ppf(u[:, i], *marginals[i]['params']))
            elif marginals[i]['type'] == 't':
                simulated_data.append(t.ppf(u[:, i], *marginals[i]['params']))
            else:
                raise ValueError("Unsupported marginal distribution type")
        return np.array(simulated_data).T
