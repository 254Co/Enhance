# copula_analysis/marginal_analysis.py
from pyspark.sql import DataFrame
from scipy.stats import norm, t, kstest

class MarginalDistribution:
    def __init__(self, data: DataFrame):
        self.data = data
    
    def choose_distribution(self, column, distribution_type):
        self.column = column
        self.data_collected = self.data.select(column).rdd.flatMap(lambda x: x).collect()
        
        if distribution_type == 'normal':
            self.dist = norm
        elif distribution_type == 't':
            self.dist = t
        else:
            raise ValueError("Unsupported distribution type")
    
    def estimate_parameters(self):
        self.params = self.dist.fit(self.data_collected)
        return self.params
    
    def goodness_of_fit(self):
        ks_stat, ks_p_value = kstest(self.data_collected, self.dist.cdf, self.params)
        return ks_stat, ks_p_value
