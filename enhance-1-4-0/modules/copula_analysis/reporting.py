# copula_analysis/reporting.py
import pandas as pd

def generate_report(simulated_data, var, cvar):
    report = {
        "Simulated Data": simulated_data,
        "Value at Risk": var,
        "Conditional Value at Risk": cvar
    }
    return pd.DataFrame(report)
