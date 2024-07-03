import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_robust_standard_errors(model):
    logger.info("Computing robust standard errors")
    return model.get_robustcov_results(cov_type='HC3')
