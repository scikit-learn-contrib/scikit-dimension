class EstimatorFailure(Exception):
    """An Exception class to be used when an estimator cannot return an output."""
    pass

class ConvergenceFailure(Exception):
    """An Exception class to be used when an estimator's algorithm does not converge. """
    pass