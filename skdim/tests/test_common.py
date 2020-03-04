import pytest

from sklearn.utils.estimator_checks import check_estimator

from skltemplate import TemplateEstimator


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
