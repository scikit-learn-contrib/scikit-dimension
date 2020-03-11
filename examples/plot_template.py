"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skltemplate.template.TemplateEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt

X = np.arange(100).reshape(100, 1)
y = np.zeros((100, ))

plt.plot(y,y)
plt.show()
