import matplotlib.pyplot as plt
from numpy import array, exp
from scipy.optimize import curve_fit

x = array([1, 2, 3, 4])
y = array([1, 2, 3, 4])
y_err = [0.3, 0.3, 0.3, 0.3]

y_1 = (1.15 * x - 0.4)
y_2 = (0.85 * x + 0.4)

plt.scatter(x, y)
plt.errorbar(x, y, yerr=y_err, ls="none")
plt.plot(x, y_1)
plt.plot(x, y_2)
plt.show()
