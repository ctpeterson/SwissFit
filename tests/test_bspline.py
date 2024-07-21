from scipy.interpolate.BSpline import basis_element
from swissfit.machine_learning import kolmogrov_arnold

if __name__ == '__main__':
    xknots = list(range(10))
    scipy_bspline = basis_element(xknots)
    swissfit_bspline = BSpline(xknots, len(xknots) - 2)
    print(scipy_bspline(5.), swissfit_bspline(5.))
