""" SwissFit imports """
from swissfit import fit # SwissFit fitter
from swissfit.optimizers import scipy_least_squares # SciPy's trust region reflective
from swissfit.optimizers import scipy_basin_hopping
from swissfit.interpolators import bspline # SwissFit-native b-spline

""" Other imports """
import gvar as gvar # Peter Lepage's GVar library
import numpy as np # NumPy
import example_tools
from scipy.interpolate import BSpline

if __name__ == '__main__':
   # Set interpolating function up
   degree = 3
   spline = bspline.BSpline(degree=degree)

   i = 0
   def fit_fcn(b,l,p):
        global spline,i
        xr = (b * p['c'][0] - 1.) * l**p['c'][1] # (b/bc-1)*l**1/nu
        spline.set_xknots(xr) # Set xknots for smoothing spline
        return spline(xr,p['p']) # Return result of smoothing spline
    
   # Get Ising model data
   Kl, Kh = 1. / 2.3, 1. / 2.22 # K = J/T, where T is the standard Ising temp.
   volumes = ['64', '96', '128', '256'] # Ns values
   data, starting_parameters = example_tools.potts2_data('u', Kl, Kh, volumes)  

   # Set up priors
   prior = {}
   prior['c'] = [gvar.gvar('2.0(2.0)'), gvar.gvar('1.0(1.0)')]

   # Create SwissFit object
   fitter = fit.SwissFit(
        udata = data,
        prior = prior,
        p0 = {
            'p': [0. for _ in range(len(data['x'])-degree-1)],
            'c': gvar.mean(starting_parameters['c'])
            },
        fit_fcn = fit_fcn,
   )

   # Create optimizer object
   local_optimizer = scipy_least_squares.SciPyLeastSquares()
   optimizer = scipy_basin_hopping.BasinHopping(
        optimizer_arguments = {'disp':True},
        local_estimator = local_optimizer
   )

   # Do fit!
   fitter(optimizer)

   # Print result & get parameters
   print(fitter)
   p = fitter.p