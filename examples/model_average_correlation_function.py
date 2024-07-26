from swissfit import fit as fitter
from swissfit.optimizers import scipy_least_squares
from swissfit.model_averaging import model_averaging
import gvar as gv
import numpy as np
from functools import partial

def model(nt,p):
    return p['a0'][0]*gv.exp(-p['e0']*nt)

def create_synthetic_data(
        a0=2.0,a1=10.4,
        e0=0.8,e1=1.16,
        NT=32,
        rho=0.6,
        var=0.09,
        nsamp=1000
    ): 
    # arXiv:2008.01069 
    # jwsitison/improved_model_avg_paper/improved_model_averaging/synth_data.py
    def F(nt): return a0*np.exp(-e0*nt)+a1*np.exp(-e1*nt)
    def corr_fcn(nt,ntp): return rho**np.abs(nt-ntp)

    Fnt = np.fromfunction(F,(NT,))
    corr = np.fromfunction(corr_fcn, (NT,NT))
    eta = gv.raniter(gv.correlate([gv.gvar(0.,np.sqrt(var))]*NT,corr))
    
    dataset = [Fnt*(1.+next(eta)) for _ in range(nsamp)]
    return [*range(NT)], gv.dataset.avg_data(dataset)

if __name__ == '__main__':
    tmin_max = 28
    nsamples = 1000
    nexperiments = 5
    a0,e0 = 2.0,0.8

    optimizer = scipy_least_squares.SciPyLeastSquares()
    p0 = {'a0': [0.1], 'e0': [0.1]}

    for experiment in range(nexperiments):
        gv.ranseed(int('98765432'+str(experiment)))
        xdata, ydata = create_synthetic_data(a0=a0,e0=e0,nsamp=nsamples)

        fits = [
            fitter.SwissFit(
                data = {'x': xdata[nt:], 'y': ydata[nt:]},
                p0 = p0,
                fit_fcn = model
            )(optimizer)
            for nt in range(0,tmin_max+1)
        ]
        
        model_average = model_averaging.BayesianModelAveraging(
            models=fits, ydata=ydata
        )
        parameters = model_average.p
        sfa0,sfe0 = parameters['a0'][0],parameters['e0'][0]

        print('experiment ',experiment+1)
        print('a0:',sfa0)
        print('e0:',sfe0)
        print('are the parameters correlated? ',not gv.uncorrelated(sfa0,sfe0),'\n')

