from swissfit import fit as fitter
from swissfit.optimizers import scipy_least_squares
from swissfit.model_averaging import model_averaging
import gvar as gv
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

plt.rcParams['text.usetex'] = True
plt.rcParams["mathtext.default"] = "regular"

def model(nt,p):
    return p['a0'][0]*gv.exp(-p['e0']*nt)

def weight(fit,NT):
    ncut = NT - len(fit.data['y'])
    aic = fit.chi2 + 2.*ncut
    return np.exp(-0.5*aic)

def weights(fits,NT):
    w = [*map(partial(weight,NT=NT),fits)]
    return np.array(w)/sum(w)

def model_average_boot(o,w):
    mean = sum(gv.mean(o)*w)
    err = sum((gv.sdev(o)**2.+gv.mean(o)**2.)*w) - mean**2.
    return gv.gvar(mean,np.sqrt(err))

def get_model_average(fits,NT):
    w = weights(fits,NT)
    ps = [fit.p for fit in fits]
    a0s = [p['a0'][0] for p in ps]
    e0s = [p['e0'][0] for p in ps]
    return model_average_boot(a0s,w),model_average_boot(e0s,w)

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

        tca0,tce0 = get_model_average(fits,len(ydata))
        sfa0,sfe0 = parameters['a0'][0],parameters['e0'][0]

        print('experiment ',experiment+1)
        print('a0 from this code vs. SwissFit:',tca0,sfa0)
        print('e0 from this code vs. SwissFit:',tce0,sfe0)
        print('are the parameters correlated? ',not gv.uncorrelated(sfa0,sfe0),'\n')

