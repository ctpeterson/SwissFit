# External imports
import gvar as gv
import numpy as np

# Grab Ising model data
def ising_data():
    # Data specifications
    Th, Tl = 2.3, 2.22
    vols = ['64', '96', '128', '256']

    # Get data
    data = {'x': [], 'y': []}
    for vol in vols:
        with open('./example_data/wolff_' + vol + '.dat', 'r') as in_file:
            for ln in in_file.readlines():
                spln = ln.split(); T = float(spln[0]); 
                if (Tl <= T <= Th):
                    beta = 1. / T; Om = float(spln[3]) ; Oe = float(spln[4]);
                    data['x'].append([beta, float(vol)])
                    data['y'].append(0.5 * (3. - gv.gvar(Om, Oe)))
    for key in data.keys(): data[key] = np.array(data[key])
    
    # Return observable choice & data
    return data

# Get Ising model priors
def ising_prior(neural_net, lmbda):
    # p0 & prior
    prior = {}

    # Critical parameters
    prior['c'] = [gv.gvar('2.0(2.0)'), gv.gvar('1.0(1.0)')]

    # Network priors
    prior = neural_net.network_priors(
        prior_choice_weight = {
            lyr: {
                'prior_type': 'ridge_regression',
                'mean': 0.,
                'standard_deviation': lmbda
            } for lyr in neural_net.topo.keys()
            if neural_net.topo[lyr]['activation'] == 'linear'
        }, prior = prior
    )

    # Return priors
    return prior
