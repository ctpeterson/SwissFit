# External imports
import gvar as gv
import numpy as np
from functools import partial

""" Create custom step taking function for basin hopping """
# Create custom step taking function
def create_take_step(alpha, fitter, scipy_basin_hopping, force_positivity = True):
    # Basin hopping step that forces positivity on parameters
    if force_positivity:
        positivity_enforcing_step = partial(
            scipy_basin_hopping.take_step_biased, # Take step forcing positivity in step
            indices = [ind for ind in range(len(fitter.p0['I']['c']))], # Indices of crit. prms.
            array_size = len(fitter.pmean), # Number of parameters
        )
    else: positivity_enforcing_step = partial(
            scipy_basin_hopping.take_step_biased, # Take step forcing positivity in step
            indices = [], # Indices of crit. prms.
            array_size = len(fitter.pmean), # Number of parameters
    )

    # Custom take_step function for SciPy's basin hopping
    def take_step(x):
        # Randomly sample alpha from [0, alpha]
        scipy_basin_hopping.take_step_biased.stepsize = np.random.uniform(0., alpha)
    
        # Return perturbed coordinates
        return positivity_enforcing_step(x)

    # Return step taking function
    return take_step

""" For grabbing data """

# Grab 2-state Potts (Ising) model data
def potts2_data(obs, Kl, Kh, vols, data_in_fit = True):
    """ Grab data for fit """
    # Temperature range
    Tl, Th = 1. / Kh, 1. / Kl

    # Create dictionary of data
    data = {'x': [], 'y': []}

    # Grab data
    for vol in vols:
        vl = float(vol)
        with open('./example_data/potts2/potts2_' + vol + '.dat', 'r') as in_file:
            for ln in in_file.readlines():
                # Get temperature and coupling K = J/T
                spln = ln.split(); T = float(spln[0]); K = 1. / T;

                # Grab data in fit, if requested
                if (Tl <= T <= Th) and (data_in_fit):
                    if obs == 'u': U = 0.5 * (3. - gv.gvar(float(spln[3]), float(spln[4])))
                    elif obs == 'sus': U = gv.gvar(float(spln[7]), float(spln[8]))
                    data['x'].append([K, vl])
                    data['y'].append(U)

                # Grab data not in fit, if requested
                elif ((Tl > T) or (T > Th)) and (not data_in_fit):
                    if obs == 'u': U = 0.5 * (3. - gv.gvar(float(spln[3]), float(spln[4])))
                    elif obs == 'sus': U = gv.gvar(float(spln[7]), float(spln[8]))
                    data['x'].append([K, vl])
                    data['y'].append(U)

    # Convert data to numpy array
    for key in data.keys(): data[key] = np.array(data[key])

    """ Starting parameters for empirical Bayes """
    if obs == 'u':
        starting_parameters = gv.load('example_data/potts2/potts2_u_starting_parameters.bin')['I']
    elif obs == 'sus':
        starting_parameters = gv.load('example_data/potts2/potts2_sus_starting_parameters.bin')['I']
    
    """ Return data """
    return data, starting_parameters

# Grab 3-state Potts data
def potts3_data(obs, Kl, Kh, vols, data_in_fit = True):
    """ Grab data for fit """
    # Temperature range
    Tl, Th = 1. / Kh, 1. / Kl

    # Create dictionary of data
    data = {'x': [], 'y': []}

    # Grab data
    for vol in vols:
        with open('./example_data/potts3/potts3_' + vol + '.dat', 'r') as in_file:
            for line in in_file.readlines():
                # Grab data at this line
                try: err, mean, T = [*map(float, [line.split()[-(index + 1)] for index in range(3)])]
                except IndexError: pass
                    
                # Save binder cumulant data
                if ('(Binder Ratio)' in line) and (obs == 'u'):
                    u, K = 0.5 * (3. - gv.gvar(mean, err)), 1. / T
                    if (Th >= T >= Tl) and (data_in_fit):
                        data['x'].append([K, np.log(float(vol))])
                        data['y'].append(u)
                    elif ((Th < T) or (T < Tl)) and (not data_in_fit):
                        data['x'].append([K, np.log(float(vol))])
                        data['y'].append(u)

                # Save connected susceptibility data
                if ('(Connected Susceptibility)' in line) and (obs == 'sus'):
                    chi, K = gv.gvar(mean, err), 1. / T
                    if (Th >= T >= Tl) and (data_in_fit):
                        data['x'].append([K, gv.log(float(vol))])
                        data['y'].append(gv.log(chi))
                    elif ((Th < T) or (T < Tl)) and (not data_in_fit):
                        data['x'].append([K, gv.log(float(vol))])
                        data['y'].append(gv.log(chi))
                        
    # Convert to numpy array
    for key in data.keys(): data[key] = np.array(data[key])
    
    """ Starting parameters for empirical Bayes """
    if obs == 'u':
        starting_parameters = gv.load('example_data/potts3/potts3_u_starting_parameters.bin')['I']
    elif obs == 'sus':
        starting_parameters = gv.load('example_data/potts3/potts3_sus_starting_parameters.bin')['I']
    
    """ Return data """
    return data, starting_parameters

# Grab 4-state clock data
def clock4_data(obs, Kl, Kh, vols, data_in_fit = True):
    """ Grab data for fit """
    # Temperature range
    Tl, Th = 1. / Kh, 1. / Kl

    # Create dictionary of data
    data = {'x': [], 'y': []}

    # Grab data
    for vol in vols:
        with open('./example_data/clock4/clock4_' + vol + '.dat', 'r') as in_file:
            for line in in_file.readlines():
                # Grab data at this line
                try: err, mean, T = [*map(float, [line.split()[-(index + 1)] for index in range(3)])]
                except IndexError: pass
                    
                # Save binder cumulant data
                if ('(Binder Ratio)' in line) and (obs == 'u'):
                    u, K = 0.5 * (3. - gv.gvar(mean, err)), 1. / T
                    if (Th >= T >= Tl) and (data_in_fit):
                        data['x'].append([K, float(vol)])
                        data['y'].append(u)
                    elif ((Th < T) or (T < Tl)) and (not data_in_fit):
                        data['x'].append([K, float(vol)])
                        data['y'].append(u)

                # Save connected susceptibility data
                if ('(Connected Susceptibility)' in line) and (obs == 'sus'):
                    chi, K = gv.gvar(mean, err), 1. / T
                    if (Th >= T >= Tl) and (data_in_fit):
                        data['x'].append([K, float(vol)])
                        data['y'].append(chi)
                    elif ((Th < T) or (T < Tl)) and (not data_in_fit):
                        data['x'].append([K, float(vol)])
                        data['y'].append(chi)
                        
    # Convert to numpy array
    for key in data.keys(): data[key] = np.array(data[key])
    
    """ Starting parameters for empirical Bayes """
    if obs == 'u':
        starting_parameters = gv.load('example_data/clock4/clock4_u_starting_parameters.bin')['I']
    elif obs == 'sus':
        starting_parameters = gv.load('example_data/clock4/clock4_sus_starting_parameters.bin')['I']
    
    """ Return data """
    return data, starting_parameters

# Grab infty-state clock data
def clockinf_data(obs, Kl, Khs, vols, data_in_fit = True):
    """ Grab data """
    # Create dictionary of data
    data = {'x': [], 'y': []}
    
    # Grab data
    for vol in vols:
        fln = './example_data/clockinf/clockinf_' + vol + '_' + obs + '.dat'
        with open(fln, 'r') as in_file:
            for line in in_file.readlines():
                K, mean, err = [*map(float, line.split())]
                O = gv.gvar(mean, err)
                if (Kl <= K <= Khs[vol]) and (data_in_fit):
                    data['x'].append([K, np.log(float(vol))])
                    data['y'].append(O if obs == 'u' else gv.log(O))
                elif ((Kl > K) or (K <= Khs[vol])) and (not data_in_fit):
                    data['x'].append([K, np.log(float(vol))])
                    data['y'].append(O if obs == 'u' else gv.log(O))

    # Convert to numpy array
    for key in data.keys(): data[key] = np.array(data[key])

    """ Get starting parameters for empirical Bayes """
    if obs == 'u':
        starting_parameters = gv.load(
            'example_data/clockinf/clockinf_u_starting_parameters.bin'
        )['I']
    elif obs == 'sus':
        starting_parameters = gv.load(
            'example_data/clockinf/clockinf_sus_starting_parameters.bin'
        )['I']
    
    """ Return data """
    return data, starting_parameters

""" Tools for getting clock4 interpolations & interpolation data """

def clock4_interp_data(vols, return_starting_parameters = False):
    # Get fit data
    data = {vol: {'x': [], 'y': []} for vol in vols}
    for vol in vols:
        with open('./example_data/clock4/clock4_' + vol + '.dat', 'r') as in_file:
            for line in in_file.readlines():
                if ('(Energy)' in line):
                    err, mean, T = [*map(float, [line.split()[-(index + 1)] for index in range(3)])]
                    data[vol]['x'].append([1. / T])
                    data[vol]['y'].append(gv.gvar(mean, err))
        for key in data[vol].keys(): data[vol][key] = np.array(data[vol][key])

    # Get starting parameters
    starting_parameters = {vol: None for vol in vols}
    if return_starting_parameters:
        for vol in vols:
            starting_parameters[vol] = gv.load(
                './example_data/clock4/clock4_energy_starting_parameters_' + vol + '.bin'
            )['I']

    # Return what is needed
    if return_starting_parameters:
        return data, starting_parameters
    else: return data

def clock4_interp(vols):
    interp_params = {}
    for vol in vols:
        interp_params[vol] = gv.load('./example_data/clock4/clock4_energy_' + vol + '.bin')['I']
    return interp_params

def clockinf_interp(vols):
    interp_params = {}
    for vol in vols:
        interp_params[vol] = gv.load(
            './example_data/clockinf/clockinf_helicity_' + vol + '.bin'
        )['I']
    return interp_params

def clockinf_interp_data(vols, return_starting_parameters = False):
    data = {vol: {'x': [], 'y': []} for vol in vols}
    for vol in vols:
        file_name = './example_data/clockinf/clockinf_helicity_' + vol + '.dat'
        with open(file_name, 'r') as in_file:
            for line in in_file.readlines():
                K, mean, err = [*map(float, line.split())]
                data[vol]['x'].append([K])
                data[vol]['y'].append(gv.gvar(mean, err))
        for key in data[vol].keys(): data[vol][key] = np.array(data[vol][key])
    return data
