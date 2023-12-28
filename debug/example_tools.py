import gvar as gv
import numpy as np

""" Dumping data """
def dump_xy_data():
    vols = ['128', '160', '256', '320', '512']
    Thigh = 0.94
    Tlows = {'512': 0.905, '320': 0.9075, '256': 0.9075, '160': 0.91, '128': 0.911}
    with open('./xy_data/sus.txt', 'w+') as out_file:
        with open('./xy_data/sus_data.txt', 'r') as in_file:
            for line in in_file.readlines():
                l, beta, mean, err = list(map(float, line.split()))
                l_str = str(int(round(l)))
                if (str(int(round(l))) in vols):
                    if (Tlows[l_str] <= 1. / beta <= Thigh):
                        O = beta * gv.gvar(mean, err) / l**2.
                        out_file.write(' '.join([
                            l_str, str(beta),
                            str(O.mean), str(O.sdev)
                        ]) + '\n')

""" Grab data """
def get_xy_data():
    vols = ['128', '160', '256', '320', '512']
    data = {'x': [], 'y': []}
    Thigh = 0.94
    Tlows = {'512': 0.905, '320': 0.9075, '256': 0.9075, '160': 0.91, '128': 0.911}
    with open('./xy_data/sus.txt', 'r') as in_file:
        for line in in_file.readlines():
            l, beta, mean, err = list(map(float, line.split()))
            l_str = str(int(round(l)))
            if (str(int(round(l))) in vols):
                if (Tlows[l_str] <= 1. / beta <= Thigh):
                    data['x'].append([1. / beta, l])
                    data['y'].append(gv.gvar(mean / l**2., err / l**2.))
    return data

def get_nf8_data():
    """ Set things up """
    path = './nf8_data/'
    vols = ['8', '10', '12', '16', '20', '24', '32']
    c1 = '0.45'
    data = {'x': [], 'y': []}
    min_g2 = 36.; max_g2 = 141.; # 50
    blt_fac = 1. # Bloat all errors by 1.5x to account for underestimated autocorrelation time
    xtra_blt = 1. # Increase error on points that have way underestimated error
    #err_blt = {'16': 1., '20': 1., '24': blt_fac, '32': 1.}
    min_max_g2 = {
        '8':  [40., 75.],
        '10': [40., 75.],
        '12': [40., 100.],
        '16': [min_g2, max_g2], '20': [29., max_g2],
        '24': [min_g2, max_g2], '32': [min_g2, max_g2]}
    L0 = 32.
    blt_fac = 3.

    """ Grab data """
    # Get data
    for vol in vols:
        with open(path + 'g2' + '_' + vol + vol + '_' + c1, 'r') as in_file:
            for line in in_file.readlines():
                spln = line.split()
                cp = round(float(spln[0]), 2)
                vl = float(vol)
                vli = str(int(round(vl)))
                g2 = gv.gvar(float(spln[1]) / L0, blt_fac * float(spln[2]) / L0)
                mng2, mxg2 = min_max_g2[vli][0] / L0, min_max_g2[vli][-1] / L0
                if mng2 <= gv.mean(g2) <= mxg2:
                    data['x'].append([1. / cp, vl])
                    data['y'].append(g2)

    """ Return data """
    return data
    
def get_data(obs):
    """ Grab data """
    Th = 2.325; Tl = 2.22;
    vols = ['48', '64', '96', '128']
    Ths = {'48': Th, '64': Th, '96': 1. / 0.431, '128': 1. / 0.433}

    data = {'x': [], 'y': []}
    for vol in vols:
        vl = float(vol)
        with open('./ising_data/wolff_' + vol + '.dat', 'r') as in_file:
            for ln in in_file.readlines():
                spln = ln.split(); T = float(spln[0]); 
                if Tl <= T <= Ths[vol]:
                    beta = 1. / T;
                    if obs == 'u':
                        Om = float(spln[3]) ; Oe = float(spln[4]);
                        O = 0.5 * (3. - gv.gvar(Om, Oe))
                        U = gv.gvar(O.mean, O.sdev)
                    elif obs == 'sus':
                        Om = float(spln[7]) ; Oe = float(spln[8]);
                        U = gv.gvar(Om / vl, Oe / vl)
                    data['x'].append([beta, vl])
                    data['y'].append(U)
    for key in data.keys(): data[key] = np.array(data[key])

    # Return observable choice & data
    return data

def get_all_ising_data(obs):
    """ Grab data """
    Th = 2.325; Tl = 2.22;
    vols = ['48', '64', '96', '128']
    Ths = {'48': Th, '64': Th, '96': 1. / 0.431, '128': 1. / 0.433}

    data = {'x': [], 'y': []}
    for vol in vols:
        vl = float(vol)
        with open('./ising_data/wolff_' + vol + '.dat', 'r') as in_file:
            for ln in in_file.readlines():
                spln = ln.split(); T = float(spln[0]);
                if (T > Ths[vol]) or (T < Tl):
                    beta = 1. / T;
                    if obs == 'u':
                        Om = float(spln[3]) ; Oe = float(spln[4]);
                        O = 0.5 * (3. - gv.gvar(Om, Oe))
                        U = gv.gvar(O.mean, O.sdev)
                    elif obs == 'sus':
                        Om = float(spln[7]) ; Oe = float(spln[8]);
                        U = gv.gvar(Om / vl, Oe / vl)
                    data['x'].append([beta, vl])
                    data['y'].append(U)
    for key in data.keys(): data[key] = np.array(data[key])

    # Return observable choice & data
    return data

""" Create priors """
def create_prior_and_p0_xy(topo, data, nn, seed, scling = None):
    import numpy.random as rand
    rand.seed(seed)
    
    """ Setup """
    prs = {}; p0 = {}; prH = {}; p0H = {};

    """ Set up priors on physical parameters & their starting values """
    if scling is None: prs['c'] = [
            gv.gvar('1.0(1.0)'),
            1. / gv.gvar('0.9(0.1)')#,
            #gv.gvar('0.5(0.5)')
    ]
    elif scling == '2nd': prs['c'] = [1. / gv.gvar('0.9(0.2)'), gv.gvar('2.0(2.0)')]
    prs['eta'] = [gv.gvar('0.5(0.5)')]
    for key in prs.keys(): p0[key] = gv.mean(prs[key])

    """ Neural network weight initialization """
    for lyri, lyr in enumerate(topo.keys()):
        # Initialize entries of weight initialization
        p0[lyr + '.weight'], p0[lyr + '.bias'] = [], []
        prs[lyr + '.weight'] = []

        # Set priors on weights according to https://arxiv.org/pdf/1704.08863.pdf
        x0 = [gv.gvar('0(0)')]
        activation = nn.activation[topo[lyr]['act']](x0)[0]
        act_term = gv.mean(1. + activation**2.)
        dact_term = gv.mean(activation.deriv(x0))[0]**2.
        output_variance = 1.
        sig = np.sqrt(output_variance / (topo[lyr]['in'] * act_term * dact_term))
        
        # Set weights initialization/priors & bias initialization
        for wi, w in enumerate(range(topo[lyr]['in'] * topo[lyr]['out'])):
            prs[lyr + '.weight'].append(gv.gvar(0., sig))
            p0[lyr + '.weight'].append(rand.normal(0., sig)) #gv.sample(gv.gvar(0., sig)))
        for b in range(topo[lyr]['out']): p0[lyr + '.bias'].append(0.)
    return {'I': prs}, {'I': p0}

def create_prior_and_p0_nf8(obs, topo, data, nn, seed):
    import numpy.random as rand
    rand.seed(seed)
    
    """ Setup """
    prs = {}; p0 = {}; prH = {}; p0H = {};

    """ Set up priors on physical parameters & their starting values """
    if obs == '2nd': prs['c'] = [gv.gvar('8.7(1.0)'), gv.gvar('1.0(1.0)')]
    elif obs == 'bkt': prs['c'] = [gv.gvar('0.5(0.5)'), gv.gvar('8.7(1.0)')]
    for key in prs.keys(): p0[key] = gv.mean(prs[key])

    """ Neural network weight initialization """
    for lyri, lyr in enumerate(topo.keys()):
        # Initialize entries of weight initialization
        p0[lyr + '.weight'], p0[lyr + '.bias'] = [], []
        prs[lyr + '.weight'] = []

        # Set priors on weights according to https://arxiv.org/pdf/1704.08863.pdf
        x0 = [gv.gvar('0(0)')]
        activation = nn.activation[topo[lyr]['act']](x0)[0]
        act_term = gv.mean(1. + activation**2.)
        dact_term = gv.mean(activation.deriv(x0))[0]**2.
        #if topo[lyr]['act'] == 'linear': output_variance = np.var(gv.mean(data['y']), ddof = 1)
        #else: output_variance = 1.
        output_variance = 1.
        sig = np.sqrt(output_variance / (topo[lyr]['in'] * act_term * dact_term))
        
        # Set weights initialization/priors & bias initialization
        for wi, w in enumerate(range(topo[lyr]['in'] * topo[lyr]['out'])):
            prs[lyr + '.weight'].append(gv.gvar(0., sig))
            p0[lyr + '.weight'].append(rand.normal(0., sig)) #gv.sample(gv.gvar(0., sig)))
        for b in range(topo[lyr]['out']): p0[lyr + '.bias'].append(0.)
    return {'I': prs}, {'I': p0}

def create_prior_and_p0(obs, topo, data, nn, seed):
    import numpy.random as rand
    rand.seed(seed)
    
    """ Setup """
    prs = {}; p0 = {}; prH = {}; p0H = {};

    """ Set up priors on physical parameters & their starting values """
    prs['c'] = [gv.gvar('2.0(2.0)'), gv.gvar('1.0(1.0)')]
    if obs == 'sus': prs['eta'] = [gv.gvar('1.0(1.0)')]
    for key in prs.keys(): p0[key] = gv.mean(prs[key])

    """ Neural network weight initialization """
    for lyri, lyr in enumerate(topo.keys()):
        # Initialize entries of weight initialization
        p0[lyr + '.weight'], p0[lyr + '.bias'] = [], []
        prs[lyr + '.weight'] = []

        # Set priors on weights according to https://arxiv.org/pdf/1704.08863.pdf
        x0 = [gv.gvar('0(0)')]
        activation = nn.activation[topo[lyr]['act']](x0)[0]
        act_term = gv.mean(1. + activation**2.)
        dact_term = gv.mean(activation.deriv(x0))[0]**2.
        #if topo[lyr]['act'] == 'linear': output_variance = np.var(gv.mean(data['y']), ddof = 1)
        #else: output_variance = 1.
        output_variance = 1.
        sig = np.sqrt(output_variance / (topo[lyr]['in'] * act_term * dact_term))
        
        # Set weights initialization/priors & bias initialization
        for wi, w in enumerate(range(topo[lyr]['in'] * topo[lyr]['out'])):
            prs[lyr + '.weight'].append(gv.gvar(0., sig))
            p0[lyr + '.weight'].append(rand.normal(0., sig)) #gv.sample(gv.gvar(0., sig)))
        for b in range(topo[lyr]['out']): p0[lyr + '.bias'].append(0.)
    return {'I': prs}, {'I': p0}
