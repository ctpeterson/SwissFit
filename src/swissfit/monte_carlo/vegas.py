import vegas as _vegas
from .montecarlo import MonteCarlo as _MonteCarlo

class VegasLepage(_MonteCarlo):
    def __init__(self, pdf_arguments = {}, adaptation_arguments = {}, monte_carlo_arguments = {}):
        super().__init__(monte_carlo_arguments = monte_carlo_arguments)
        self.tag = 'vegas_peter_lepage'
        self._pdf_arguments = pdf_arguments
        self._adaptation_arguments = adaptation_arguments

        if 'adapt' not in self._adaptation_arguments.keys():
            self._adaptation_arguments['adapt'] = True
        if 'nitn' not in self._adaptation_arguments.keys():
            self._adaptation_arguments['nitn'] = 10

        if 'adapt' not in self._args.keys():
            self._args['adapt'] = False
        if 'nitn' not in self._args.keys():
            self._args['nitn'] = 10
        
    def __call__(self, p, pdf):
        integrator = _vegas.PDFIntegrator(param = p, pdf = pdf, **self._pdf_arguments)
        integrator(**self._adaptation_arguments)
        self.p = integrator.stats(f = None, **self._args)
        return integrator
        
    def __str__(self):
        out = 3 * ' ' + "algorithm = Peter Lepage's Vegas++" + '\n'
        for key, item in self._pdf_arguments.items():
            out += 3 * ' ' + key + ' = ' + str(item) + '\n'
        for key, item in self._adaptation_arguments.items():
            if key != 'adapt': out += 3 * ' ' + key + ' = ' + str(item) + ' (adapt) \n'
        for key, item in self._args.items():
            if key != 'adapt': out += 3 * ' ' + key + ' = ' + str(item) + ' (MCMC) \n'
        return out
