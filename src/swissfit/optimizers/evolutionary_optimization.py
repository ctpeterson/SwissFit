from scipy.optimize import OptimizeResult as _OptimizeResult # For mocking SciPy OptimizeResult
from functools import partial as _partial # For partial evaluation of functions
import numpy as _numpy # Vectorized number crunching
import scipy as _scipy # For k-means clustering
from .optimizer import Optimizer as _Optimizer # Optimizer parent class
from .scipy_minimize import LocalOptimizer as _LocalOptimizer # Stock local optimizer

""" Stock global variables for neuroevolution """
# For generating first generation members & mutations
_random_factor = 0.5 # Random perturbation factor

# Crossover probability (80%)
_crossover_probability = 0.8

# Mutation probability (5%)
_mutation_probability = 0.05

""" 
Stock evolutionary algorithm code:

Evolutionary algorithms are very problem-specific. When they can be
tailored to a specific type of problem, they can be very powerful.
However, basic evolutionary algorithms, like the evolutionary algorithm
defined by the default behavior of _neuroevolution, should not be expected
to work out of the box for most problems. Therefore, if you are wanting
to use an evolutionary algorithm like the one below, I highly recommend you
supply your own functions for parent selection, crossover, mutation and 
whatever else best suits the problem that you want to apply an 
evolutionary algorithm to. For more advanced options, I highly recommend 
checking out the PyGAD library, which supports tons of variations on
evolutionary algorithms to tackle a variety of difficult optimization problems.
"""
# For creating first generation
def _first_generation(pool, rng, p0, population_number, fcn):
    global _random_factor
    return pool.map(
        lambda member_id: [p0v + rng.uniform(-1., 1.) * _random_factor for p0v in p0],
        list(range(population_number))
    )

# Default behavior is to perform no local optimization
def _local_optimization(pool, rng, population, fcn): return population

# Default clustering is no clustering
def _clustering(pool, rng, population, fcn): return None

# Default crossover is 80% exchange of most fit parent's genes
def _crossover(pool, rng, child, parents, fcn):
    global _crossover_probability
    fitness = [fcn(parents[0]), fcn(parents[-1])]
    return [
        [parents[fitness.argsort()[0]][gene_ind] if rng.uniform(0., 1.) < _crossover_probability
         else parents[fitness.argsort()[-1]][gene_ind]
         for gene_ind in range(len(parents[0]))]
    ] # Returns one child/parent - keeps 90% of most fit parent's genes

# Default mutation applies random perturbation to 5% of genes
def _mutation(pool, child, rng, children, fcn):
    global _mutation_probability, _random_factor
    return [
        [child_gene if rng.uniform(0., 1.) > _mutation_probability
         else child_gene + rng.uniform(-1., 1.) * _random_factor
         for child_gene in child] for child in children
    ]

# Default population selection replaces old population with new population
def _population_selection(pool, rng, population, new_population, fcn): return new_population

""" Main neuroevolution code """
# Neuroevolution
def _evolutionary_optimization(p0,
                               fcn,
                               pool,
                               niter = 100,
                               population_number = 100,
                               num_parents = 50,
                               mating_loop_size = 100,
                               local_optimization = None,
                               clustering = None,
                               first_generation = None,
                               parent_selection = None,
                               crossover = None,
                               mutation = None,
                               population_selection = None,
                               seed = 12345,
                               other_args = {
                                   'first_generation': {},
                                   'local_optimization': {},
                                   'clustering': {},
                                   'parent_selection': {},
                                   'crossover': {},
                                   'mutation': {},
                                   'population_selection': {}
                               },
                               callback = None,
                               ):
    # Seed & save random number generator
    _numpy.random.seed(seed); rng = _numpy.random;
    
    # Create first generation
    if first_generation is None: population = _numpy.array(_first_generation(
            pool, rng, p0,
            population_number,
            fcn, **other_args['first_generation']
    ))
    else: population = _numpy.array(first_generation(
            pool, rng, p0,
            population_number,
            fcn
    ))
    
    # Do first local optimization step
    if local_optimization is None: population = _local_optimization(pool, rng, population, fcn)
    else: population = local_optimization(
            pool, rng,
            population,
            fcn, **other_args['local_optimization'])

    # Order population by chi^2
    chi2 = _numpy.array(list(map(fcn, population)))
    population = _numpy.array(population)[chi2.argsort(), :]
    chi2 = chi2[chi2.argsort()]
    
    # Run cluster algorithm & classify
    if clustering is None: labels = _clustering(pool, rng, population, fcn)
    else: labels = clustering(
            pool, rng,
            population,
            fcn, **other_args['clustering'])

    # Callback
    if callback and callback(-1, population[0], chi2): pass
    
    # Run through iterations of evolutionary algorithm
    for itn in range(niter):
        # Mate selection & mating loop
        new_population = []
        for child in range(mating_loop_size):
            # Select parents
            if parent_selection is None:
                # Select random parent
                parent_1 = population[rng.randint(0, population_number // 4)]
                
                # Select random partner
                parent_2 = population[rng.randint(0, population_number // 4)]

                # Save parents
                parents = [parent_1, parent_2]
            else: parents = parent_selection(
                    pool, rng,
                    child, population,
                    labels, fcn,
                    **other_args['parent_selection']
            )
                
            # Crossover
            if crossover is None: children = _crossover(pool, rng, child, parents, fcn)
            else: children = crossover(
                    pool, rng, child, parents,
                    fcn, **other_args['crossover']
            )

            # Mutation
            if mutation is None: children = _mutation(pool, rng, child, children, fcn)
            else: children = mutation(
                    pool, rng, child, children,
                    fcn, **other_args['mutation']
            )

            # Add children to new population
            for new_child in children: new_population.append(new_child)
            
        # Run local optimization
        if local_optimization is None: new_population = _local_optimization(
                pool, rng, new_population, fcn
        )
        else: new_population = local_optimization(
                pool, rng, new_population,
                fcn, **other_args['local_optimization']
        )
        
        # Select most fit members of the population
        if population_selection is None: population = _population_selection(
                pool, rng, population, new_population, fcn
        )
        else: population = population_selection(
                pool, rng,
                population, new_population,
                fcn, **other_args['population_selection']
        )

        # Order population by chi^2
        chi2 = _numpy.array(list(map(fcn, population)))
        population = _numpy.array(population)[chi2.argsort(), :]
        chi2 = chi2[chi2.argsort()]
        
        # Classify members of population
        if clustering is None: labels = _clustering(pool, rng, population, fcn)
        else: labels = clustering(
                pool, rng, population,
                fcn, **other_args['clustering']
        )
        
        # Callback
        if callback and callback(itn, population[0], chi2): break

""" Neuroevolution class """
class EvolutionaryOptimization(_Optimizer):
    def __init__(self,
                 fcn = None,
                 optimizer_arguments = {},
                 pool = None,
                 ):
        super().__init__(
            fcn = fcn, pool = pool,
            optimizer_arguments = optimizer_arguments
        )
    def __call__(self, p0):
        return _evolutionary_optimization(p0, self._fcn, self._pool, **self._args)
