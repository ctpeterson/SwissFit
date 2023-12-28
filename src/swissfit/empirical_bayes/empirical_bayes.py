# Mock "pool" class that makes abstracting pool in code easier
class MockSerialPool(object):
    def __init__(self): return None
    def map(self, fcn, x): return list(map(fcn, x))

# Parent empirical Bayes class
class EmpiricalBayes(object):
    def __init__(self, fcn = None, arguments = None, pool = None):
        self._fcn, self._args = fcn, arguments
        self._pool = MockSerialPool() if pool is None else pool
