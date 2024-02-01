from setuptools import setup, find_packages

VERSION = '0.0.0'
DESCRIPTION = 'SwissFit'
LONG_DESCRIPTION = 'SwissFit: A cheesy multitool for complicated least squares fitting'

setup(
    name = 'swissfit',
    package_dir = {'':'src/'},
    packages = [
        'swissfit',
        'swissfit/numerical_tools',
        'swissfit/machine_learning',
        'swissfit/optimizers',
        'swissfit/other_tools',
        'swissfit/empirical_bayes'
    ], 
    version = VERSION,
    
    author = "Curtis Taylor Peterson",
    author_email = "curtistaylorpetersonwork@gmail.com",

    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,

    keywords = ['least squares'],
    classifiers = ['Programming Language :: Python :: 3']
)
