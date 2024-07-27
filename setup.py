from setuptools import setup, find_packages
import os as os

VERSION = '0.22'
DESCRIPTION = 'SwissFit'
LONG_DESCRIPTION = 'SwissFit: a cheesy multitool for fitting models to data'

# Dependencies (https://stackoverflow.com/questions/26900328/install-dependencies-from-setup-py)
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = [] 
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

# SwissFit setup
setup(
    name = 'swissfit',
    package_dir = {'': 'src/'},
    packages = [
        'swissfit',
        'swissfit/numerical_tools',
        'swissfit/machine_learning',
        'swissfit/optimizers',
        'swissfit/other_tools',
        'swissfit/empirical_bayes',
        'swissfit/model_averaging',
        'swissfit/monte_carlo'
    ], 
    version = VERSION,
    
    author = "Curtis Taylor Peterson",
    author_email = "curtistaylorpetersonwork@gmail.com",

    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,

    keywords = ['least squares'],
    classifiers = ['Programming Language :: Python :: 3.10'],

    install_requires=install_requires
)
