import setuptools

REQUIRED = [
    'numpy',
    'scipy',
    'pylops',
    'requests',
    'h5py',
    'tqdm'
]

with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()
    setuptools.setup(
        name = 'trips',
        version = '0.1.0',
        author = 'Mirjeta Pasha, Connor Sanderford',
        description = 'A package implementing approaches for regularization of ill-posed inverse problems.',
        long_description = LONG_DESCRIPTION,
        long_description_content_type ='text/markdown',
        url = 'https://sites.google.com/view/mirjeta-pasha/home',
        packages = setuptools.find_packages(),
        python_requires = '>=3.5',
        install_requires = REQUIRED,
        classifiers = ["Programming Language :: Python :: 3",
        "License :: OSI Approved :: The Unlicense (Unlicense)",
        "Operating System :: OS Independent"]
    )