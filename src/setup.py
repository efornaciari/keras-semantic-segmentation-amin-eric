from distutils.core import setup

setup(
    name='segmentation',
    version='0.0.1',
    description='Python Distribution Utilities',
    author='Amin Zadeh Shirazi, Eric Fornaciari',
    author_email='efornaci@gmail.com',
    packages=[
        'segmentation',
        'segmentation.data'
    ],
    install_requires=[
        'numpy',
        'pandas',
        'keras',
        'tensorflow'
    ])
