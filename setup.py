from setuptools import setup, find_packages


install_requires = [
    'numpy',
    'matplotlib',
    'pandas',
    'scipy',
    'scikit-learn',
    'jupyterlab',
    'ipykernel',
    'wget',
    'seaborn',
    'plotly',
    'statsmodels'
]

setup(
    name='algo_ecg',
    version='0.0.1',
    packages=['algo_ecg'],
    url='https://github.com/jessie831024/algo-ecg',
    license='Apache',
    author='Jessie Li',
    author_email='jessie831024@gmail.com',
    include_package_data=True,
    description='Detect arrhythmias using ECG data.',
    install_requires=install_requires
)
