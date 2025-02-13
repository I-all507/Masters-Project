from setuptools import setup, find_packages

setup(
    name='waveform',
    version='0.1',
    packages=find_packages(),
    package_data={
        'waveform': ['PSDs/ET_psd.txt','PSDs/CE_psd.txt']
    }
)