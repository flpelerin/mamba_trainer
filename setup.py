from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="mamba_trainer",
    version="0.1",
    packages=find_packages(),
	install_requires=requirements,
)
