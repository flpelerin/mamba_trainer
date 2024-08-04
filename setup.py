from setuptools import setup, find_packages

setup(
    name="example_project",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'example_project= example_project.main:hello_world',
        ],
    },
)
