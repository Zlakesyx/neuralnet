from setuptools import setup
from setuptools import find_packages

setup(
    name="neuralnet",
    version="0.0.1",
    description="Not optimized toy neural network",
    author="Zlakesyx",
    url="https://github.com/zlakesyx/neuralnet",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    license="MIT",
    install_requires=["wheel"],
)
