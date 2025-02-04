from setuptools import setup, find_packages

setup(
    name="csbosons_data_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List dependencies, 
        'pandas>=1.0.0',
        'numpy>=1.18.0',
    ],
    author="Ethan McGarrigle",
    author_email="ethancmcg@gmail.com",
    description="A collection of helper functions for data analysis of field output data from CSBosonsCpp.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/datahelpers",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
