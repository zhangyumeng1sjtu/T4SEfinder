import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="T4SEfinder",
    version="0.1.0",
    author="Yumeng Zhang",
    author_email="zhangyumeng1@sjtu.edu.cn",
    description="A bioinformatics tool for genome-scale prediction of bacterial type IV secreted effectors using pre-trained protein language model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhangyumeng1sjtu/T4SEfinder",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch == 1.7.0',
        'pandas == 1.2.0',
        'numpy == 1.20.2',
        'joblib == 0.17.0',
        'biopython == 1.79',
        'scikit_learn == 0.24.2',
        'tape_proteins'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause \"New\" or \"Revised\" License (BSD-3-Clause)",
        "Operating System :: OS Independent",
    ],
)
