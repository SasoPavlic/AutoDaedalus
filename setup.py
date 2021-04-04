import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepswarm",
    version="0.0.10",
    author="Edvinas Byla",
    author_email="edvinasbyla@gmail.com",
    description="Neural Architecture Search Powered by Swarm Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pattio/DeepSwarm",
    packages=setuptools.find_packages(),
    package_data={'deepswarm': ['../settings/default.yaml']},
    install_requires=[
        'colorama',
        'pyyaml',
        'scikit-learn',
        'matplotlib'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
