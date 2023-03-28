import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    packages=setuptools.find_packages(),
    name="corr_pooling_seirsplus",
    version='0.1',
    description='Models of SEIRS epidemic dynamics with extensions, including network-structured populations, testing, contact tracing, and social distancing. Large-scale screening through pooled testing.',
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/zyyjjj/corr_pooling_seirsplus",
    author='Jiayue Wan, Yujia Zhang',
    license='MIT',
    install_requires=['numpy', 'scipy', 'networkx', "scikit-learn", "node2vec"],
    zip_safe=False)