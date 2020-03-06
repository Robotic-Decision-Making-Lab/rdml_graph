from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='rdml_graph',
      version='0.0.2',
      description='Package for Robot Decision Making Lab various graph functions Ian developed',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://tb.d',
      author='Ian Rankin',
      author_email='rankini@oregonstate.edu',
      license='None',
      packages=['rdml_graph'],
      install_requires=['numpy>=1.3.0','matplotlib>=2.0.0', 'scipy>=1.0.0', 'tqdm>=3.0.0'],
      python_requires='>=2.7',
      zip_safe=False)
