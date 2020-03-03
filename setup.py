from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='rdml_graph',
      version='0.0.1',
      description='Package for Robot Decision Making Lab various graph functions Ian developed',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://tb.d',
      author='Ian Rankin',
      author_email='rankini@oregonstate.edu',
      license='None',
      packages=['rdml_graph'],
      install_requires=['numpy>=1.14.0','matplotlib>=2.1.2', 'scipy>=1.2.3'],
      python_requires='>=2.7',
      zip_safe=False)
