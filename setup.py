from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='rdml_graph',
      version='0.0.4',
      description='Package for Robot Decision Making Lab with various graph functions Ian developed',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/ianran/rdml_graph',
      author='Ian Rankin',
      author_email='rankini@oregonstate.edu',
      license='MIT',
      #packages=['rdml_graph', 'rdml_graph.core'],
      package_dir={"": "src"},
      packages=find_packages(where="src"),
      install_requires=['numpy>=1.3.0','matplotlib>=2.0.0', 'scipy>=1.0.0', 'tqdm>=3.0.0', 'shapely', 'graphviz>=0.16.0', 'haversine>=2.3.0', 'oyaml>=1.0.0', 'statistics'],
      extras_require={'Saving graphs': ["pickle"]},
      python_requires='>=2.7',
      zip_safe=False)
