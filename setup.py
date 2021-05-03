import setuptools
import pathlib

# Read the README file
this_directory = pathlib.Path(__file__).parent
with open((this_directory / "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
  name = "cafcore",
  version = "0.1.0",
  description = "Core functions used to process data for R.J. Cook Agronomy Farm LTAR",
  long_description = long_description,
  long_description_content_type="text/markdown",
  packages = setuptools.find_packages(),
  license="CC0",
  author = "Bryan Carlson",
  author_email = "bryan.carlson@usda.gov",
  url = "https://github.com/cafltar/cafcore",
  keywords = [],
  install_requires=  ["pathlib", "pandas", "numpy", "geopandas"],
  classifiers=[
    "Development Status :: 4 - Beta",     
		"Intended Audience :: Science/Research",      
		"Topic :: Scientific/Engineering",
		"License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",  
		"Programming Language :: Python :: 3"
  ],
  include_package_data=True
)