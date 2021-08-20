import os
from setuptools import setup, find_packages

pkg_name = "duqo"
package_dirs = {pkg_name: pkg_name}

base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, "requirements.txt"), "r") as f:
    reqs = f.readlines()

with open(os.path.join(base_dir, "ReadMe.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# inits = [f"{fn}.__init__" for fn in [pkg_name] + [f"{pkg_name}.{mn}" for mn in submodules]]
# pures = [f"stochos_adapter.remote.{name}" for name in ["train_helper", "update_helper", "opti_helper"]]
setup(name=pkg_name,
      version="0.0a0",
      author="Can Bogoclu",
      url="https://github.com/canbooo/duqo",
      install_requires=reqs,
      author_email="can.bogoclu@gmail.com",
      description="A framework for (D)esign (U)ncertainty (Q)uantification and (O)ptimization",
      long_description=long_description,
      long_description_content_type="text/markdown",
      project_urls={
          "Bug Tracker": "https://github.com/canbooo/duqo/issues",
      },
      classifiers=[
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Programming Language :: Python",
          "Topic :: Software Development",
          "Topic :: Scientific/Engineering",
          "Operating System :: Microsoft :: Windows",
          "Operating System :: POSIX",
          "Operating System :: Unix",
          "Operating System :: MacOS",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "License :: OSI Approved :: BSD License",
      ],
      package_dir=package_dirs,
      packages=[pkg_name + "." + p for p in find_packages(where=pkg_name)],
      py_modules=[f"{pkg_name}.__init__"],
      python_requires=">=3.7",
      package_data={
          "": ["requirements.txt", "ReadMe.md"],
      },
      )
