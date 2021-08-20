
from setuptools import setup, find_packages



pkg_name = "duqo"
package_dirs = {"": pkg_name}


with open("requirements.txt", "r") as f:
    reqs = f.readlines()

with open("ReadMe.md", "r", encoding="utf-8") as fh:
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
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ],
      package_dir=package_dirs,
      packages=find_packages(where=pkg_name),
      )
