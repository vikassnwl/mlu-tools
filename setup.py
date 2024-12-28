from setuptools import setup, find_packages
import codecs
import os
import re

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.16'
DESCRIPTION = 'Contains useful functions and classes'

HYPHEN_E_DOT = r"-e .+"
def get_requirements(file_path):
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [
            req.replace("\n", "")
            for req in requirements
            if not re.search(HYPHEN_E_DOT, req)
        ]
    return requirements

# Setting up
setup(
    name="mlu_tools",
    version=VERSION,
    author="Vikas Sanwal",
    author_email="<vikassnwl@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    keywords=['python'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
