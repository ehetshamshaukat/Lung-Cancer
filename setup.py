from setuptools import setup,find_packages
from typing import List

hypen_e="-e ."

def get_packages(file_path)->List[str]:
    requirements=[]
    with open(file_path) as file:
        requirements = file.readlines()
        requirements=[req.replace("\n","")for req in requirements]
        if hypen_e in requirements:
            requirements.remove(hypen_e)
    return requirements
setup(
    name="Lung Cancer",
    version="0.1",
    aurthor="ehetsham",
    author_email="ehetsham.s@gmail.com",
    packages=find_packages(),
    install_requires=get_packages("requirements.txt")
)