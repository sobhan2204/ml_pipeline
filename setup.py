from setuptools import find_packages,setup
from typing import List  # This library is important to read the list 

HYPEN_E_DOT='-e .' 

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements] # remove /n whenever it end a line reading

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT) 
    
    return requirements

setup(
name='ml-pipeline',
version='0.0.1',
author='sobhan',
author_email='sobhan2108@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)