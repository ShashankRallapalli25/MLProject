from setuptools import find_packages,setup
from typing import List

hyphe_e_dot = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if hyphe_e_dot in requirements:
            requirements = requirements.remove(hyphe_e_dot)


setup(
    name='MLproject',
    version='0.0.1',
    author='shashank',
    author_email= 'shashank.rvss@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)