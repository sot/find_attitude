# Licensed under a 3-clause BSD style license - see LICENSE.rst
from distutils.core import setup

setup(name='find_attitude',
      version='0.2',
      description='Find attitude given a list of ACA star yag/zag coordinates',
      author='Tom Aldcroft',
      author_email='taldcroft@cfa.harvard.edu',
      packages=['find_attitude', 'find_attitude.web'],
      package_data={'find_attitude.web': ['templates/*/*.html', 'templates/*.html']},
      license='BSD',
      )
