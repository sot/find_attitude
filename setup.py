# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(name='find_attitude',
      description='Find attitude given a list of ACA star yag/zag coordinates',
      author='Tom Aldcroft',
      author_email='taldcroft@cfa.harvard.edu',
      packages=['find_attitude', 'find_attitude.web_find_attitude', 'find_attitude.tests'],
      package_data={'find_attitude.web_find_attitude': ['templates/*/*.html', 'templates/*.html']},
      tests_require=['pytest'],
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      cmdclass=cmdclass,
      license='BSD',
      )
