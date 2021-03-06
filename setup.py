from setuptools import setup

setup(
    name='infirun',
    version='0.1.0',
    packages=['infirun', 'infirun.pipe', 'infirun.pipe.test', 'infirun.util', 'infirun.util.test', 'infirun.train',
              'infirun.train.test'],
    url='https://github.com/zh217/infirun.git',
    license='MIT',
    author='Ziyang Hu',
    author_email='hu.ziyang@cantab.net',
    description='TBA',
    tests_require=['pytest', 'pytest-timeout'],
    install_requires=['tarjan'],
    entry_points={
        'console_scripts': ['infirun=infirun.cli:main'],
    },
)
