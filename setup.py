from setuptools import setup, find_packages
from DaPy import __version__, _unittests
from DaPy.core.base.constant import PYTHON2

pkg = find_packages()
_unittests()

requirements = [
        'sklearn >= 0.19.2',    
        'numpy >= 1.16.1',
    ]

setup(
    name='SARW',
    version=__version__,
    description='Syntactic Attention-based Random Walk Model for Sentence Representation',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ],
    author='Xuansheng Wu',
    author_email='wuxsmail@163.com',
    maintainer='Xuansheng Wu',
    maintainer_email='wuxsmail@163.com',
    platforms=['all'],
    url='http://dapy.kitgram.cn',
    license='GPL v3',
    packages=pkg,
    package_dir={'DaPy.datasets': 'DaPy/datasets'},
    package_data={'DaPy.datasets': ['adult/*.*', 'example/*.*', 'iris/*.*', 'wine/*.*']},
    zip_safe=True,
    install_requires=requirements

)
