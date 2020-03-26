from setuptools import setup, find_packages
from SARW import __version__

pkg = find_packages()

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
    url='https://github.com/JacksonWuxs/Syntactic-Attention-based-Random-Walk',
    license='GPL v3',
    packages=pkg,
    package_dir={},
    package_data={},
    zip_safe=True,
    install_requires=requirements

)
