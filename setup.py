from setuptools import find_packages, setup

NAME = 'TESSELLATE'
DESCRIPTION = 'All Sky Transient Search through TESS data'
URL = 'https://github.com/rhoxu/TESSELLATE'
EMAIL = 'roxburghhugh@gmail.com'
AUTHOR ='Hugh Roxburgh'
VERSION = '2.0.1'
REQUIRED = ['astrocut @ git+https://github.com/CheerfulUser/astrocut.git',
            'photutils>=1.4',
            'tessreduce @ git+https://github.com/CheerfulUser/TESSreduce.git@dev',#git+https://github.com/rhoxu/TESSreduce.git',#@b3054d51f2f0993a2ce386d5f2ea7635e5aa2288'
            'sourcedetect @ git+https://github.com/andrewmoore73/SourceDetect.git',
            'shapely',
            'mastcasjobs',
            'astro-datalab',
            'duckdb',
            'pyarrow',
            'scipy',
            'requests',
            'skyfield',
            'spiceypy',
            'rebound',
            'assist',
            ]



setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author_email=EMAIL,
    author=AUTHOR,
    license='MIT',
    packages=find_packages(),#['tessellate'],
    install_requires=REQUIRED,
    include_package_data=True
)




