from setuptools import setup, find_packages


VERSION = '0.0.1'
DESCRIPTION = 'Stitching Panoramic Mosaics for Argus'

# Setting up
setup(
    name="panoMosaics",
    version=VERSION,
    author="Erin McgGowan",
    author_email="<egm5434@nyu.edu>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'opencv-contrib-python', 'Pillow', 'shapely', 'numba'],
    keywords=['python', 'video'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)