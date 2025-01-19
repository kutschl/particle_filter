from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'particle_filter'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test'], include=['particle_filter', 'particle_filter.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xml')),
        # TODO: remove 
        (os.path.join('share', package_name, 'launch'), glob('launch/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lukas Kutsch',
    maintainer_email='lukas.kutsch@gmail.com',
    description='The SynPF particle filter model optimized for autonomous racing.',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'particle_filter = particle_filter.particle_filter:main',
        ],
    },
)
