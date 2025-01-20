from setuptools import find_packages, setup
import os 
from glob import glob 

package_name = 'particle_filter'

setup(
    name=package_name,
    version='0.0.1',
    packages=[
        package_name,
        f'{package_name}.lidar_downsampling',
        f'{package_name}.motion_models'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.png'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lukas',
    maintainer_email='lukas.kutsch@gmail.com',
    description='LAMARRacing particle filter',
    license='MIT license',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'particle_filter = particle_filter.particle_filter_node:main',
        ],
    },
)
