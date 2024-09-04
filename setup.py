from setuptools import setup
import os
from glob import glob

package_name = 'path_plan'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'path'), glob('path/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), ['config/param.yaml']),  
    ],
    install_requires=['setuptools', 'numpy', 'matplotlib'],
    zip_safe=True,
    maintainer='sw',
    maintainer_email='kimsanmaro@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_plan_node  = path_plan.path_plan_node:main',
            'ppnavi_plan     = path_plan.ppnavi_plan:main',
            'global_path_planner = path_plan.global_path_planner:main',
            'path_plan_vectorField = path_plan.path_plan_vectorField:main',
            'vf_example      = path_plan.vf_example:main',
            'dynamic_launch  = path_plan.dynamic_launch:main'
        ],
    },
)
