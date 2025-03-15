from setuptools import setup
import os
from glob import glob

package_name = "stl_rover"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "model_trained"),
            glob("model_trained/*"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="heisenberga",
    maintainer_email="davideberga0@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["STL = stl_rover.turtlebot3_STL:main", "paper = stl_rover.turtlebot3_STL_paper:main"],
    },
)
