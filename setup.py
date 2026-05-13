#!/usr/bin/env python

from os import path, walk

import sys
from setuptools import setup, find_packages

NAME = "orange-experiment-analytics"

VERSION = "1.0.0"

AUTHOR = 'Revelo, d.o.o.'

URL = 'https://revelo.ai/'
DESCRIPTION = "Add-on containing Experiment Analytics specific widgets"
LONG_DESCRIPTION = """
Orange Experiment Analytics
===========================

Experiment Analytics specific Orange3 add-on.
"""

LICENSE = "BSD"

KEYWORDS = [
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    'orange3 add-on',
]

PACKAGES = find_packages()

PACKAGE_DATA = {
    'orangecontrib.experiment_analytics': ['tutorials/*.ows'],
    'orangecontrib.experiment_analytics.widgets': ['icons/*'],
}

DATA_FILES = [
    # Data files that will be installed outside site-packages folder
]

INSTALL_REQUIRES = [
    'AnyQt',
    'numpy',
    'Orange3 >=3.31.0',
    'orange-widget-base',
    'orange-canvas-core',
    'pandas >=0.23',
    'pymssql',
    'scipy >= 1.8.0',
    'scikit-learn',
    'statsmodels',
]

ENTRY_POINTS = {
    # Entry points that marks this package as an orange add-on. If set, addon will
    # be shown in the add-ons manager even if not published on PyPi.
    'orange3.addon': (
        'experiment_analytics = orangecontrib.experiment_analytics',
    ),
    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    'orange.widgets.tutorials': (
        # Syntax: any_text = path.to.package.containing.tutorials
        'exampletutorials = orangecontrib.experiment_analytics.tutorials',
    ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/example/widgets/__init__.py
        'Experiment Analytics = orangecontrib.experiment_analytics.widgets',
    ),

    # Register widget help
    "orange.canvas.help": (
        'html-index = orangecontrib.experiment_analytics.widgets:WIDGET_HELP_PATH',)
}

NAMESPACE_PACKAGES = ["orangecontrib"]

TEST_SUITE = "orangecontrib.experiment_analytics.tests.suite"


def include_documentation(local_dir, install_dir):
    global DATA_FILES

    doc_files = []
    for dirpath, dirs, files in walk(local_dir):
        doc_files.append(
            (
                dirpath.replace(local_dir, install_dir),
                [path.join(dirpath, f) for f in files],
            )
        )
    DATA_FILES.extend(doc_files)


if __name__ == '__main__':
    include_documentation('doc/_build/htmlhelp', 'help/orange3-experiment_analytics')
    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        license=LICENSE,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=NAMESPACE_PACKAGES,
        zip_safe=False,
    )
