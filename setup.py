# coding=utf-8
"""
期货量化交易系统 v2.0
安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取README
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
long_description = ''
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

# 读取requirements
requirements = []
req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
if os.path.exists(req_path):
    with open(req_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)

setup(
    name='futures-quant',
    version='2.0.0',
    description='期货量化交易系统 - 回测、模拟盘、实盘一体化平台',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Tretra',
    python_requires='>=3.8',
    packages=find_packages(exclude=['tests', 'tests.*', 'data', 'data.*']),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'flake8>=6.0.0',
            'black>=23.0.0',
            'mypy>=1.5.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'futures-quant=run:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
    keywords='futures, quantitative trading, backtest, tqsdk',
)
