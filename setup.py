import os
from setuptools import setup, find_packages

setup(
        name='dan',
        version=0.1,
        description='dan furnace',
        # long_description=readme(),
        author='GG',
        author_email='fengqian1991@gmail.com',
        keywords='computer vision, dl',
        url='https://github.com/foocker/Dan',
        # data_files=[] #安装过程中需要安装的静态文件(如配置文件cfg等
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        # package_data={'dan.ops': ['*/*.so']},  #希望被打包的文件
        # exclude_package_data = [] #不打包某些文件
        classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        license='Apache License 2.0',
        extras_require={
            'all': 'requirements.txt',
        },
        
        # ext_modules 参数用于构建 C 和 C++ 扩展扩展包
        
        # install_requires = [] 表明当前模块需要哪些包，如果没有，则会从PyPI中下载安装
        # setup_requires = [] setup.py本身的依赖包，这里列出的包不会自动安装
        # test_requires = [] 仅在测试时候需要使用的依赖，在正常发布的代码中没有用，在执行python
        # setup.py test时候安装setup_requires和test_requires中的依赖
        # dependency_links=[] 用于安装setup_requires和test_requires
        # etras_requires=[] 表示该模块会依赖这些包，但是这些包不会被用到
        
        # cmdclass={'build_ext': build_ext},
        
        zip_safe=False)