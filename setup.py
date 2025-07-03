from setuptools import setup, find_packages

setup(
    name="AutoOrgan",                  # 包名（需唯一）
    version="0.1.4",                   # 版本号
    author="Wang Nannan",                # 作者名
    author_email="your.email@example.com",  # 作者邮箱
    description="A short description of the package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wangnanv5/AutoOrgan",  # 项目主页
    packages=find_packages(),          # 自动发现所有 package
    entry_points={
        'console_scripts': [
            'AutoOrgan = AutoOrgan.main:main_entry'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',           # Python 版本要求
    install_requires=['onnxruntime-gpu',
                      'scipy',
                      'scikit-image',
                      'munch',
                      'tqdm',
                      'SimpleITK'
                      ],               # 依赖项
)