from os.path import dirname, join

from setuptools import find_packages, setup

with open(join(dirname(__file__), "foolbox/VERSION")) as f:
    version = f.read().strip()

try:
    # obtain long description from README
    readme_path = join(dirname(__file__), "README.rst")
    with open(readme_path, encoding="utf-8") as f:
        README = f.read()
        # remove raw html not supported by PyPI
        README = "\n".join(README.split("\n")[3:])
except IOError:
    README = ""


install_requires = [
    "numpy",
    "scipy",
    "setuptools",
    "eagerpy>=0.30.0",
    "GitPython>=3.0.7",
    "typing-extensions>=3.7.4.1",
    "requests>=2.24.0",
]
tests_require = ["pytest>=7.1.1", "pytest-cov>=3.0.0"]


setup(
    name="foolbox",
    version=version,
    description="Foolbox is an adversarial attacks library that works natively with PyTorch, TensorFlow and JAX",
    long_description=README,
    long_description_content_type="text/x-rst",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    author="Jonas Rauber, Roland S. Zimmermann",
    author_email="foolbox+rzrolandzimmermann@gmail.com",
    url="https://github.com/bethgelab/foolbox",
    license="MIT License",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={"testing": tests_require},
)
