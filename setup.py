from setuptools import setup
try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(name='ecma',
      version='0.0.1',
      description='Extended Clustering and Model Approximation',
      long_description=read_md('README.md'),
      url='https://github.com/VladimirR46/ecma.git',
      author='Vladimir Antipov',
      author_email='vantipovm@gmail.com',
      packages=['ecma'],
      package_dir={'ecma': 'ecma'},
      install_requires=[
        'numpy',
        'scikit-learn'
      ],
      zip_safe=False)