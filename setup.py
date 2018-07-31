from setuptools import setup

setup(name='bodysnatcher',
      version='0.5',
      description='Locate body parts with depth info',
      url='http://github.com/antlai/bodysnatcher',
      author='Antonio Lain',
      author_email='antlai@cafjs.com',
      license='Apache 2.0',
      packages=['bodysnatcher'],
      # use pip install --process-dependency-links  -e .
      dependency_links=['https://github.com/antlai/pylibfreenect2/tarball/master#egg=pylibfreenect2-1.0.0'],
      install_requires=[
          'cherrypy',
          'numpy',
#          'opencv-python', sometimes manually installed
          'pylibfreenect2>=1.0.0',
#          'caffe' has to be manually installed
      ],
      zip_safe=False)
