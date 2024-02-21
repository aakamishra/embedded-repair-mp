from setuptools import setup

setup(name='wall-e',
      version='0.1.0',
      install_requires=['gym','pybullet', 'opencv-python', 
                        'shapely==1.8.0', 'numpy', 'keras==2.3.1',
                        'tensorflow==1.14', 'keras-rl==0.4.2', 'torch==1.13.1',
                        'matplotlib==3.5.3', 'torch-geometric==2.3.1'],
      author="Aakash Mishra and Dea Dressel",
      author_email="aakashmishra1@gmail.com",
      description="Line follower simulator environment.",
      packages=["gym_line_follower", "examples", "media"]
      )
