import os

basepath = os.path.dirname(os.path.abspath(__file__))

titanic_path = os.path.join(basepath, 'titanic')
springleaf_path = os.path.join(basepath, 'springleaf')

__all__ = ['titanic_path', 'springleaf_path']