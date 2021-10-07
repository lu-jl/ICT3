import os

basepath = os.path.dirname(os.path.abspath(__file__))

titanic_path = os.path.join(basepath, 'titanic')
springleaf_path = os.path.join(basepath, 'springleaf')
mushrooms_path = os.path.join(basepath, 'mushrooms')
pca_path = os.path.join(basepath, 'pca')

__all__ = ['titanic_path', 'springleaf_path', 'mushrooms_path', 'pca_path']