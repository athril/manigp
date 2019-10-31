'''
Copyright: 2019-present Patryk Orzechowski
Licence: MIT
'''


from sklearn.base import BaseEstimator
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import confusion_matrix
from utils.munkres import Munkres, make_cost_matrix
from sklearn.cluster import KMeans
from methods.LLEClassifier import LLEClassifier





class MLLEClassifier(LLEClassifier):

  def __init__(self,n_neighbors=5, n_components=2, n_clusters=2, reg=0.001, method='standard', eigen_solver='auto', random_state=3319):
    super().__init__(n_neighbors, n_components, n_clusters, reg, 'ltsa', eigen_solver, random_state)


  def fit(self,X,y):
    super().fit(X,y)

  def predict(self,X_test):
    y_pred = super().predict(X_test)
    return y_pred

est = MLLEClassifier(method='modified', eigen_solver='dense')

hyper_params={
   'reg' : [0.00001, 0.0001, 0.001, 0.1, 1, 10],
   'n_neighbors' : [5,6,7,8,9,10,15,20],
}
