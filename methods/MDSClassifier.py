'''
Copyright: 2019-present Patryk Orzechowski
Licence: MIT
'''


from sklearn.base import BaseEstimator
from sklearn.manifold import MDS
from sklearn.metrics import confusion_matrix
from utils.munkres import Munkres, make_cost_matrix
from sklearn.cluster import KMeans



class MDSClassifier(BaseEstimator):

  def __init__(self,max_iter=300, n_components=2, n_clusters=2, metric=True, dissimilarity='euclidean', random_state=3319):
    self.max_iter=max_iter
    self.metric=metric
    self.dissimilarity=dissimilarity
    self.n_components=n_components
    self.n_clusters=n_clusters
    self.random_state=random_state


  def fit(self,X,y):
    self.X_train=X
    #creating a manifold on training data
    self.model = MDS(max_iter=self.max_iter, n_components=self.n_components, metric=self.metric, dissimilarity=self.dissimilarity).fit_transform(X,y)
    #determining centroids for given classes
    self.centroids = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.model)
    self.labels = self.centroids.predict(self.model) # Every point is assigned to a certain cluster.
    #assigning each centroid to the correct cluster
    confusion_m = confusion_matrix(y, self.labels)
    m = Munkres()
    cost_m = make_cost_matrix(confusion_m)
    target_cluster = m.compute(cost_m) # (target, cluster) assignment pairs.
    #saving mapping for predictions
    self.mapping = {cluster : target for target, cluster in dict(target_cluster).items()}


  def predict(self,X_test):
    if (X_test.equals(self.X_train)):
      y_pred = list(map(self.mapping.get, self.labels))
      return y_pred
    return [0]*X_test.shape[0]




hyper_params = {
    'max_iter' : [300,500],
    'metric' : [True,False],
    'dissimilarity' : ['euclidean'],
}


est=MDSClassifier()