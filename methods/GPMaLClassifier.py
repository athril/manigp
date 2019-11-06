from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, Imputer
import os
import subprocess
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_auc_score
from utils.munkres import Munkres, make_cost_matrix



class GPMaLClassifier(BaseEstimator):

  def __init__(self, dataset='', n_components=2, n_clusters=2, g=10, popsize=100, n_jobs=1, random_state=3319):
    env = dict(os.environ)
    self.dataset = dataset
    self.n_components=n_components
    self.g = g
    self.popsize = popsize
    self.random_state=random_state
    self.n_jobs=n_jobs
    self.path=os.getcwd()

  def fit(self, X, y, sample_weight=None, groups=None):
    self.dataset = self.dataset.split("/")[-1]
    self.n_clusters=len(Counter(y))

    data=pd.DataFrame(X)
    data['target']=y
    f = open(self.path+"/"+self.dataset+"."+str(self.random_state)+"-train", "w")
    f.write("classLast," + str(data.shape[1]-1) + ","+ str(self.n_components)+ ',comma\n')
    f.close()
    data.to_csv(self.path+"/"+self.dataset+"."+str(self.random_state)+"-train",header=None, index=None, mode='a', sep=',')
    #two empty lines at the end, adding header
#    z=subprocess.check_output(['java', '-cp', 'GPMaL/gp-mal-eurogp-19-bin.jar', 'featureLearn.RunGPMaL', 'dataset='+os.getcwd()+"/"+self.dataset+'.'+str(self.random_state)+'-train', 'numtrees=2', 'preprocessing=scale', 'logPrefix='+self.dataset+'-train', 'treeDepth=8', 'featureMin=0', 'featureMax=1', 'normalisePostCreation=false', 'scalePostCreation=false', 'roundPostCreation=true', 'featureLearnParamFile=GPMaL/flNeighboursFG.params', 'doNNs=false', 'random_state='+str(self.random_state)]).decode("utf-8").replace(" ","").split('\n')[-X.shape[0]-3:-2]
    z=subprocess.check_output(['java', '-cp', 'GPMaL/gp-mal-eurogp-19-bin.jar', 'featureLearn.RunGPMaL', 'dataset='+os.getcwd()+"/"+self.dataset+'.'+str(self.random_state)+'-train', 'numtrees=2', 'preprocessing=none', 'logPrefix='+self.dataset+'-train', 'treeDepth=8', 'featureMin=0', 'featureMax=1', 'normalisePostCreation=false', 'scalePostCreation=false', 'roundPostCreation=true', 'featureLearnParamFile=GPMaL/flNeighboursFG.params', 'doNNs=false', 'n_jobs='+str(self.n_jobs), 'random_state='+str(self.random_state)]).decode("utf-8").split('\n')[-X.shape[0]-3:-2]
#    z=str(z).replace("[","").replace("]",""))
#    print('\n'.join(z.replace(" ","")))
#    d=""

#    for a, b in zip(z[::2], z[1::2]):
#      d=d+str(a.replace("[","").strip())+","+str(b.replace("]","").strip())+'\n'
    print(str(z))
#    print("\n")
#    print('\n'.join(z))
#    print("\n")
    X_new=pd.read_csv(StringIO('\n'.join(z)), sep=',')
    X_new.drop('class', axis=1, inplace=True)
    for f in range(0,X.shape[1]-1):
      X_new.drop('F'+str(f), axis=1, inplace=True)
#    print(X_new)
    self.centroids=KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(X_new)
    labels = self.centroids.predict(X_new)
    confusion_m = confusion_matrix(y, labels)
    m = Munkres()
    cost_m = make_cost_matrix(confusion_m)
    target_cluster = m.compute(cost_m) # (target, cluster) assignment pairs.
    self.mapping = {cluster : target for target, cluster in dict(target_cluster).items()}


    return self;

  def predict(self, X,ic=None):
    data=pd.DataFrame(X)
    data['target']=0
    print(data)
    f = open(self.path+"/"+self.dataset+"."+str(self.random_state)+"-test", "w")
    f.write("classLast," + str(data.shape[1]-1) + ","+ str(self.n_components)+ ',comma\n')
    f.close()
    data.to_csv(self.path+"/"+self.dataset+"."+str(self.random_state)+"-test",header=None, index=None, mode='a', sep=',')
#    print(data)
#    z=subprocess.check_output(['java', '-cp', 'GPMaL/gp-mal-eurogp-19-bin.jar', 'featureLearn.LoadAndApplyModel', 'dataset='+os.getcwd()+"/"+self.dataset+"."+str(self.random_state)+'-test', 'model='+os.getcwd()+"/"+self.dataset+'.'+str(self.random_state)+'-train-gpmal.state','numtrees=2', 'preprocessing=none', 'logPrefix=fLNeighboursFG/', 'treeDepth=8', 'featureMin=0', 'featureMax=1', 'normalisePostCreation=false', 'scalePostCreation=false', 'roundPostCreation=true', 'featureLearnParamFile=GPMaL/flNeighboursFG.params', 'doNNs=false', 'n_jobs='+str(self.n_jobs), 'random_state='+str(self.random_state)]).decode("utf-8").split('\n')[-X.shape[0]-2:-1]
    z=subprocess.check_output(['java', '-cp', 'GPMaL/gp-mal-eurogp-19-bin.jar', 'featureLearn.LoadAndApplyModel', 'dataset='+os.getcwd()+"/"+self.dataset+"."+str(self.random_state)+'-test', 'model='+os.getcwd()+"/"+self.dataset+'.'+str(self.random_state)+'-train-gpmal.state','numtrees=2', 'preprocessing=none', 'logPrefix=fLNeighboursFG/', 'treeDepth=8', 'featureMin=0', 'featureMax=1', 'normalisePostCreation=false', 'scalePostCreation=false', 'roundPostCreation=true', 'featureLearnParamFile=GPMaL/flNeighboursFG.params', 'doNNs=false', 'n_jobs='+str(self.n_jobs), 'random_state='+str(self.random_state)]).decode("utf-8").split('\n')[-X.shape[0]-2:-1]
#    z=subprocess.check_output(['java', '-cp', 'GPMaL/gp-mal-eurogp-19-bin.jar', 'featureLearn.LoadAndApplyModel', 'dataset='+os.getcwd()+"/"+self.dataset+"."+str(self.random_state)+'-test', 'model='+os.getcwd()+"/"+self.dataset+'.'+str(self.random_state)+'-train-gpmal.state','numtrees=2', 'preprocessing=none', 'logPrefix=fLNeighboursFG/', 'treeDepth=8', 'featureMin=0', 'featureMax=1', 'normalisePostCreation=false', 'scalePostCreation=false', 'roundPostCreation=true', 'featureLearnParamFile=GPMaL/flNeighboursFG.params', 'doNNs=false', 'random_state='+str(self.random_state)]).decode("utf-8")
#    print('\n'.join(z))
    print(z)
    X_trans=pd.read_csv(StringIO('\n'.join(z)), sep=',')
#    print(str(X.shape))
#    print(X_trans)
#    X_trans.drop('target', axis=1, inplace=True)
    for f in range(0,X.shape[1]-1):
      print(str(f))
      X_trans.drop('F'+str(f), axis=1, inplace=True)

    print(X_trans)
    labels = self.centroids.predict(X_trans)
    y_pred = list(map(self.mapping.get, labels))
    return y_pred



#    if (len(y_pred)!=len(test) ):
#      print("ERROR!")
#    if (np.any(np.isinf(y_pred)) ):
#      print("FOUND INFS!")
#    if (np.any(np.isnan(y_pred)) ):
#      print("FOUND NANs!")
    

    return y_pred


est=GPMaLClassifier()

hyper_params={
#  'random_state':[1001]
  'n_components':[2]
}