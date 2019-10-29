'''
Copyright: 2019-present Patryk Orzechowski
Licence: MIT
'''


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

hyper_params={
  'solver' : ['svd','lsqr', 'eigen']
}

est=LinearDiscriminantAnalysis(n_components=2)