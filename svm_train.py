from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import math
import numpy as np

pasta_features = "feature_extr"
file_name = "data"
test_size = 0.2

for subj in range(1, 16):
    pasta_data = str(subj)
    print("[*] Treinando", subj)
    path_data = pasta_features + "/" + pasta_data + "/"

    data = []
    labels = []

    with open("%s%s" % (path_data, file_name), 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split(' ')
            features = []
            labels.append(temp[0])
            for i in range(1, len(temp) - 1):
                features.append(float(temp[i].split(':')[1]))
            data.append(features)
    data = np.array(data)
    for i in range(len(data)):
        som = data[i].sum()
        for feat in range(len(data[i])):
            data[i][feat] = data[i][feat] / som


    #   Normalizando os feature vectors
    #data = preprocessing.normalize(data, norm="max", axis=1)
    #data = preprocessing.scale(data, axis=1)
    #data = preprocessing.normalize(data, axis=1)


    #   Dividindo os dados em train e test datasets
    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=test_size, random_state=203)

    #   Fazendo scaling
    #   Ranges dos parâmetros a serem testados
    C_range = np.logspace(-5, 15, 21, base=2)
    gamma_range = np.logspace(-15, 3, 19, base=2)
    parameters = dict(svc__gamma=gamma_range, svc__C=C_range, svc__kernel=["rbf"])
   
    #   Faz um split no subset de treino, primeiro ele é dividido em treino e teste e depois o treino é dividido em 10 para fazer o 10 fold cross-validation e testar contra o subset de teste (que não é o mesmo que o data_test)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    svc = make_pipeline(preprocessing.StandardScaler(), svm.SVC(kernel="rbf", cache_size=2048))

    #svc = make_pipeline(svm.SVC(kernel="rbf"))
    #   Faz gridsearch
    clf = GridSearchCV(svc, param_grid=parameters, cv=cv, scoring="accuracy", refit="accuracy")
    clf.fit(data_train, label_train)
    #   Treina o modelo com os melhores parâmetros que foram encontrados no CV
    
    svc.set_params(**clf.best_params_)
    print(svc)

    cv_score = cross_val_score(estimator=svc, X=data_train, y=label_train, cv=cv)
    print("[*] Crossval score", cv_score.sum()/len(cv_score))
    #clf.fit(data_train, label_train)
    #   Testa o modelo contra o data_test, que foi separado lá no começo, de forma que os dados não vazam.
    pred = clf.predict(data_test)

    #   Eu rodo este script e faço um pipe com > no terminal para salvar o log, tipo:
    #   python3 svm_train.py > dump_teste.txt
    print("[*] Melhores parâmetros: %s, com score: %f" % (clf.best_params_, clf.best_score_))
    print("[*] Accuracy: %f" % (metrics.accuracy_score(label_test, pred)))
    #print("[*] Precision: %f" % (metrics.precision_score(label_test, pred, average=None)))
    print("[*] Labels da prediction", pred)
    print("[*] Labels do teste", label_test)
    print("[*] Report do scikit", metrics.classification_report(label_test, pred))
    print("\n")

   





