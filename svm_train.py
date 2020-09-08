from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
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

    #   Normalizando os feature vectors
    data = preprocessing.normalize(data)
    #   Fazendo scaling
    data = preprocessing.scale(data)
    #   Dividindo os dados em train e test datasets
    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=test_size, random_state=203)


    #   Ranges dos parâmetros a serem testados
    C_range = np.logspace(-2, 10, 14, base=2)
    gamma_range = np.logspace(-9, 3, 13)
    parameters = dict(gamma=gamma_range, C=C_range, kernel=["rbf"])
   
    #   Faz um split no subset de treino, primeiro ele é dividido em treino e teste e depois o treino é dividido em 10 para fazer o 10 fold cross-validation e testar contra o subset de teste (que não é o mesmo que o data_test)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=test_size, random_state=42)
    
    svc = svm.SVC(kernel="rbf")
    #   Faz gridsearch
    clf = GridSearchCV(svc, param_grid=parameters, cv=cv)
    #   Treina o modelo com os melhores parâmetros que foram encontrados no CV
    clf.fit(data_train, label_train)
    #   Testa o modelo contra o data_test, que foi separado lá no começo, de forma que os dados não vazam.
    pred = clf.predict(data_test)

    #   Eu rodo este script e faço um pipe com > no terminal para salvar o log, tipo:
    #   python3 svm_train.py > dump_teste.txt
    print("[*] Melhores parâmetros: %s, com score: %f" % (clf.best_params_, clf.best_score_))
    print("[*] Accuracy: %f" % (metrics.accuracy_score(label_test, pred)))
    print("[*] Labels da prediction", pred)
    print("[*] Labels do teste", label_test)
    print("[*] Report do scikit", metrics.classification_report(label_test, pred))
    print("\n")


