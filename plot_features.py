import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

subj = [10, 11]
emot = [2]
duration = 10 #data points
pasta_features = "feature_extr"
file_name = "data"

for s in subj:
    path = pasta_features + "/" + str(s) + "/" + file_name

    data = []
    labels = []

    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split(' ')
            features = []
            labels.append(temp[0])
            for i in range(1, len(temp) - 1):
                features.append(float(temp[i].split(":")[1]))
            data.append(features)

    data = np.array(data)
    #data = preprocessing.normalize(data, norm="max", axis=1)

    
    for i in range(len(data)):
        som = data[i].sum()
        for feat in range(len(data[i])):
            data[i][feat] = data[i][feat] / som

    print(data[0])
    #exit()
    #data = preprocessing.scale(data)

    data_labeled = []
    for x in range(4):
        data_labeled.append([])

    for idx, value in enumerate(labels):
        data_labeled[int(value)].append(data[idx])


    for em in emot:
        plt.figure()
        for b in range(0, 5):

            temp = [[]] * 12
            idx = np.arange(b, 60 + b, step=5)
            
            for i in range(len(idx)):
                temp2 = []
                for t in range(len(data_labeled[em])):
                    if t > duration:
                        break
                    temp2.append(data_labeled[em][t][idx[i]])
                temp[i] = temp2

            for i in range(len(temp)):
                plt.subplot(12,5, (b * 12) + i + 1)
                plt.ylim([0, 10])
                plt.hist(temp[i], duration)

plt.show()
