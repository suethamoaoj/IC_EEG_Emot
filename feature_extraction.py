#   Este script serve para extrair as características

import scipy.signal as sp
import numpy as np 

#   Sampling rate do dataset
SR_const = 1000
#   Lista de eletrodos pares a serem usados
LST_eletr = "T7,T8,TP7,TP8,FT7,FT8,Fp1,Fp2,F7,F8,F3,F4,FC3,FC4,P7,P8,C3,C4,CP3,CP4,P3,P4,O1,O2".split(',')
#   Índices dos eletrodos pares
eletrIds = [0] * len(LST_eletr)
with open("Channel_Order.txt", "r") as f:
    each = f.readlines()
    each = list(map(str.strip, each))
    exclude = [each.index("CB1"), each.index("CB2")]

    for el in LST_eletr:
        mod = 0
        for idx in exclude:
            if each.index(el) > idx:
                mod = mod + 1
        eletrIds[LST_eletr.index(el)] = each.index(el) - mod
print("[*] Eletrodos: ", eletrIds)
#   Pasta com os dados preprocessados
pasta_wica = "wica_raw"
#   Pasta para salvar as features extraídas
pasta_features = "feature_extr"
#   Tamanho da janela a ser usada no spectogram
window_size = 4000  #   4000 me deu o melhor resultado. Equivale a 4 segundos (SR=1000Hz).


#   Lê os dados preprocessados da pasta_wica
def ler_dados_wica(sec, subj, film):
    raw = []
    with open("%s/%d/%d %d" % (pasta_wica, subj, sec, film), "r") as f:
        lines = f.readlines()
        for line in lines:
            arrayTemp = [float(x) for x in line.split(' ')]
            raw.append(arrayTemp)

    return raw


#   A única diferença para a função que eu usava antes é o mode="magnitude", que melhorou meus resultados.
def spectrogram2(data, eletr):
    f, t, Sxx = sp.spectrogram(np.array(data[eletr]), SR_const, window=sp.get_window("hann", 4000), nfft=4000, noverlap=0, scaling="spectrum")
    return [f, t, Sxx]

#   Tenho duas funções para fazer a extração de características porque ficou subjetivo no paper no qual me baseei qual seria a ordem das operações, mas a que mais confio nos resultados é a averageBand2_spec2
#   Nesta função, eu tiro a média de power de cada banda para cada eletrodo interessante e a subtraio do mesmo mas para o eletrodo simétrico 
def averageBand_spec2(data):
    eletrs = []
    for el in range(len(eletrIds)):
        #   Para cada eletrodo
        eletrs.append([])
        f, t, Sxx = spectrogram2(data[0], eletrIds[el])
        temp = np.swapaxes(Sxx, 0, 1)
        freq_res = f[1] - f[0]
        for T in range(len(t)):
            #   Para cada tempo do eletrodo
            eletrs[el].append([0] * 5)
            for b in range(5):
                #   Para cada frequência do eletrodo
                if b == 0:
                    low = 1
                    high = 3
                elif b == 1:
                    low = 4
                    high = 7
                elif b == 2:
                    low = 8
                    high = 13
                elif b == 3:
                    low = 14
                    high = 30
                elif b == 4:
                    low = 31
                    high = 50
                idx_band = np.logical_and(f >= low, f <= high)
                eletrs[el][T][b] = temp[T][idx_band].sum() / np.count_nonzero(idx_band)

    indicesAssim = []
    for el in range(len(eletrs) // 2):
        indicesAssim.append([])
        for T in range(len(eletrs[el * 2])):
            indicesAssim[el].append([])
            for b in range(5):
                indicesAssim[el][T].append([])
                indicesAssim[el][T][b] = eletrs[el * 2][T][b] - eletrs[(el * 2) + 1][T][b]

    return [indicesAssim, data[2][0]]

#   Nesta função eu subtraio a voltagem de cada par de eletrodos interessantes e os transformo em um só canal para depois tirar a média de poder do par em cada banda
def averageBand2_spec2(data):
    indicesAssim = []
    for el in range(len(eletrIds) // 2):
        indicesAssim.append([])
        for T in range(len(data[0][el * 2])):
            indicesAssim[el].append(data[0][eletrIds[el * 2]][T] - data[0][eletrIds[(el * 2) + 1]][T])
    eletrs = []
    for el in range(len(indicesAssim)):
        #   Para cada eletrodo
        eletrs.append([])
        f, t, Sxx = spectrogram2(indicesAssim, el)
        temp = np.swapaxes(Sxx, 0, 1)
        freq_res = f[1] - f[0]
        for T in range(len(t)):
            #   Para cada tempo do eletrodo
            eletrs[el].append([0] * 5)
            for b in range(5):
                #   Para cada frequência do eletrodo
                if b == 0:
                    low = 1
                    high = 3
                elif b == 1:
                    low = 4
                    high = 7
                elif b == 2:
                    low = 8
                    high = 13
                elif b == 3:
                    low = 14
                    high = 30
                elif b == 4:
                    low = 31
                    high = 50
                idx_band = np.logical_and(f >= low, f <= high)
                eletrs[el][T][b] = temp[T][idx_band].sum() / np.count_nonzero(idx_band)
    return [eletrs, data[2][0]]

#   Formatar os dados para o formato do libSVM
def format_LibSVM(data):
    stringTotal = ""
    for t in range(len(data[0][0])):
        stringTemp = "%d " % data[1]
        ft = 1
        for e in range(len(data[0])):
            for b in range(len(data[0][0][0])):
                stringTemp = stringTemp + "%d:%s " % (ft, str("{:f}".format(float(data[0][e][t][b]) * 10e10)))
                ft = ft + 1
        stringTotal = stringTotal + stringTemp  + "\n"
    return [stringTotal, len(data[0][0])]

#   Simplesmente retorna a label de cada filme
def emotion(session, film):
    session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
    session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1];
    session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0];

    emo = eval("session%s_label[%s]" % (session, film))

    emots = ["Neutral", "Sad", "Fear", "Happy Emotions"]
    #0 = Neutral; 1 = Sad; 2 = Fear; 3 = Happy Emotions;

    return [emo, emots[emo]]

#   Prepara os dados para extrair features
def prep_data(sec, subj, film):
    data = ler_dados_wica(sec, subj, film)

    #   Incorporar metadados
    emot = emotion(sec, film)
    
    return [data, SR_const, emot]

#   Faz o procedimento todo e salva os dados com as features extraídas em pasta_features
def data_full(subjFrom, subjTo):
    pasta = pasta_features + "/" + str(subjFrom)

    dados = []
    for sec in range(1, 4):
        for subj in range(subjFrom, subjTo):
            for film in range(24): #24
                print("Loading: ", sec, subj, film)
                dados.append(averageBand2_spec2(prep_data(sec, subj, film)))

    print("[*] Formatando para LibSVM")
    stringTemp = ""
    for i in range(len(dados)):
        temp = format_LibSVM(dados[i])
        splitS = temp[0].splitlines()
        temp[0] = '\n'.join(splitS[:len(splitS) - 1]) + '\n'
        stringTemp = stringTemp + temp[0]
        print("\t[*] %d" % (i))


    data = '\n'.join(stringTemp.splitlines())

    with open("%s/%s" % (pasta, "data"), "w+") as f:
        f.write(data)

    print("[*] Terminado.")

for subj in range(12, 13):
    data_full(subj, subj + 1)    #   Salvará as características de todos os subjs de 1 até 15 
