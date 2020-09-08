#   Este script serve para carregar os dados crus que estão em formato txt, preprocessá-los e salvá-los.

import mne
from ica import ica1
from rwt.wavelets import *
from rwt.utilities import *
from rwt import rdwt, irdwt
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import math

#   Vars globais
wica_bool = True#   Ativa ou desativa o wICA antes de salvar os dados
pasta_raw = "../raw"
pasta_wica = "wica_raw"
SR_const = 1000 #   Sampling rate
QTD_comps = 60  #   Quantidade de componentes a seres usados no wICA
Kthr = 1.25     #   Threshold dos artefatos. Padrão: 1.15


#   Lê os dados de um subj em uma seção em um filme e transforma-os em um objeto mne.raw
def read_data(sec, subj, film):
    print("[*] Lendo:", sec, subj, film)
    path = pasta_raw + "/" + str(sec) + "/" + str(subj) + "/" + str(film)
    data = []

    with open(path, 'r') as f:
        for l in f:
            split = l.split(' ')
            split = [float(x) for x in split]
            data.append(split)

    for e in range(len(data)):
        for t in range(len(data[e])):
            data[e][t] = data[e][t] * 10e-6 #   Transforma os dados em microvolts
    
    with open("../Channel_Order.txt", 'r') as f: 
        ch_names = list(map(str.strip, f.readlines()))

    ch_types = ["eeg"] * len(ch_names)

    info = mne.create_info(
            ch_names = ch_names,
            ch_types = ch_types,
            sfreq = SR_const
    )

    raw = mne.io.RawArray(data, info)
    raw = raw.drop_channels(ch_names=["CB1", "CB2"])
    raw = raw.set_montage("standard_1020")

    return raw

#   Aplica o wica em um object mne.raw como o retornado pelo read_data, retorna um mne.raw com os artefatos filtrados
def wica(data):
    print("[*] Aplicando wica")
    raw = data.copy()

    #   Filtro de 1 a 75Hz
    raw.filter(1., 75.)
    
    raw_data = raw.get_data().copy()

    #   Tirando a tendência
    raw_data = sp.detrend(raw_data, type="constant") 

    #   Aplicar o ICA com o método infomax e extended
    A, S, W = ica1(raw_data, QTD_comps)
    #   Mixer, Sources, Unmixer
    icaEEG = np.matmul(W, raw_data)
    L = int(np.round(SR_const * 0.1))
    Nchan, Nobser = icaEEG.shape   
    if Nchan > Nobser:
        print("[!] Transpose necessário")
    N = pow(2, math.floor(math.log(Nobser, 2)))
    h = daubcqf(6)
    opt = np.zeros(QTD_comps)
    for c in range(Nchan):
        Y = icaEEG[c][0:N]
        Sig = np.median(np.abs(Y) / 0.6745)
        Thr = 4 * Sig
        idx = (np.abs(Y) > Thr).nonzero()
        idx_ext = np.zeros(int(len(idx[0]) * (2 * L + 1)))
        for k in range(len(idx[0])):
            idx_ext[(2 * L + 1) * k:(2 * L + 1) * (k + 1)] = list(range(idx[0][k] - L, idx[0][k] + L + 1))
        
        id_noise = np.setdiff1d(list(range(0, N)), idx_ext)
        id_artef = np.setdiff1d(list(range(0, N)), id_noise)
        
        if len(id_artef) > 0:
            print("[*] Artefatos encontrados")
            thld = Thr
            KK = 100
            LL = math.floor(math.log(len(Y), 2))
            x1, xh, _ = rdwt(Y, h[0], h[1], LL)
            inc = thld / 10

            while KK > Kthr:
                thld = thld + inc
                xh = hardThreshold(xh, thld)
                xd, _ = irdwt(x1, xh, h[0], h[1], LL)
                xn = np.subtract(Y, xd)
                break
                print("[*] Calculando ratios")
                cn = np.corrcoef(Y[id_noise], xn[id_noise])
                cd = np.corrcoef(Y[id_noise], xd[id_noise])
                ca = np.corrcoef(Y[id_artef], xd[id_artef])
                KK = ca[0][1] / cn[0][1]
                KKnew = ca[0][1] / cd[0][1]

            opt[c] = thld
            print("[*] thld:", thld)

            Y = icaEEG[c][len(icaEEG[c]) - N:len(icaEEG[c])]
            icaEEG[c][0:N] = xn
            LL = math.floor(math.log(len(Y), 2))
            x1, xh, _ = rdwt(Y, h[0], h[1], LL)
            xh = hardThreshold(xh, thld)
            xd, _ = irdwt(x1, xh, h[0], h[1], LL)
            xn = np.subtract(Y, xd)
            print("[*] Componente", c, "filtrado")

            icaEEG[c][N:len(icaEEG[c])] = xn[len(xn)-(Nobser-N):len(xn)]
    dataWica = np.matmul(np.linalg.inv(W), icaEEG).real
    
    with open("../Channel_Order.txt", 'r') as f:
        ch_names = list(map(str.strip, f.readlines()))

    ch_names.remove("CB1")
    ch_names.remove("CB2")

    ch_types = ["eeg"] * len(ch_names)

    info = mne.create_info(
            ch_names = ch_names,
            ch_types = ch_types,
            sfreq = SR_const
    )

    artRemEEG = mne.io.RawArray(dataWica, info)

    return artRemEEG

#   Salva os dados na pasta de dados preprocessados previamente definida
def salvar_dados(subj):
    for sec in range(1, 4):
        for film in range(0, 24):
            with open("%s/%d/%d %d" % (pasta_wica, subj, sec, film), "w+") as f:
                if wica_bool is True:
                    dados, tempos = wica(read_data(sec, subj, film))[:]
                else:
                    dados, tempos = read_data(sec, subj, film)[:]
                
                print(len(dados))

                stringTotal = ""

                for el in range(len(dados)):
                    temp = ' '.join(map(str, dados[el].tolist()))
                    stringTotal = stringTotal + temp

                    if el != len(dados) - 1:
                        stringTotal = stringTotal + '\n'

                f.write(stringTotal)

for subj in range(1, 16):   #   Preprocessa e salva os dados para os subjects na range
    salvar_dados(subj)
