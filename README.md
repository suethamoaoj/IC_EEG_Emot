# IC_EEG_Emot

Todos os arquivos que usei para treinar o modelo estão aqui. Da forma que está, meus resultados são:

|Subject|10f CV|Test|Neutral|Sad|Fear|Happy Emotions|
|-------|------|----|-------|---|----|--------------|
|Subj 1|68,44%|72,50%|74,00%|80,00%|68,00%|64,00%|
|Subj 2|79,53%|80,00%|92,00%|72,00%|71,00%|88,00%|
|Subj 3|70,31%|68,75%|70,00%|65,00%|71,00%|67,00%|
|Subj 4|67,19%|76,25%|91,00%|77,00%|67,00%|67,00%|
|Subj 5|68,91%|71,25%|85,00%|60,00%|76,00%|58,00%|
|Subj 6|71,41%|77,50%|69,00%|100,00%|74,00%|73,00%|
|Subj 7|73,59%|72,50%|64,00%|91,00%|87,00%|50,00%|
|Subj 8|79,22%|83,75%|88,00%|77,00%|88,00%|88,00%|
|Subj 9|61,25%|66,25%|74,00%|74,00%|52,00%|62,00%|
|Subj 10|70,94%|67,50%|71,00%|72,00%|58,00%|67,00%|
|Subj 11|73,44%|75,00%|86,00%|72,00%|62,00%|83,00%|
|Subj 12|64,38%|67,50%|68,00%|71,00%|65,00%|60,00%|
|Subj 13|70,16%|71,25%|70,00%|67,00%|75,00%|83,00%|
|Subj 14|73,75%|72,50%|65,00%|90,00%|72,00%|62,00%|
|Subj 15|68,13%|67,50%|67,00%|69,00%|69,00%|64,00%|
|Avg|70,71%|72,67%|75,60%|75,80%|70,33%|69,07%|

Obs.: o dataset de treino possui 357 data points e o de teste, 40. Mais dados podem ser vistos no arquivo <b>log_treino.txt</b>.

Para os dados maiores, incluí somente um exemplo (pastas <b>wica_raw</b> e <b>raw</b>).<br><br>
Em <b>feature_extr</b> estão todos os dados usados para treino e teste de cada subject.<br><br>
<b>preprocess.py</b> aplica, em cima dos dados crus do dataset SEED-IV que eu deixo na pasta raw, um filtro bandpass de 1-75Hz e depois o algortimo do wICA para remoção de artefatos, feito com base neste paper (https://scholarworks.gvsu.edu/cgi/viewcontent.cgi?article=1799&context=theses), resultando em dados na pasta wica_raw.<br><br>
<b>feature_extraction.py</b> extrai as características dos dados de wica_raw e os transforma nos dados de treino na pasta feature_extr, com base no método proposto em:<br>
<b>Yuan-Pin Lin, Chi-Hong Wang, Tzyy-Ping Jung, Tien-Lin Wu, Shyh-Kang Jeng, Jeng-Ren Duann, & Jyh-Horng Chen. (2010). EEG-Based Emotion Recognition in Music Listening. IEEE Transactions on Biomedical Engineering, 57(7), 1798–1806. </b><br><br>
<b>svm_train.py</b> treina os modelos para cada subject a partir dos dados em feature_extr, utilizando o SVM do scikit-learn.<br><br>
Tenho uma lista muito maior de referências, mas preciso organizá-las ainda.

