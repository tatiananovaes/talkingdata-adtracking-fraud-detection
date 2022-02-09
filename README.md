Data: fevereiro de 2022


# Detecção de Fraudes no Tráfego de Cliques em Propagandas de Aplicações Mobile
## talkingdata-adtracking-fraud-detection

O objetivo deste projeto é criar um modelo de aprendizado de máquina para prever se um usuário fará o
download de um aplicativo depois de clicar em um anúncio para dispositivos móveis. Portanto, o objetivo
é criar um modelo de classificação para determinar se um clique é fraudulento ou não.

O problema foi inicialmente apresentado em uma competição do Kaggle (https://www.kaggle.com/) pela TalkingData (https://www.talkingdata.com/), grande plataforma de Big Data
independente da China, que cobre mais de 70% dos dispositivos móveis ativos em todo o país.   

Trata-se de um projeto da Formação Cientista de Dados da Data Science Academy (DSA).


# Dados

Os dados foram obtidos na plataforma do Kaggle em: 
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

Neste projeto, foi somente utilizado o dataset train.csv, o qual foi subdividido em treino e teste.

### Dicionário de dados // Attribute Information

ip:                   ip address of click
app:                  app id for marketing (cat)
device:               device type id of user mobile phone (e.g., iphone6 plus, iphone 7, huawei mate 7, etc.) (cat)
os:                   os version id of user mobile phone (cat)
channel:              channel id of mobile ad publisher (cat)
click_time:           timestamp of click (UTC)
attributed_time:      if user download the app for after clicking an ad, this is the time of the app download
is_attributed:        the target that is to be predicted, indicating the app was downloaded

## Note: ip, app, device, os, and channel are encoded


# Dependências

O projeto foi executado por meio da linguagem R no RStudio. As principais bibliotecas utilizadas foram parallel, data.table, caret, dplyr, ROSE,
lubridate, corrplot, ggplot2, gridExtra, randomForest, rpart, xgboost, pROC e ROCR.
