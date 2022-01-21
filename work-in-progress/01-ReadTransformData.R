# 00-ReadTransformData


########## Formação Cientista de Dados - Data Science Academy (DSA) ##########

### Projeto do curso Big Data Analytics com R e Microsoft Azure - TalkingData_AdTracking-Fraud-Detection

# O objetivo do projeto é criar um modelo de aprendizado de máquina para prever se um usuário fará o
# download de um aplicativo depois de clicar em um anúncio para dispositivos móveis. Portanto, o objetivo
# é criar um modelo de classificação para determinar se um clique é fraudulento ou não.

# Fonte de dados: dataset disponível no Kaggle em: 
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data



########## Dicionário de dados // Attribute Information ##########

# ip:                   ip address of click
# app:                  app id for marketing (cat)
# device:               device type id of user mobile phone (e.g., iphone6 plus, iphone 7, huawei mate 7, etc.) (cat)
# os:                   os version id of user mobile phone (cat)
# channel:              channel id of mobile ad publisher (cat)
# click_time:           timestamp of click (UTC)
# attributed_time:      if user download the app for after clicking an ad, this is the time of the app download
# is_attributed:        the target that is to be predicted, indicating the app was downloaded

## Note: ip, app, device, os, and channel are encoded



########## Pacotes // Packages ##########

library(data.table)
library(caret)

memory.limit(size=90000)


############# Funções Auxiliares // Functions #################

read_dataset <- function(path_file){
  # exemplo path_file => "dados/train.csv"
  fread(path_file, stringsAsFactors = F, sep = ",", header =T)
}

to_factor <- function(dt, feat){
  dt <- dt[, (feat) := lapply(.SD, as.factor), .SDcols = feat]
}

missing_per_column <- function(dt){
  sapply(dt, function(x)sum(is.na(x)))
}

unique_per_column <- function(dt){
  sapply(dt, function(x) length(unique(x)))
}

del_column <- function(dt, feat) {
  dt <- dt[, (feat) := NULL]
}


####################### TRAIN DATA ############################

########## Carga de dados // Reading data files ##########

data <- read_dataset("datasets/train.csv")

#View(data)
str(data)
dim(data)

# Dimensões: 184.903.890 registros e 8 variáveis
# As variáveis ip, app, device, os e channel, além da target is_attributed, foram reconhecidas pelo R como int, mas são categóricas.


########## Variáveis categóricas // Factor variables ############## 

cols_cat <- c("ip", "app", "device", "os", "channel")
target <- c("is_attributed")

to_factor(data, c(cols_cat, target))
str(data)


levels_train <- unique_per_column(data) # variáveis categóricas com alta cardinalidade


########## Valores faltantes nos dados de treino // Missing values ##########

#any(is.na(data))
missing_per_column(data) # 184.447.044 missing de attributed_time

print("Percentual de valores faltantes do atributo 'attributed_time': ")
print(sum(is.na(data$attributed_time)) / nrow(data) * 100)

# Considerando que 99,75% dos registros relativos ao atributo "attributed_time" estão ausentes, essa variável será excluída da análise.
del_column(data, c('attributed_time'))

str(data)

# CHECAR DUPLICADOS?


########## Divisão dos dados em treino e teste // Data split ###############

split <- createDataPartition(y = data$is_attributed, p = 0.7, list = FALSE)
train_data <- data[split,]
test_data <- data[-split,]

str(train_data)
str(test_data)

########## Salvando dados de treino em disco // Saving train data file ##########

fwrite(train_data, file="datasets/transformed/train_data.csv")
fwrite(test_data, file="datasets/transformed/test_data.csv")






