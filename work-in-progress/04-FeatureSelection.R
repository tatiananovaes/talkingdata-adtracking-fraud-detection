# 04-FeatureSelection


########## Formação Cientista de Dados - Data Science Academy (DSA) ##########

### Projeto do curso Big Data Analytics com R e Microsoft Azure - TalkingData_AdTracking-Fraude-Detection


########## Pacotes // Packages ##########

library(data.table)
library(parallel)
library(randomForest)

memory.limit(size=90000)

############# Funções Auxiliares // Functions #################

read_dataset <- function(path_file){
  # exemplo path_file => "dados/train.csv"
  fread(path_file, stringsAsFactors = F, sep = ",", header =T)
}

to_factor <- function(dt, feat){
  dt <- dt[, (feat) := mclapply(.SD, as.factor, mc.cores = 1), .SDcols = feat]
}

missing_per_column <- function(dt){
  unlist(mclapply(dt, function(x) sum(is.na(x)), mc.cores = 1))
}

unique_per_column <- function(dt){
  unlist(mclapply(dt, function(x) length(unique(x)), mc.cores = 1))
}

del_column <- function(dt, feat) {
  dt <- dt[, (feat) := NULL]
}



####################### TRAIN DATA ############################

########## Carga de dados // Reading data files ##########

train_data_final <- read_dataset("datasets/transformed/train_data_feat.csv")
str(train_data_final)

########## Variáveis categóricas // Factor variables ############## 

cols_cat <- c("ip", "app", "device", "os", "channel", "day_of_week_click")
target <- c("is_attributed")

to_factor(train_data_final, c(cols_cat[-1], target))
str(train_data_final)

# Sumário estatístico

summary(train_data_final)


##########   Seleção de Variáveis // Feature Selection ########## 

# Modelo randomForest para criar um plot de importância das variáveis

modelo <- randomForest( is_attributed ~ .,
                        data = train_data_final, 
                        ntree = 100, nodesize = 10, importance = TRUE)

varImpPlot(modelo)

# Variáveis mais relevantes:
## Método MeanDecreaseAccuracy => hour_click, channel, app, diff_time_click, count_click_by_ip
## Método MeanDecreaseGini => app, channel, dif_time_clic, count_by_ip, device, os



