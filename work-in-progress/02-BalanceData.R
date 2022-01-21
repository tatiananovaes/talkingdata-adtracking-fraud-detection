# 01-BalanceData


########## Formação Cientista de Dados - Data Science Academy (DSA) ##########

### Projeto do curso Big Data Analytics com R e Microsoft Azure - TalkingData_AdTracking-Fraude-Detection


########## Pacotes // Packages ##########

library(data.table)
library(ROSE)

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

train_data_imbalanced <- read_dataset("datasets/transformed/train_data.csv")
str(train_data_imbalanced) # 6 atributos como int


########## Variáveis categóricas // Factor variables ############## 

cols_cat <- c("ip", "app", "device", "os", "channel")
target <- c("is_attributed")

to_factor(train_data_imbalanced, c(cols_cat, target))
str(train_data_imbalanced)

missing_per_column(train_data_imbalanced) # 0 missing

levels_train <- unique_per_column(train_data_imbalanced) # variáveis categóricas com alta cardinalidade



########## Proporção de classes da target // Imbalanced data ##########

# Verificando o balanceamento das classes
# A classe/target está desbalanceada.

round(prop.table(table(train_data_imbalanced$is_attributed)) * 100, digits = 1) # 99.8 x 0,2

# Representação gráfica do desbalanceamento

barplot(prop.table(table(train_data_imbalanced$is_attributed)), col = "darkred")
title(main="Proporção das classes da target", xlab = "Classe", ylab = "Proporção")



########## Balanceamento com o método undersampling do pacote ROSE // Balancing with ROSE ##########

train_data_rose <- ovun.sample(is_attributed ~ ., data = train_data_imbalanced, method = "under", seed = 123)$data
#?ovun.sample

#View(train_data_rose)
class(train_data_rose)
str(train_data_rose) 
dim(train_data_rose)
prop.table(table(train_data_rose$is_attributed))

# 639.209 observações e 7 variáveis
# 49.98% x 50,02%


# Representação gráfica dos dados antes e após o balanceamento

par(mfrow = c(1,2))
barplot(prop.table(table(train_data_imbalanced$is_attributed)), col = "darkred")
title(main="Proporção das classes da target antes do balanceamento", xlab = "Classe", ylab = "Proporção")

barplot(prop.table(table(train_data_rose$is_attributed)), col = "darkblue")
title(main="Proporção das classes da target após o balanceamento", xlab = "Classe", ylab = "Proporção")

##########

missing_per_column(train_data_rose) # 0 missing

levels_train <- unique_per_column(train_data_rose)

########## Salvando dados balanceados em disco // Saving balanced data file ##########
fwrite(train_data_rose, file="datasets/transformed/train_data_rose.csv")


####################### TEST DATA ############################
# (...)
