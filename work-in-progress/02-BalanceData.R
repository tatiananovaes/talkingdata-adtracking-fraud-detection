# 01-BalanceData


########## Formação Cientista de Dados - Data Science Academy (DSA) ##########

### Projeto do curso Big Data Analytics com R e Microsoft Azure - TalkingData_AdTracking-Fraude-Detection


########## Pacotes // Packages ##########

library(data.table)
library(parallel)
library(dplyr)
library(ROSE)

memory.limit(size=90000)

############# Funções Auxiliares // Functions #################

read_dataset <- function(path_file){
  # exemplo path_file => "dados/train.csv"
  fread(path_file, stringsAsFactors = F, sep = ",", header =T)
}

to_factor <- function(dt, feat){
  dt <- dt[, (feat) := mclapply(.SD, as.factor, mc.cores = 1), .SDcols = feat]
}
to_numeric <- function(dt, feat){
  dt <- dt[, (feat) := mclapply(.SD, as.numeric, mc.cores = 1), .SDcols = feat]
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

train_data <- read_dataset("datasets/transformed/train_data.csv")
str(train_data) # 5 atributos como int
class(train_data)

########## Variáveis categóricas // Factor variables ############## 

cols_cat <- c("ip", "app", "device", "os", "channel")
target <- c("is_attributed")

to_factor(train_data, c(cols_cat, target))
to_numeric(train_data, cols_cat)
str(train_data)

missing_per_column(train_data) # 0 missing

unique_per_column(train_data) # variáveis categóricas com alta cardinalidade



########## Proporção de classes da target // Imbalanced data ##########

# Verificando o balanceamento das classes
# A classe/target está desbalanceada.

round(prop.table(table(train_data$is_attributed)) * 100, digits = 1) # 99.8 x 0,2

tabela_train_prop <- train_data %>% group_by(is_attributed) %>%
  summarise(Total = length(is_attributed)) %>%
  mutate(Taxa = Total / sum(Total) * 100)
print(tabela_train_prop)

# Representação gráfica do desbalanceamento

barplot(prop.table(table(train_data$is_attributed)), col = "darkred")
title(main="Proporção das Classes - Dados de Treino", xlab = "Classe", ylab = "Proporção")



########## Balanceamento com o método undersampling do pacote ROSE // Balancing with ROSE ##########

train_data_rose <- ovun.sample(is_attributed ~ ., data = train_data, method = "under", seed = 123)$data
#?ovun.sample

#View(train_data_rose)
class(train_data_rose)
str(train_data_rose) 
dim(train_data_rose)
prop.table(table(train_data_rose$is_attributed))

# 821.957 observações e 7 variáveis (90% train)
# 49.98% x 50,02%


# Representação gráfica dos dados antes e após o balanceamento

par(mfrow = c(1,2))
barplot(prop.table(table(train_data$is_attributed)), col = "darkred")
title(main="Antes do balanceamento", xlab = "Classe", ylab = "Proporção")

barplot(prop.table(table(train_data_rose$is_attributed)), col = "darkblue")
title(main="Após o balanceamento", xlab = "Classe", ylab = "Proporção")

##########

missing_per_column(train_data_rose) # 0 missing

unique_per_column(train_data_rose)

########## Salvando dados balanceados em disco // Saving balanced data file ##########
fwrite(train_data_rose, file="datasets/transformed/train_data_rose.csv")


#################
## Dados de teste não devem ser balanceados.



