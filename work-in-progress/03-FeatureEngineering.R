# 03-FeatureEngineering


########## Formação Cientista de Dados - Data Science Academy (DSA) ##########

### Projeto do curso Big Data Analytics com R e Microsoft Azure - TalkingData_AdTracking-Fraud-Detection


########## Pacotes // Packages ##########

library(data.table)
library(parallel)
library(lubridate)
#library(forcats)
library(ggplot2)
library(gridExtra)
library(corrplot)


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
  unlist(mclapply(dt, function(x)sum(is.na(x)), mc.cores = 1))
}

unique_per_column <- function(dt){
  unlist(mclapply(dt, function(x) length(unique(x)), mc.cores = 1))
}

del_column <- function(dt, feat) {
  dt <- dt[, (feat) := NULL]
}


####################### TRAIN DATA ############################

########## Carga de dados // Reading data files ##########

train_data_feat <- read_dataset("datasets/transformed/train_data_rose.csv")
str(train_data_feat)


########## Variáveis categóricas // Factor variables ############## 

cols_cat <- c("ip", "app", "device", "os", "channel")
target <- c("is_attributed")

to_factor(train_data_feat, c(cols_cat, target))
to_numeric(train_data_feat, cols_cat)
str(train_data_feat)


########## Engenharia de atributos (treino) // Feature engineering ##########

### Criando novas variáveis // Creating new variables

train_data_feat <- train_data_feat[, `:=`(count_click_by_ip = .N) , by = ip ]


train_data_feat <- train_data_feat[, `:=`(diff_time_click = as.double.difftime(max(as.POSIXct(click_time)) - min(as.POSIXct(click_time)))) , by = ip ]


train_data_feat <- train_data_feat[, ':=' (day_of_week_click = wday(ymd_hms(click_time)),
                                           day_of_year_click = yday(ymd_hms(click_time)),
                                           hour_click = hour(ymd_hms(click_time)),
                                           minute_click = minute(ymd_hms(click_time))
)]                            

train_data_feat <-  del_column(train_data_feat, c('ip','click_time'))

to_factor(train_data_feat, c("day_of_week_click"))
to_numeric(train_data_feat, c("day_of_week_click"))


#### Agrupando níveis das variáveis categóricas // Lumping together factor levels

#train_data_feat <- train_data_feat[, (cols_cat[-1]) := mclapply(.SD, function(x) fct_lump_n(x, 20), mc.cores = 1), .SDcols = cols_cat[-1]]

#str(train_data_feat)
#unique_per_column(train_data_feat)

# Sumário estatístico
summary(train_data_feat)


### Visualizando atributos em relação à target # REVER = PERDEU O SENTIDO

g1 <- ggplot(train_data_feat, aes(app, fill=is_attributed)) + geom_bar(position='fill') + scale_fill_manual(values = c("#999999", "#E69F00"))
g2 <- ggplot(train_data_feat, aes(device, fill=is_attributed)) + geom_bar(position='fill') + scale_fill_manual(values = c("#999999", "#E69F00"))
g3 <- ggplot(train_data_feat, aes(os, fill=is_attributed)) + geom_bar(position='fill') + scale_fill_manual(values = c("#999999", "#E69F00"))
g4 <- ggplot(train_data_feat, aes(channel, fill=is_attributed)) + geom_bar(position='fill') + scale_fill_manual(values = c("#999999", "#E69F00"))

grid.arrange(g1, g2, g3, g4, ncol = 2)


### Normalizando variáveis numéricas

cols_num <- c("count_click_by_ip", "diff_time_click", "day_of_year_click","hour_click", "minute_click")
train_data_feat <- train_data_feat[, (cols_num) := mclapply(.SD, scale, mc.cores = 1), .SDcols = cols_num]


#View(train_data_feat)
str(train_data_feat)
missing_per_column(train_data_feat) # 0 missing


# Sumário estatístico
summary(train_data_feat)

b1 <- boxplot(count_click_by_ip ~ is_attributed, data=train_data_feat, col=c("yellow", "blue"), xlab="Download", ylab="count_click_by_ip")
b2 <- boxplot(diff_time_click ~ is_attributed, data=train_data_feat, col=c("yellow", "blue"), xlab="Download", ylab="diff_time_click")
b3 <- boxplot(day_of_year_click ~ is_attributed, data=train_data_feat, col=c("yellow", "blue"), xlab="Download", ylab="day_of_year_click")
b4 <- boxplot(hour_click ~ is_attributed, data=train_data_feat, col=c("yellow", "blue"), xlab="Download", ylab="hour_click")
b5 <- boxplot(minute_click ~ is_attributed, data=train_data_feat, col=c("yellow", "blue"), xlab="Download", ylab="minute_click")

# Insight: a análise dos boxplots revela diferença significativa entre as medianas apenas em relação à variável diff_time_click.


# Correlação entre as variáveis numéricas
cor_data <- data.frame(train_data_feat$count_click_by_ip, train_data_feat$diff_time_click,
                       train_data_feat$day_of_year_click, train_data_feat$hour_click, train_data_feat$minute_click)
str(cor_data)
corr <- cor(cor_data)
corrplot(corr, method = "number")

### correlação entre as variáveis não é significante



# Gravando arquivo com dataset pré-processado

fwrite(train_data_feat, file="datasets/transformed/train_data_feat.csv")

rm(train_data_feat)


####################### TEST DATA ############################

########## Carga de dados // Reading data files ##########

test_data_feat <- read_dataset("datasets/transformed/test_data.csv")
str(test_data_feat)


########## Variáveis categóricas // Factor variables ############## 

cols_cat <- c("ip", "app", "device", "os", "channel")
target <- c("is_attributed")

to_factor(test_data_feat, c(cols_cat, target))
to_numeric(test_data_feat, cols_cat)
str(test_data_feat)


########## Engenharia de atributos (teste) // Feature engineering (test) ##########


### Criando novas variáveis // Creating new variables


test_data_feat <- test_data_feat[, `:=`(count_click_by_ip = .N) , by = ip ]


test_data_feat <- test_data_feat[, `:=`(diff_time_click = as.double.difftime(max(as.POSIXct(click_time)) - min(as.POSIXct(click_time)))) , by = ip ]


test_data_feat <- test_data_feat[, ':=' (day_of_week_click = wday(ymd_hms(click_time)),
                                         day_of_year_click = yday(ymd_hms(click_time)),
                                         hour_click = hour(ymd_hms(click_time)),
                                         minute_click = minute(ymd_hms(click_time))
)]


test_data_feat <-  del_column(test_data_feat, c('ip','click_time'))


to_factor(test_data_feat, c("day_of_week_click"))
to_numeric(test_data_feat, c("day_of_week_click"))

#### Agrupando níveis das variáveis categóricas // Lumping together factor levels

#test_data_feat <- test_data_feat[, (cols_cat[-1]) := mclapply(.SD, function(x) fct_lump_n(x, 20), mc.cores = 1), .SDcols = cols_cat[-1]]

#str(test_data_feat)
#unique_per_column(test_data_feat)


### Normalizando variáveis numéricas

test_data_feat <- test_data_feat[, (cols_num) := mclapply(.SD, scale, mc.cores = 1), .SDcols = cols_num]


#View(test_data_feat)
str(test_data_feat)
missing_per_column(test_data_feat) # 0 missing

# Sumário estatístico
summary(test_data_feat)


# Gravando arquivo com dataset pré-processado

fwrite(test_data_feat, file="datasets/transformed/test_data_feat.csv")



