# 05-ModelTraining


########## Formação Cientista de Dados - Data Science Academy (DSA) ##########

### Projeto do curso Big Data Analytics com R e Microsoft Azure - TalkingData_AdTracking-Fraude-Detection


########## Pacotes // Packages ##########

library(data.table)
library(parallel)
library(rpart)
require(xgboost)
library(caret)
library(pROC)
library(ROCR)
library(gmodels)


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

metricas <- function(model, pred, nome){
  results <- caret::confusionMatrix(table(data = pred, reference = test_target))
  acuracia <- results$overall['Accuracy']
  kappa <- results$overall['Kappa']
  curva_roc <- multiclass.roc(response = test_target, predictor = as.numeric(as.factor(pred)))
  auc <- curva_roc$auc
  return(c(nome, round(acuracia, 4), round(kappa, 4), round(auc, 4)))
}

########## Carga de dados // Reading data files ##########

train_data <- read_dataset("datasets/transformed/train_data_feat.csv")
str(train_data)
unique(train_data$is_attributed)

test_data <- read_dataset("datasets/transformed/test_data_feat.csv")
str(test_data)

########## Variáveis categóricas // Factor variables ############## 

cols_cat <- c("ip", "app", "device", "os", "channel", "day_of_week_click")
target <- c("is_attributed")

to_factor(train_data, c(cols_cat[-1], target))
to_numeric(train_data, cols_cat[-1])
str(train_data)

to_factor(test_data, c(cols_cat[-1], target))
to_numeric(test_data, cols_cat[-1])
str(test_data)

# Sumário estatístico

summary(train_data)
summary(test_data)

########## Treinando os modelos // Model training ##############

set.seed(1234)

formula_model_a <- as.formula("is_attributed ~ .")
formula_model_b <- as.formula("is_attributed ~ hour_click + app + diff_time_click + channel + count_click_by_ip") # Método MeanDecreaseAccuracy
formula_model_c <- as.formula("is_attributed ~ app + diff_time_click + channel + count_click_by_ip + device") # Método MeanDecreaseGini
formula_model_d <- as.formula("is_attributed ~ app + channel + device + os + diff_time_click + count_click_by_ip")

test_target <- test_data$is_attributed



# Versão 1 - Regressão Logística com todas as variáveis

model_glm1 <- glm(formula = formula_model_a, data = train_data, family = 'binomial')
summary(model_glm1)
prob_glm1 <- predict(model_glm1, test_data, type = 'response')
pred_glm1 <- ifelse(prob_glm1 > 0.5, 1, 0)

vetor_model_glm1 <- metricas(model_glm1, pred_glm1,'Modelo Regressão Logística 1')
print(vetor_model_glm1)
saveRDS(model_glm1, 'models/model_glm1.rds')

# day_of_year_click = NA (sem significância)
# acuracia 0.7207 kappa 0.0091 auc 0.7607


# Versão 2 - Regressão Logística com variáveis selecionadas com Random Forest (Método MeanDecreaseAccuracy)

model_glm2 <- glm(formula = formula_model_b, data = train_data, family = 'binomial')
summary(model_glm2)
prob_glm2 <- predict(model_glm2, test_data, type = 'response')
pred_glm2 <- ifelse(prob_glm2 > 0.5, 1, 0)

vetor_model_glm2 <- metricas(model_glm2, pred_glm2,'Modelo Regressão Logística 2')
print(vetor_model_glm2)
saveRDS(model_glm2, 'models/model_glm2.rds')

# todas as variáveis com ***
# acuracia 0.7205 kappa 0.0091 auc 0.7603


# Versão 3 - Regressão Logística com variáveis selecionadas (Método MeanDecreaseGini)

model_glm3 <- glm(formula = formula_model_c, data = train_data, family = 'binomial')
summary(model_glm3)
prob_glm3 <- predict(model_glm3, test_data, type = 'response')
pred_glm3 <- ifelse(prob_glm3 > 0.5, 1, 0)

vetor_model_glm3 <- metricas(model_glm3, pred_glm3,'Modelo Regressão Logística 3')
print(vetor_model_glm3)
saveRDS(model_glm3, 'models/model_glm3.rds')

# todas as variáveis com ***
# acuracia 0.7225 kappa 0.0092 auc 0.7601


# Versão 4- Regressão Logística com nova composição de variáveis

model_glm4 <- glm(formula = formula_model_d, data = train_data, family = 'binomial')
summary(model_glm4)
prob_glm4 <- predict(model_glm4, test_data, type = 'response')
pred_glm4 <- ifelse(prob_glm4 > 0.5, 1, 0)

vetor_model_glm4 <- metricas(model_glm4, pred_glm4,'Modelo Regressão Logística 4')
print(vetor_model_glm4)
saveRDS(model_glm4, 'models/model_glm4.rds')

# todas as variáveis com ***, exceto os com **
# acuracia 0.72125 kappa 0.0091 auc 0.7595


# Versão 5 - Random forest (a partir da formula_b)

model_rf1 <- rpart(formula_model_b, data = train_data, control = rpart.control(cp= .0005))
summary(model_rf1)
pred_rf1 <- predict(model_rf1, test_data, type = 'class')

vetor_model_rf1 <- metricas(model_rf1, pred_rf1,'Modelo Random Forest 1')
print(vetor_model_rf1)
saveRDS(model_rf1, 'models/model_rf1.rds')
# acuracia 0.8192 kappa 0.0192 auc 0.8586


# Versão 6- Random forest (a partir da formula_c)

model_rf2 <- rpart(formula_model_c, data = train_data, control = rpart.control(cp= .0005))
summary(model_rf2)
pred_rf2 <- predict(model_rf2, test_data, type = 'class')

vetor_model_rf2 <- metricas(model_rf2, pred_rf2,'Modelo Random Forest 2')
print(vetor_model_rf2)
saveRDS(model_rf2, 'models/model_rf2.rds')
# acuracia 0.8198 kappa 0.0193 auc 0.8589


# Versão 7- Random forest (a partir da formula_c, outros parâmetros)

model_rf3 <- rpart(formula_model_c, data = train_data, control = rpart.control(minsplit=50, cp= .005, maxdepth = 20))
summary(model_rf3)
pred_rf3 <- predict(model_rf3, test_data, type = 'class')

vetor_model_rf3 <- metricas(model_rf3, pred_rf3,'Modelo Random Forest 3')
print(vetor_model_rf3)
saveRDS(model_rf3, 'models/model_rf3.rds')
# acuracia 0.8197 kappa 0.0188 auc 0.8514


# Versão 8 - XGBoost 1 (a partir da formula_b - mesmas variáveis)

features_train1 <- as.matrix(train_data[,c("app", "diff_time_click", "channel", "count_click_by_ip", "hour_click")])
label_train1 <- as.matrix(train_data[,c("is_attributed")])

features_test1 <- as.matrix(test_data[,c("app", "diff_time_click", "channel", "count_click_by_ip", "hour_click")])
label_test1 <- as.matrix(test_data[,c("is_attributed")])

dtrain1 <- xgb.DMatrix(data = features_train1, label = label_train1) # convertendo em matriz densa
dtest1 <- xgb.DMatrix(data = features_test1, label = label_test1)

model_xgboost1 <- xgboost(data = dtrain1,
                         max.depth = 2,
                         eta = 1,
                         nthread = 2,
                         nrounds = 2,
                         objective = "binary:logistic")

prob_xgboost1 <- predict(model_xgboost1, dtest1)
pred_xgboost1 <- as.numeric(prob_xgboost1 > 0.5)

vetor_model_xgboost1 <- metricas(model_xgboost1, pred_xgboost1,'Modelo xGBoost 1')
print(vetor_model_xgboost1)
xgb.save(model_xgboost1, 'models/model_xgboost1')

# acuracia 0.7795 kappa 0.0148 auc 0.8356


# Versão 9- XGBoost 2 (a partir da formula_c - mesmas variáveis)

features_train2 <- as.matrix(train_data[,c("app", "diff_time_click", "channel", "count_click_by_ip", "device")])
label_train2 <- as.matrix(train_data[,c("is_attributed")])

features_test2 <- as.matrix(test_data[,c("app", "diff_time_click", "channel", "count_click_by_ip", "device")])
label_test2 <- as.matrix(test_data[,c("is_attributed")])

dtrain2 <- xgb.DMatrix(data = features_train2, label = label_train2) # convertendo em matriz densa
dtest2 <- xgb.DMatrix(data = features_test2, label = label_test2)

model_xgboost2 <- xgboost(data = dtrain2,
                          max.depth = 2,
                          eta = 1,
                          nthread = 2,
                          nrounds = 2,
                          objective = "binary:logistic")

prob_xgboost2 <- predict(model_xgboost2, dtest2)
pred_xgboost2 <- as.numeric(prob_xgboost2 > 0.5)

vetor_model_xgboost2 <- metricas(model_xgboost2, pred_xgboost2,'Modelo xGBoost 2')
print(vetor_model_xgboost2)
xgb.save(model_xgboost2, 'models/model_xgboost2')

# acuracia 0.7795 kappa 0.0148 auc 0.8356


# Versão 10- XGBoost com Cross Validation

params <- list(booster = "gbtree",
               objective = "binary:logistic",
               eta = c(0.5, 0.9, 1),
               max_depth = c(2, 3),
               min_child_weight = 1,
               subsample = c(0.5, 0.6, 1),
               colsample_bytree=1
)
?xgb.cv
xgbcv <- xgb.cv(params = params, data = dtrain2, nfold = 10, nrounds = 500,
                      showsd = T, stratified = T, print_every_n = 10, early_stop_round = 20,
                      metrics = "auc", maximize = T)

xgbcv$evaluation_log

max_auc <- max(xgbcv$evaluation_log[, 4]) # max test_auc_mean
xgbcv$evaluation_log[,4] == max_auc # 500 -> iteração com maior auc


# max_auc #100 - model training
model_xgbcv <- xgb.train (params = params, data = dtrain2, nrounds = 500,
                   watchlist = list(val=dtest2,train=dtrain2),
                   print_every_n = 100, early_stop_round = 20,
                   maximize = T , metrics = "auc")

prob_xgbcv <- predict(model_xgbcv,dtest2)
pred_xgbcv <- as.numeric(prob_xgbcv > 0.5)

vetor_model_xgbcv <- metricas(model_xgbcv, pred_xgbcv, 'Modelo xGBoost CV')
print(vetor_model_xgbcv)
xgb.save(model_xgbcv, 'models/model_xgbcv')

# acuracia 0.7701 kappa 0.0147 auc 0.8469


## Comparando modelos
compara_modelos <- rbind(vetor_model_glm1, vetor_model_glm2, vetor_model_glm3, vetor_model_glm4,
                         vetor_model_rf1, vetor_model_rf2, vetor_model_rf3,
                         vetor_model_xgboost1, vetor_model_xgboost2, vetor_model_xgbcv)
rownames(compara_modelos) <- c(1:10)
colnames(compara_modelos) <- c('Modelo', 'Acuracia', 'Kappa', 'AUC')
compara_modelos <- as.data.frame(compara_modelos)
print(compara_modelos)

ggplot(compara_modelos, aes(x=Modelo, y=Acuracia, fill=Modelo)) + geom_bar(stat='identity') + theme(axis.text.x = element_text(angle=45,hjust=1,size=8))
ggplot(compara_modelos, aes(x=Modelo, y=AUC, fill=Modelo)) + geom_bar(stat='identity')+ theme(axis.text.x = element_text(angle=45,hjust=.9,size=8))


# O modelo Random Forest 2 apresentou acurácia e curva AUC melhores, sendo o mais indicado para as previsões.






