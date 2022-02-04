# 05-ModelTraining


########## Formação Cientista de Dados - Data Science Academy (DSA) ##########

### Projeto do curso Big Data Analytics com R e Microsoft Azure - TalkingData_AdTracking-Fraude-Detection


########## Pacotes // Packages ##########

library(data.table)
library(parallel)
library(class) # para modelo knn
library(e1071) # para modelo svm
library(rpart) # para modelo Random Forest
require(xgboost)
library(caret)
library(pROC)
library(ROCR)
library(gmodels) # para avaliar modelo com CrossTable


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

formula_model_a <- as.formula("is_attributed ~ .")
formula_model_b <- as.formula("is_attributed ~ hour_click + app + diff_time_click + channel + count_click_by_ip") # Método MeanDecreaseAccuracy
formula_model_c <- as.formula("is_attributed ~ app + diff_time_click + channel + count_click_by_ip + device") # Método MeanDecreaseGini
formula_model_d <- as.formula("is_attributed ~ app + channel + device + os + diff_time_click + count_click_by_ip")

test_target <- test_data$is_attributed

metricas <- function(model, pred, nome){
  results <- caret::confusionMatrix(table(data = pred, reference = test_target))
  acuracia <- results$overall['Accuracy']
  kappa <- results$overall['Kappa']
  curva_roc <- multiclass.roc(response = test_target, predictor = as.numeric(as.factor(pred)))
  auc <- curva_roc$auc
  return(c(nome, round(acuracia, 4), round(kappa, 4), round(auc, 4)))
}


# Versão 1 - Regressão Logística com todas as variáveis

model_glm1 <- glm(formula = formula_model_a, data = train_data, family = 'binomial')
summary(model_glm1)
prob_glm1 <- predict(model_glm1, test_data, type = 'response')
pred_glm1 <- ifelse(prob_glm1 > 0.5, 1, 0)

vetor_model_glm1 <- metricas(model_glm1, pred_glm1,'Modelo Regressão Logística 1')
print(vetor_model_glm1)
saveRDS(model_glm1, 'models/model_glm1.rds')

# day_of_year_click = NA (sem significância)
# acuracia 0.7201 kappa 0.0091 auc 0.7604


# Versão 2 - Regressão Logística com variáveis selecionadas com Random Forest (Método MeanDecreaseAccuracy)

model_glm2 <- glm(formula = formula_model_b, data = train_data, family = 'binomial')
summary(model_glm2)
prob_glm2 <- predict(model_glm2, test_data, type = 'response')
pred_glm2 <- ifelse(prob_glm2 > 0.5, 1, 0)

vetor_model_glm2 <- metricas(model_glm2, pred_glm2,'Modelo Regressão Logística 2')
print(vetor_model_glm2)
saveRDS(model_glm2, 'models/model_glm2.rds')

# todas as variáveis com ***
# acuracia 0.7200 kappa 0.0091 auc 0.7600


# Versão 3 - Regressão Logística com variáveis selecionadas (Método MeanDecreaseGini)

model_glm3 <- glm(formula = formula_model_c, data = train_data, family = 'binomial')
summary(model_glm3)
prob_glm3 <- predict(model_glm3, test_data, type = 'response')
pred_glm3 <- ifelse(prob_glm3 > 0.5, 1, 0)

vetor_model_glm3 <- metricas(model_glm3, pred_glm3,'Modelo Regressão Logística 3')
print(vetor_model_glm3)
saveRDS(model_glm3, 'models/model_glm3.rds')

# todas as variáveis com ***
# acuracia 0.7216 kappa 0.0092 auc 0.7614


# Versão 4- Regressão Logística com nova composição de variáveis

model_glm4 <- glm(formula = formula_model_d, data = train_data, family = 'binomial')
summary(model_glm4)
prob_glm4 <- predict(model_glm4, test_data, type = 'response')
pred_glm4 <- ifelse(prob_glm4 > 0.5, 1, 0)

vetor_model_glm4 <- metricas(model_glm4, pred_glm4,'Modelo Regressão Logística 4')
print(vetor_model_glm4)
saveRDS(model_glm4, 'models/model_glm4.rds')

# todas as variáveis com ***, exceto os com **
# acuracia 0.7205 kappa 0.0091 auc 0.7609


# Versão 5 - Random forest (a partir da formula_c = melhor resultado com Regressão Logística)

model_rf1 <- rpart(formula_model_c, data = train_data, control = rpart.control(cp= .0005))
summary(model_rf1)
pred_rf1 <- predict(model_rf1, test_data, type = 'class')

vetor_model_rf1 <- metricas(model_rf1, pred_rf1,'Modelo Random Forest 1')
print(vetor_model_rf1)
saveRDS(model_rf1, 'models/model_rf1.rds')
# acuracia 0.8204 kappa 0.0193 auc 0.8577


# Versão 6- Random forest (a partir da formula_c, outros parâmetros)

model_rf2 <- rpart(formula_model_c, data = train_data, control = rpart.control(minsplit=50, cp= .005, maxdepth = 20))
summary(model_rf2)
pred_rf2 <- predict(model_rf2, test_data, type = 'class')

vetor_model_rf2 <- metricas(model_rf2, pred_rf2,'Modelo Random Forest 2')
print(vetor_model_rf2)
saveRDS(model_rf2, 'models/model_rf2.rds')
# acuracia 0.8204 kappa 0.0189 auc 0.8502



# Versão 7 - XGBoost 1 (a partir da formula_c - mesmas variáveis)

features_train <- as.matrix(train_data[,c("app", "diff_time_click", "channel", "count_click_by_ip", "device")])
label_train <- as.matrix(train_data[,c("is_attributed")])

features_test <- as.matrix(test_data[,c("app", "diff_time_click", "channel", "count_click_by_ip", "device")])
label_test <- as.matrix(test_data[,c("is_attributed")])

dtrain <- xgb.DMatrix(data = features_train, label = label_train) # convertendo em matriz densa
dtest <- xgb.DMatrix(data = features_test, label = label_test)

model_xgboost1 <- xgboost(data = dtrain,
                         max.depth = 2,
                         eta = 1,
                         nthread = 2,
                         nrounds = 2,
                         objective = "binary:logistic")

prob_xgboost1 <- predict(model_xgboost1, dtest)
pred_xgboost1 <- as.numeric(prob_xgboost1 > 0.5)

vetor_model_xgboost1 <- metricas(model_xgboost1, pred_xgboost1,'Modelo xGBoost')
print(vetor_model_xgboost1)
xgb.save(model_xgboost1, 'models/model_xgboost1')

# acuracia 0.7795 kappa 0.0148 auc 0.8349


# Versão 8- XGBoost 2 (a partir da formula_c - mesmas variáveis, com variações nos parâmetros)

model_xgboost2 <- xgboost(data = dtrain,
                          max.depth = 3,
                          eta = .9,
                          nthread = 2,
                          nrounds = 2,
                          objective = "binary:logistic")

prob_xgboost2 <- predict(model_xgboost2, dtest)
pred_xgboost2 <- as.numeric(prob_xgboost2 > 0.5)

vetor_model_xgboost2 <- metricas(model_xgboost2, pred_xgboost2,'Modelo xGBoost 2')
print(vetor_model_xgboost2)
xgb.save(model_xgboost2, 'models/model_xgboost2')

# acuracia 0.8186 kappa 0.0182 auc 0.8417


# Versão 9- XGBoost com Cross Validation

params <- list(booster = "gbtree",
               objective = "binary:logistic",
               eta = c(0.5, 0.9, 1),
               max_depth = c(2, 3),
               min_child_weight = 1,
               subsample = c(0.5, 0.6, 1),
               colsample_bytree=1
)

xgbcv <- xgb.cv(params = params, data = dtrain, nfold = 10, nrounds = 100,
                      showsd = T, stratified = T, print_every_n = 10, early_stop_round = 20,
                      eval_metric = "auc", maximize = F)

max_auc <- max(xgbcv$evaluation_log[, 2])
xgbcv$evaluation_log[,2] == max_auc # 100


# max_auc #100 - model training
model_xgbcv <- xgb.train (params = params, data = dtrain, nrounds = 100,
                   watchlist = list(val=dtest,train=dtrain),
                   print_every_n = 20, early_stop_round = 20,
                   maximize = F , eval_metric = "auc")

prob_xgbcv <- predict (model_xgbcv,dtest)
pred_xgbcv <- as.numeric(prob_xgbcv > 0.5)

vetor_model_xgbcv <- metricas(model_xgbcv, pred_xgbcv, 'Modelo xGBoost CV')
print(vetor_model_xgbcv)
xgb.save(model_xgbcv, 'models/model_xgbcv')

# acuracia 0.8222 kappa 0.02 auc 0.8688


# Versão 10- XGBoost com Cross Validation 2

params <- list(booster = "gbtree",
               objective = "binary:logistic",
               eta = c(0.5, 0.9, 1),
               max_depth = c(2, 3),
               min_child_weight = 1,
               subsample = c(0.5, 0.6, 1),
               colsample_bytree=1
)

xgbcv2 <- xgb.cv(params = params, data = dtrain, nfold = 10, nrounds = 200,
                showsd = T, stratified = T, print_every_n = 10, early_stop_round = 20,
                eval_metric = "auc", maximize = F)

max_auc <- max(xgbcv2$evaluation_log[, 2])
xgbcv2$evaluation_log[,2] == max_auc # 200


# max_auc #200 - model training
model_xgbcv2 <- xgb.train (params = params, data = dtrain, nrounds = 200,
                          watchlist = list(val=dtest,train=dtrain),
                          print_every_n = 20, early_stop_round = 20,
                          maximize = F , eval_metric = "auc")

prob_xgbcv2 <- predict (model_xgbcv2,dtest)
pred_xgbcv2 <- as.numeric(prob_xgbcv2 > 0.5)

vetor_model_xgbcv2 <- metricas(model_xgbcv2, pred_xgbcv2, 'Modelo xGBoost CV2')
print(vetor_model_xgbcv2)
xgb.save(model_xgbcv2, 'models/model_xgbcv2')
# acuracia 0.8179 kappa 0.0194 auc 0.8655

## incluir seed e rodar novamente


# Versão 5 - SVM (não usar, pois o treinamento é muito lento)
?svm
model_svm <- svm(formula_model,
                 data = train_data,
                 scale = FALSE,
                 type = 'C-classification',
                 kernel = 'radial')
pred_svm <- predict(model_svm, test_data)
pred_result_svm <- ifelse(pred_svm > 0.5, 1, 0)



## Comparando modelos
compara_modelos <- rbind(vetor_model_glm1, vetor_model_glm2, vetor_model_glm3, vetor_model_glm4,
                         vetor_model_rf, vetor_model_rf2, 
                         vetor_model_xgboost1, vetor_model_xgboost2, vetor_model_xgbcv)
rownames(compara_modelos) <- c(1:9)
colnames(compara_modelos) <- c('Modelo', 'Acuracia', 'Kappa', 'AUC')
compara_modelos <- as.data.frame(compara_modelos)
print(compara_modelos)

ggplot(compara_modelos, aes(x=Modelo, y=Acuracia, fill=Modelo)) + geom_bar(stat='identity')
ggplot(compara_modelos, aes(x=Modelo, y=AUC, fill=Modelo)) + geom_bar(stat='identity')

modelo_final <- 'avaliar melhor resultado'

# vetor resultado MLII pg. 82
# curva gráfico ROC MLI pg. 135
# saveRDS pl 84 e xgbdump p. 282
# testar tune()

# Plot do modelo de melhor acurácia
?prediction
modelo_final <- model_xgbcv
previsoes <- (predict(modelo_final, features_test, type = "class"))
avaliacao <- (prediction(previsoes, test_data$is_attributed))
class(previsoes)
class(test_data$is_attributed)

modelo_final <- model_xgbcv
previsoes <- pred_xgbcv
avaliacao <- prediction(previsoes, test_data$is_attributed) 

# Função para Plot ROC
plot_roc <- function(avaliacao, title){
  perf <- performance(avaliacao, "tpr", "fpr")
  plot(perf,col="black", lty=1, lwd=2,
       main=title, cex.main=0.6, cex.lab=0.8, xaxs="i", yaxs="i")
  abline(0,1, col="red")
  auc <- performance(avaliacao, "auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc, 2)
  legend(0.4,0.4,legend=c(paste0("AUC: ",auc)), cex=0.6, bty="n", box.col="white")
}
# Plot
par(mfrow=c(1,2))
plot_roc(avaliacao, title = "Curva ROC")






