data_binary = read.csv("data_binary.csv", header = TRUE)
names(data_binary) = c("top-left-square", "top-middle-square", "top-right-square", "middle-left-square", "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square", "bottom-right-square", "class")

set.seed(9850)
sample <- sample.int(n = nrow(data_binary), size = floor(.5 * nrow(data_binary)), replace = F)
data_binary_train_split <- data_binary[sample,]
data_binary_test_split <- data_binary[-sample,]

data_binary_train <- data_binary_train_split[, 1:ncol(data_binary_train_split) - 1]
data_binary_train <- as.data.frame(lapply(data_binary_train, as.numeric))
data_binary_train_source <- factor(data_binary_train_split[,ncol(data_binary_train_split)])

data_binary_test <- data_binary_test_split[, 1:ncol(data_binary_test_split) - 1]
data_binary_test <- as.data.frame(lapply(data_binary_test, as.numeric))
Przewidywanie <- factor(data_binary_test_split[, ncol(data_binary_test_split)])

library(randomForest)

#params
ntree <- 10
mtry <- 2
importance <- TRUE

model <- randomForest(data_binary_train_source ~ .,
                      data = data_binary_train,
                      ntree = ntree,
                      mtry = mtry,
                      importance = importance)

# Predicting on train set
predTrain <- predict(model, data_binary_train, type = "class")
# Checking classification accuracy
table(predTrain, data_binary_train_source)

# Predicting on train set
Resultat <- predict(model, data_binary_test, type = "class")
# Checking classification accuracy
m <- mean(Resultat == Przewidywanie)
t <- table(Resultat, Przewidywanie)

cat("Parametry dla funkcji:\n")
cat("ntree = ", ntree, "\n")
cat("mtry = ", mtry, "\n")
cat("importance = ", importance, "\n")
cat("shuffle = ", TRUE, "\n\n")

print(t)

cat("\nWynik klasyfikcaji =", m * 100, "%\n")











'
set.seed(9850)
randomValues <- runif(nrow(data_binary))
shuffled_data_binary = data_binary[order(randomValues),]

# 958
dmsTrainning <- shuffled_data_binary[1:440, 1:ncol(shuffled_data_binary) - 1]
dmsTrainning <- as.data.frame(lapply(dmsTrainning, as.numeric))
dmsTrainningSource <- factor(shuffled_data_binary[1:440, ncol(shuffled_data_binary)])

dmsTesting <- shuffled_data_binary[441:958, 1:ncol(shuffled_data_binary) - 1]
dmsTesting <- as.data.frame(lapply(dmsTesting, as.numeric))
dmsTestingSource <- factor(shuffled_data_binary[441:958, ncol(shuffled_data_binary)])

library(randomForest)
model <- randomForest(dmsTrainningSource ~ ., data = dmsTrainning, importance = TRUE)

# Predicting on train set
predTrain <- predict(model, dmsTrainning, type = "class")
# Checking classification accuracy
table(predTrain, dmsTrainningSource)

#predValid <- predict(model, data_binary_test_num, type = "class")
# Checking classification accuracy
#mean(predValid == data_binary_test_source)
#table(predValid, dmsTestingSource)



set.seed(9850)
sample <- sample.int(n = nrow(data_binary), size = floor(.5 * nrow(data_binary)), replace = F)
data_binary_train <- factor(data_binary[sample,])
data_binary_test <- factor(data_binary[-sample,])
print(nrow(data_binary_test))
# convert to numeric
data_binary_train_num <- as.data.frame(lapply(data_binary_train[1:ncol(data_binary_train) - 1], as.numeric))
data_binary_test_num <- as.data.frame(lapply(data_binary_test[1:ncol(data_binary_test) - 1], as.numeric))

data_binary_train_source <- factor(data_binary_train[ncol(data_binary_train)])
data_binary_test_source <- data_binary_test[ncol(data_binary_test)]

library(randomForest)'