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
