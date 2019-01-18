data_binary_local = read.csv("data_binary.csv", header = TRUE)
names(data_binary_local) = c("top-left-square", "top-middle-square", "top-right-square", "middle-left-square", "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square", "bottom-right-square", "class")

library(randomForest)

binaryRF <- function(data_binary, ntree, mtry, importance, partition) {
    set.seed(9850)
    sample <- sample.int(n = nrow(data_binary), size = floor(partition * nrow(data_binary)), replace = F)
    data_binary_train_split <- data_binary[sample,]
    data_binary_test_split <- data_binary[-sample,]

    data_binary_train <- data_binary_train_split[, 1:ncol(data_binary_train_split) - 1]
    data_binary_train <- as.data.frame(lapply(data_binary_train, as.numeric))
    data_binary_train_source <- factor(data_binary_train_split[, ncol(data_binary_train_split)])

    data_binary_test <- data_binary_test_split[, 1:ncol(data_binary_test_split) - 1]
    data_binary_test <- as.data.frame(lapply(data_binary_test, as.numeric))

    Przewidywanie <- factor(data_binary_test_split[, ncol(data_binary_test_split)])

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

    trainSize <- nrow(data_binary_train)
    testSize <- nrow(data_binary_test)

    result <- list("m" = m, "t" = t, "trainSize" = trainSize, "testSize" = testSize)
    return(result)
}

#params
ntree_local <- 500
mtry_local <- 5
importance_local <- FALSE
partition_local <- 0.75

'
accVec <- vector()
for (i in 1:10) {
    result <- binaryRF(data_binary_local, ntree_local, i, importance_local, partition_local)
    accVec <- c(accVec, result$m)
}
plot(1:10, accVec, xlab = "Wartoœæ mtry", ylab = "Przynale¿noœæ")
'

result <- binaryRF(data_binary_local, ntree_local, mtry_local, importance_local, partition_local)

cat("Informacje o zbiorze:\n")
cat("Rozmiar zbioru ucz¹cego:", result$trainSize, "\n")
cat("Rozmiar zbioru testuj¹cego:", result$testSize, "\n\n")

cat("Parametry dla funkcji:\n")
cat("ntree = ", ntree_local, "\n")
cat("mtry = ", mtry_local, "\n")
cat("importance = ", importance_local, "\n")
cat("shuffle = ", TRUE, "\n\n")

print(result$t)

cat("\nWynik klasyfikcaji =", result$m * 100, "%\n")
