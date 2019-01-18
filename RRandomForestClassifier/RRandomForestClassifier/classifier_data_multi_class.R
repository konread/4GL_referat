library(randomForest)

normalize <- function(x) {
    return((x - min(x)) / (max(x) - min(x)))
}

data_multi_class_local = read.csv("data_multi_class.csv", header = TRUE)
names(data_multi_class_local) = c("Cultivar", "Alcohol", "Malic_acid", "Ash", "Alkalinity_ash", "Magnesium", "Phenols", "Flavanoids", "NF_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD", "Proline")

multiClassRF <- function(data_multi_class, ntree, mtry, importance, partition) {
    set.seed(9850)
    sample <- sample.int(n = nrow(data_multi_class), size = floor(partition * nrow(data_multi_class)), replace = F)
    data_multi_class_train_split <- data_multi_class[sample,]
    data_multi_class_test_split <- data_multi_class[-sample,]

    data_multi_class_train <- data_multi_class_train_split[, 2:ncol(data_multi_class_train_split)]
    data_multi_class_train <- as.data.frame(lapply(data_multi_class_train, normalize))
    data_multi_class_train_source <- factor(data_multi_class_train_split[, 1])

    data_multi_class_test <- data_multi_class_test_split[, 2:ncol(data_multi_class_test_split)]
    data_multi_class_test <- as.data.frame(lapply(data_multi_class_test, normalize))
    data_multi_class_test_source <- factor(data_multi_class_test_split[, 1])

    model <- randomForest(data_multi_class_train_source ~ .,
                      data = data_multi_class_train,
                      ntree = ntree,
                      mtry = mtry,
                      importance = importance)

    # Predicting on train set
    predTrain <- predict(model, data_multi_class_train, type = "class")
    # Checking classification accuracy
    table(predTrain, data_multi_class_train_source)

    Resultat <- predict(model, data_multi_class_test, type = "class")
    Przewidywanie <- data_multi_class_test_source
    # Checking classification accuracy
    m <- mean(Resultat == Przewidywanie)
    t <- table(Resultat, Przewidywanie)

    trainSize <- nrow(data_multi_class_train)
    testSize <- nrow(data_multi_class_test)

    result <- list("m" = m, "t" = t, "trainSize" = trainSize, "testSize" = testSize)
    return(result)
}

#params
ntree_local <- 500
mtry_local <- 2
importance_local <- TRUE
partition_local <- 0.75


accVec <- vector()
for (i in 1:10) {
    result <- multiClassRF(data_multi_class_local, ntree_local, i, importance_local, partition_local)
    accVec <- c(accVec, result$m)
}
plot(1:10, accVec, xlab = "Wartoœæ mtry", ylab = "Przynale¿noœæ")

result <- multiClassRF(data_multi_class_local, ntree_local, mtry_local, importance_local, partition_local)

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