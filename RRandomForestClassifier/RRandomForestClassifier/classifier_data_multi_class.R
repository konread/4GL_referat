data_multi_class = read.csv("data_multi_class.csv", header = TRUE)

names(data_multi_class) = c("Cultivar", "Alcohol", "Malic_acid", "Ash", "Alkalinity_ash", "Magnesium", "Phenols", "Flavanoids", "NF_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD", "Proline")

set.seed(9850)
randomValues <- runif(nrow(data_multi_class))
shuffled_data_multi_class = data_multi_class[order(randomValues),]

normalize <- function(x) {
    return((x - min(x)) / (max(x) - min(x)))
}

shuffled_data_multi_class_normalized <- as.data.frame(lapply(shuffled_data_multi_class[, 2:ncol(shuffled_data_multi_class)], normalize))

dmsTrainning <- shuffled_data_multi_class_normalized[1:122, 2:ncol(shuffled_data_multi_class_normalized)]
dmsTesting <- shuffled_data_multi_class_normalized[123:177, 2:ncol(shuffled_data_multi_class_normalized)]
dmsTrainningSource <- factor(shuffled_data_multi_class[1:122, 1])
Przewidywanie <- factor(shuffled_data_multi_class[123:177, 1])

library(randomForest)

#params
ntree <- 10
mtry <- 2
importance <- TRUE

model <- randomForest(dmsTrainningSource ~ .,
                      data = dmsTrainning,
                      ntree = ntree,
                      mtry = mtry,
                      importance = importance)

# Predicting on train set
predTrain <- predict(model, dmsTrainning, type = "class")
# Checking classification accuracy
table(predTrain, dmsTrainningSource)

Resultat <- predict(model, dmsTesting, type = "class")
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
