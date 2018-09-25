Machine learning with UCI wine quality dataset

##Load the dataset first
url<- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
white.data<-read.csv(url, header=TRUE, sep=";")
str(white.data)
##When see the structure of our dataser, we get to know that there re 4898 observations of 12 different variables.
##The target variable is named quality. Predictor variables are numeric type and target variables is of integer type.
table(white.data$quality)
##we see that their is a big imbalance here.
##Visualise the data using baseplot
par(mfrow = c(4,3))
for(i in 1:11){plot(white.data[,i],white.data[,"quality"],xlab=names(white.data)[i],ylab="quality",col="red", cex=0.8,cex.lab=1.3);abline(lm(white[, "quality"]~white[,i]),lty=2,lwd=2)}
##Now here a problem with scatterplots is the overlapping of dots when a variable is quantitative. to deal with this use jitter.
##To draw a better scatterplot use jitter function
for(i in 1:11){plot(white.data[,i],jitter(white.data[,"quality"]),xlab=names(white.data)[i],ylab="quality",col="red", cex=0.8,cex.lab=1.3);abline(lm(white.data[, "quality"]~white.data[,i]),lty=2,lwd=2)}
##Now while observing the plots we can see some outliers. we will have to treat the outliers.
##In the predictor variable, the outlier has a value of around 65 which is much larger than the
##second largest value which is around 31. so we have to remove that value from residual.sugar.
maxsugar<-which(white.data$residual.sugar==max(white.data$residual.sugar))
white.data<-white.data[-maxsugar,]
install.packages("caTools")
##Splitting the dataset into training and testing set.
library(caTools)
split<-sample.split(white.data, SplitRatio=0.7)
train<-subset(white.data, split==TRUE)
test<-subset(white.data, split==FALSE)
str(train)
##Now we know that we have to use random forest so at last we calculate the accuracy using confusion matrix.
##So convert the quality variabe to a factor.
train$quality<-as.factor(train$quality)
test$quality<-as.factor(test$quality)

##random forest using cross validation 
install.packages("randomForest")
library(randomForest)
install.packages("caret")
library(caret)
install.packages("e1071")
library(e1071)
set.seed(1234)

##Now we will construct and evaluate the model.
trControl <- trainControl(method = "repeatedcv",number = 5, repeats = 5,search = "grid")

##Search the best value of mtry.
tunegrid<-expand.grid(.mtry=c(1:10))
rf<-train(quality~., data=train, method="rf",metric="Accuracy", tuneGrid=tunegrid,importance=TRUE,ntree=300)
print(rf)
best_mtry<-rf$bestTune$mtry
best_mtry

##Search the best value of maxnodes.
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
    set.seed(1234)
    rf_maxnode <- train(quality~.,
        data = train,
        method = "rf",
        metric = "Accuracy",
        tuneGrid = tunegrid,
        trControl = trControl,
        importance = TRUE,
        maxnodes = maxnodes,
        ntree = 300)
    current_iteration <- toString(maxnodes)
    store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

##Serching for higher values of maxnodes to see if we can get higher accuracy
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(15: 30)) {
    set.seed(1234)
    rf_maxnode <- train(quality~.,
        data = train,
        method = "rf",
        metric = "Accuracy",
        tuneGrid = tunegrid,
        trControl = trControl,
        importance = TRUE,
        maxnodes = maxnodes,
        ntree = 300)
    current_iteration <- toString(maxnodes)
    store_maxnode[[current_iteration]] <- rf_maxnode
}
results_node <- resamples(store_maxnode)
summary(results_mtry)

##Again repeating the above step.
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(30: 50)) {
    set.seed(1234)
    rf_maxnode <- train(quality~.,
        data = train,
        method = "rf",
        metric = "Accuracy",
        tuneGrid = tuneGrid,
        trControl = trControl,
        importance = TRUE,
        maxnodes = maxnodes,
        ntree = 300)
    key <- toString(maxnodes)
    store_maxnode[[key]] <- rf_maxnode
}
results_node <- resamples(store_maxnode)
summary(results_node)

##We get value of max node to be equal to 26. at this max node we get maximum accuracy

##searching the best ntrees.
store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
    set.seed(5678)
    rf_maxtrees <- train(quality~.,
        data = train,
        method = "rf",
        metric = "Accuracy",
        tuneGrid = tuneGrid,
        trControl = trControl,
        importance = TRUE,
        maxnodes = 26,
        ntree = ntree)
    key <- toString(ntree)
    store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)

##ntree =2000: 2000 trees will be trained
##mtry=1: 1 features is chosen for each iteration
##maxnodes = 26: Maximum 26 nodes in the terminal nodes (leaves)

##Now build the model using thhe values above

fit_rf <- train(quality~.,train,method = "rf",metric = "Accuracy",tuneGrid = tuneGrid,trControl = trControl,importance = TRUE,ntree = 2000,maxnodes = 26)
pred<-predict(fit_rf, newdata=test)
table(test$quality, pred)


