setwd('C:/Users/Giorgos/Desktop/ergasies_metaptxiakwn/ergasia filippaki_(predictive analytics)/ergasia_examinou/ergasia_mfilip_2021/ergasia_mfilip_2021/dataset_merous_1_kai_2')


# 1)
# install.packages("rpart")
library("rpart")
# Kyphosis
# a factor with ?evels absent present indicating if a kyphosis (a type of deformation) was present after the operation.
# Age
# in months
# Number
# the number of vertebrae involved
# Start
# the number of the first (topmost) vertebra operated on.
df = kyphosis
summary(df)?# Boxplot of the variable NUMBER
boxplot(df$Number, main="The Variable: Number",  ylab="The number of vertebrae involved")

outliers = df$Number[df$Number>8]
outliers
which( df$Number %in% outliers)

plot(df$Number, df$Age)
identify(df$Number, df$Age)


# ?)
df = read.csv("capital.csv", sep = ';')
summary(df)
#df$gender <- as.factor(df$gender)

# 2i) ?????????????? ???????????????? ????????????????????
xtabs(balance ~ gender , data=df)
with(df, discretePlot(gender, scale="frequency"))
barplot( table(df$gender) )
prop.table(table(df$gender))

# Simple Pie Chart
bal_gender <- c(173191, 52913)
lbls <- c("Male", "Female")
pie(bal_gender, labels = lbls, main?"Balance by gender")

# 2ii) Boxplot by Group
boxplot(balance ~ gender, data=df)

# 2iii)
library(psych)
describeBy(df$balance, df$gender)

summary(df$balance)
library(pastecs)
stat.desc(df$balance) 

# 2iv)
library("car")
qqPlot(df$balance)


# 3)
data(mt?ars)
head(mtcars)
summary(mtcars)
str(mtcars)
#df = read.csv("mtcars.csv", sep = ',')

t.test( mpg ~ am, data = mtcars,conf.level = 0.95)


# 4)
df <- read.delim("OctopusF.txt")
summary(df$Weight)
library(pastecs)
stat.desc(df$Weight) 

hist(df$Weight)

li?rary("car")
qqPlot(df$Weight)

# Calculate the mean and standard error
l.model <- lm(df$Weight ~ 1, df)
# Calculate the confidence interval
confint(l.model, level=0.95)


# 5)
library(MASS)
df = survey
contigency_table = table(df$Smoke, df$Exer)
contigency?table
chisq.test(contigency_table)


# 6)
# Loading
library("readxl")
# xls files
df <- read_excel("Concrete_Data.xls")
str(df)
df <-scaleddata<-scale(df)
# Training and Test Data
set.seed(653)
df <- df[sample(nrow(df)), ]
trainset <- df[1:721, ]
testset <? df[722:1030, ]
#Neural Network
library(neuralnet)
nn <- neuralnet(Concrete ~ Cement + Slag + Ash + Water + Superplasticizer + CoarseAggregate + FineAggregate + Age, data=trainset, hidden=c(4,1), linear.output=FALSE, threshold=0.5)
nn$result.matrix
plot(nn?
#Test the resulting output
temp_test <- subset(testset, select = c("Cement", "Slag", "Ash", "Water", "Superplasticizer", "CoarseAggregate", "FineAggregate", "Age"))
head(temp_test)
nn.results <- compute( nn, temp_test )
results <- data.frame( actual = tes?set$Concrete, prediction = nn.results$net.result )
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)


# 7)
library(ggplot2)

df <- read.delim("faithfull.txt")
str(d?)
summary(df$Weight)

reg.lm <- lm(eruptions ~ waiting, data = df)
summary(reg.lm)

x80.dat <- data.frame(waiting = 80)
predict(reg.lm, newdata = x80.dat, interval = 'confidence')
predict(reg.lm, newdata = x80.dat, interval = 'prediction')

eruption.res = ?esid(reg.lm)
plot(df$waiting, eruption.res, ylab="Residuals", xlab="Waiting Time", main="Eruptions") 
abline(0, 0)   

library("car")
qqPlot(eruption.res)


# 8) 
library(MASS)
data(stackloss)
str(stackloss)
stackloss.lm <- lm(stack.loss ~ Air.Flow + Water?Temp + Acid.Conc., data = stackloss)
summary(stackloss.lm)
new_data = data.frame(Air.Flow=72, Water.Temp=20, Acid.Conc.=85)
predict(stackloss.lm, new_data)
predict(stackloss.lm, newdata = new_data, interval = 'confidence')
predict(stackloss.lm, newdata = n?w_data, interval = 'prediction')


# 9)
# Loading
library("readxl")
# xls files
df <- read_xlsx("market.xlsx")
market.lm <- lm(Sales ~ Preis + Costs + Arrivals, data = df)
summary(market.lm)
# x-correlation
cor(df[,2:4])
# Interactions
market_inter.lm <- l?(Sales ~ Preis + Costs + Arrivals + Preis*Costs + Preis*Arrivals + Costs*Arrivals, data = df)
summary(market_inter.lm)
# Standardized Residuals
market.stdres = rstandard(market.lm)
summary(market.stdres)
# Residuals
market.res = residuals(market.lm)
summar?(market.res)
# Boxplot()
# 2ii) Boxplot by Group
boxplot(market.stdres, data=df)
# QQ-plot
qqPlot(market.stdres)
plot(market.lm)


lmtest::bptest(market.lm)  # Breusch-Pagan test


# 10)
# I)
df = read.csv("insurance.csv", sep = ',')
df$sex = as.factor(df$?ex)
df$smoker = as.factor(df$smoker)
df$region = as.factor(df$region)
df <- sapply(df, unclass)
# II)
library(psych)   
library(psychTools)  #additional tools and data are here
describe(df)  #basic descriptive statistics
corr.test(df)
# III)
setCor(y = 7,x?1:6, data = df)


#11)
# Loading
library("readxl")
# xls files
df <- read_excel("mf.xls")

# I)
df$amount_cut = cut(df$`Dollar Claim Amount`,3)

tb1 = table(df$amount_cut, df$Shift)
chi2_1 = chisq.test(tb1)
chi2_1$p.value

tb2 = table(df$amount_cut, df$`Co?plaint Code`)
chi2_2 = chisq.test(tb2)
chi2_2$p.value

tb3 = table(df$amount_cut, df$`Manufacturing Plant`)
chi2_3 = chisq.test(tb3)
chi2_3$p.value

tb4 = table(df$`Complaint Code`, df$Shift)
chi2_4 = chisq.test(tb4)
chi2_4$p.value

tb5 = table(df$`Complai?t Code`, df$`Manufacturing Plant`)
chi2_5 = chisq.test(tb5)
chi2_5$p.value


# II)
tab1 = table(df$`Complaint Code`, df$`Manufacturing Plant`)
chi2 = chisq.test(tab1)
chi2$expected
chi2$p.value

# III)
df1 = df[,c(1,4)]
df1 = df1[ df1$`Manufacturing Plant`?< 3,  ]
t.test(df1$`Dollar Claim Amount` ~ df1$`Manufacturing Plant`, data = df1)

#12)
#????)
library("fpc")
library("dbscan")
x <- c(1, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 8, 9, 9) 
y <- c(5, 6, 2, 3, 5, 6, 7, 8, 3, 5, 7, 8, 4, 6, 8, 4, 5, 6, 7, 4, 3, 2, 5) 
data <- data.frame(x=x, y=y) 
db <- fpc::dbscan(data , eps = 1, Min?ts = 3)
plot(db, data , main = "DBSCAN")

#??????)
plot_fun <- function(data, nc){
  wss <- (nrow(data)-1) * sum(apply(data,2,var))
  for(i in 2:nc){wss[i] <- sum(kmeans(data,centers=i)$withinss)}
  plot(1:nc, wss, type = "b" ,xlab="Number of Clusters",ylab="Total within Sum of Square")}

plot_fun(data, ?row(data)-1)

kc <- kmeans(data, 4)
plot(data, col=kc$cluster)


#13)
x <- c(0.4005, 0.2148, 0.3457, 0.2652, 0.0789, 0.4548)
y <- c(0.5306, 0.3854, 0.3156, 0.1875, 0.4139, 0.3022)
data <- cbind(x,y)
#?????????????? ?????? ???????????????????? ????????????????
dist(data, method = "euclidean", diag = TRUE, upper = TRUE)
#?? ??????????????
hc <- hclust(dist(data ), method = "single" )
plot(hc,hang = -1)

#?? ??????????????
hc <- hclust(dist(data), method = "complete")
plot(hc, hang = -1)

