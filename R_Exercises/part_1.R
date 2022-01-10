setwd('C:/Users/Giorgos/Desktop/ergasies_metaptxiakwn/ergasia filippaki_(predictive analytics)/ergasia_examinou/ergasia_mfilip_2021/ergasia_mfilip_2021/dataset_merous_1_kai_2')
#1
library(MASS) #pima.te - diabetes dataset is in MASS library
data <- Pima.te?set.seed(789) 
training = sample(nrow(data),265,replace = FALSE)
train = data[training, ]
test = data[-training, ]
model.all <- glm(type ~.,data = train,family = "binomial")
summary(model.all)
model2 <- glm(type ~ npreg+glu+skin+bmi+ped+age, data = train,f?mily 
              = "binomial")
summary(model2)
model3 <- glm(type ~ npreg+glu+skin+bmi+ped, data = train,family = 
                "binomial")
summary(model3)
model4 <- glm(type ~ npreg+glu+bmi+ped, data = train,family = 
                "binomial")
sum?ary(model4)
newdata = read.csv('meros1_exc1_new_data.csv')
pred_new <- predict(model4, newdata = newdata, type = "response")
pred_new
# 2
library(cluster)
df2 = read.csv('decathlon.csv',sep=';')
df2_std <- as.data.frame( scale( df2[,2:11] ) )
summary( df2_?td )
dist_mat <- dist( df2_std, method = 'euclidean')
hclust_avg <- hclust(dist_mat, method = 'average')
plot(hclust_avg)
# prune(hclust_avg)
# cutree(hclust_avg, k = 1:5)
# draw.clust (prune.clust (agnes (df2_std), k=6))
# kmeans clustering
cluster_kmeans? <- kmeans(df2_std, 7)
cluster_kmeans7
# 3
# 3.1
library(astsa)
library(TTR)
ts = cmort
model_ts <- arima(ts,c(2,0,0))
ts_forecast <- predict(model_ts, n.ahead = 4)
msft_forecast_values <- ts_forecast$pred
msft_forecast_se <- ts_forecast$se
lower_bound = m?ft_forecast_values - 1.96*msft_forecast_se
upper_bound = msft_forecast_values + 1.96*msft_forecast_se
confidence_int = cbind(lower_bound, upper_bound)
msft_forecast_values
confidence_int
# 3.2
ts = arima.sim( n = 500, list(ar = 0.9, ma = -0.9, sd = 1) )
ac?(ts, main = 'ARMA(1,1) ACF')
pacf(ts, main = 'ARMA(1,1) PACF')
model_ts <- arima(ts,c(1,0,1))
model_ts$coef
# 3.3
ts = oil
fit = auto.arima(ts)
fit
library(aTSA)
ts.diag(estimate(ts,p=1,d=1,q=3))
# 3.4
ts = globtemp
fit = auto.arima(ts)
fit
model_ts <- ari?a(ts,c(1,1,3))
library(aTSA)
ts.diag(estimate(ts,p=1,d=1,q=3))
ts_forecast <- predict(model_ts, n.ahead = 10)
ts_forecast$pred
# 3.5
ts = chicken
fit = auto.arima(ts)
fit
model_ts <- arima(ts,c(2,1,1))
library(aTSA)
ts.diag(estimate(ts,p=2,d=1,q=1))
ts_for?cast <- predict(model_ts, n.ahead = 12)
ts_forecast$pred
# 4
dat = c(1,3,6,4,
        2,2,7,5,
        2,4,8,5,
        1,3,8,6)
ts <- ts(dat, start=c(2016, 1), end=c(2019, 4), frequency=4)
plot.ts(ts)
# 4.I
library(fpp) 
ts_comp <- decompose(ts)
ts_SA <- ?s - ts_comp$seasonal
plot.ts(ts_SA)
ts_comp$figure # Seasonality Indexes
# 4.II

x <- (1:length(ts))
ts.plot(ts)
plot(x,ts)
lines(predict(lm(ts~x)),col='green')
# 4.III
decomp <- stl(ts, s.window="periodic")
plot(decomp)
# 5
dat5 = c(37.44, 44.14, 46.25, 4?.99, 51.84, 49.10, 58.56, 58.02, 
         70.28)
ts5 <- ts(dat5, start=c(2011), end=c(2019), frequency=1)
plot.ts(ts5)
x5 <- (1:length(ts5))
ts.plot(ts5)
plot(x5,ts5)
lines(predict(lm(ts5~x5)),col='green')
# 6
df6 = read.csv('FXUSDCAD.csv')
library(xts)
t?6 <- xts(df6[,-1], order.by=as.Date(df6[,1], "%d/%m/%Y"))
# 6.I
plot.ts(ts6)
# 6.II
x6 <- (1:length(ts6))
plot.ts(ts6)
plot(x6,ts6)
lines(predict(lm(ts6~x6)),col='green')
# 6.III
# Quadratic trend
plot.ts(ts6)
plot(x6,ts6)
lines( predict( lm( ts6 ~ x6 + I(?6^2) ) ), col='green' )
# 6.IV)
dts6 = diff(ts6, differences = 1)
plot.ts(dts6)
d2ts6 = diff(ts6, differences = 2)
plot.ts(d2ts6)
# 6.V)
library(xts)
library("forecast")
model1 <- tslm( ts(ts6) ~ trend )
model1_f22 = forecast(model1, h=22)
model1_f22
plot(?odel1_f22)
model2 <- tslm( ts(ts6) ~ trend + I(trend^2) )
model2_f22 = forecast(model2, h=22)
model2_f22
plot(model2_f22)
model3 <- tslm( ts(dts6) ~ 1 )
model3_f22 = forecast(model3, h=22)
model3_f22
plot(model3_f22)
model4 <- tslm( ts(d2ts6) ~ 1 )
model4_?22 = forecast(model4, h=22)
model4_f22
plot(model4_f22)
# 6.VI)
df6$dates = as.Date(df6$date)
df6$months = months(df6$dates)
df6$Jan = ifelse(df6$months == "January", 1, 0)
model6VI <- tslm( ts(df6$FXUSDCAD) ~ df6$Jan)
model6VI
# 6.VII)
model6VII <- tslm( ?s(ts6) ~ trend + I(trend^2) + df6$Jan)
summary(model6VII)
# 6.VIII)
ts6_lag = lag(ts6, 1)
plot(as.vector(ts6_lag), df6$FXUSDCAD, main="FX on lag(FX)", 
     xlab="lag(FX)", ylab="FX", pch=19)
# 6.IX)
cor(as.vector(ts6_lag), df6$FXUSDCAD, method = "pearson"? use = 
      "complete.obs")
acf(ts6, lag = 1)
# 6.X)
AC <- function(y, k) {
  y0 <- y
  n0 <- NROW(y)
  k <- 2
  y <- y0[(k+1):n0]
  x <- y0[1:(n0-k)]
  n <- NROW(x)
  sx = sum((x-sum(x)/n)^2) 
  sy = sum((y-sum(y)/n)^2) 
  corr = (sum((x-sum(x)/n)*(y-su?(y)/n)))/(sx*sy)
  return(corr)
}
AC( as.vector(ts6), 2 )
# 6.????
model_ts_AR1 <- arima(ts6,c(1,0,0))
model_ts_AR1
model_ts_MA1 <- arima(ts6,c(0,0,1))
model_ts_MA1

# 6.??????
model_ts_ARMA212 <- arima(ts6,c(2,1,2))
model_ts_ARMA212
accuracy(model_ts_AR1)
accuracy(model_ts_MA1)
accuracy(model_ts_ARMA212)

