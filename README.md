# Data_Science_Machine_Learning

---
title: "01 - Introduction to Machine Learning Comprehension Check"
author: "Alessandro Corradini - Harvard Data Science Professional"
output:
  html_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

<br/>

## **Introduction to Machine Learning**
### **Question 1**

"Turbochamp" was one of the first algorithms written by famed British computer scientist, mathematician and crypto-analyst Alan Turing in the late 1940s. The artificial intelligence program was based on programmable rules derived from theory or first principles and could 'think' two moves ahead.

True or False: A key feature of machine learning is that the algorithms are built on data.

- True [X]
- False

<br/>

### **Question 2**

True or False: In machine learning, we build algorithms that take feature values (X) and train a model using known outcomes (Y) that is then used to predict outcomes when presented with features without known outcomes.

- True [X]
- False

---
title: "02 - Machine Learning Basics Comprehension Check"
author: "Alessandro Corradini - Harvard Data Science Professional"
output:
  html_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

<br/>

## **Basics of Evaluating Machine Learning Algorithms**
### **Question 1**

For each of the following, indicate whether the outcome is continuous or categorical.

- Digit reader: categorical
- Movie recommendation ratings: continuous
- Spam filter: categorical
- Number of hospitalizations: continuous
- Siri: categorical

<br/>

### **Question 2**

How many features are available to us for prediction in the mnist digits dataset?

```{r, include=TRUE}
library(dslabs)
mnist <- read_mnist()
ncol(mnist$train$images)
```

<br/>

### **Question 3**

In the digit reader example, the outcomes are stored here: ```y <- mnist$train$labels.
Do the following operations have a practical meaning?
```
y[5] + y[6]
y[5] > y[6]
```
- Yes, because 9 + 2 = 11 and 9 > 2. [X]
- No, because y is not a numeric vector.
- No, because 11 is not one digit, it is two digits.
- No, because these are labels representing a category, not a number. A 9 represents a type of digit, not the number 9.
<br/>
## **Confusion Matrix**
The following questions all ask you to work with the dataset described below.
The ```reported_heights``` and heights datasets were collected from three classes taught in the Departments of Computer Science and Biostatistics, as well as remotely through the Extension School. The Biostatistics class was taught in 2016 along with an online version offered by the Extension School. On 2016-01-25 at 8:15 AM, during one of the lectures, the instructors asked student to fill in the sex and height questionnaire that populated the reported_height dataset. The online students filled out the survey during the next few days, after the lecture was posted online. We can use this insight to define a variable which we will call type, to denote the type of student, inclass or online.
The code below sets up the dataset for you to analyze in the following exercises:
```{r, include=TRUE}
library(dslabs)
library(dplyr)
library(lubridate)
data("reported_heights")
dat <- mutate(reported_heights, date_time = ymd_hms(time_stamp)) %>%
  filter(date_time >= make_date(2016, 01, 25) & date_time < make_date(2016, 02, 1)) %>%
  mutate(type = ifelse(day(date_time) == 25 & hour(date_time) == 8 & between(minute(date_time), 15, 30), "inclass","online")) %>%
  select(sex, type)
y <- factor(dat$sex, c("Female", "Male"))
x <- dat$type
```

<br/>

### **Question 1**

What is the propotion of females in class and online? (That is, calculate the proportion of the in class students who are female and the proportion of the online students who are female.)

```{r, include=TRUE}
dat %>% group_by(type) %>% summarize(prop_female = mean(sex == "Female"))
```

<br/>

### **Question 2**

If you used the type variable to predict sex, what would the prediction accuracy be?

```{r, include=TRUE}
y_hat <- ifelse(x == "online", "Male", "Female") %>% 
      factor(levels = levels(y))
mean(y_hat==y)
```

<br/>

### **Question 3**

Write a line of code using the table function to show the confusion matrix, assuming the prediction is y_hat and the truth is y.

```{r, include=TRUE}
table(y_hat, y)
```

<br/>

### **Question 4**

What is the sensitivity of this prediction?

```{r, include=TRUE}
library(caret)
sensitivity(y_hat, y)
```

<br/>

### **Question 5**

What is the specificity of this prediction?

```{r, include=TRUE}
library(caret)
specificity(y_hat, y)
```

<br/>

### **Question 6**

What is the prevalence (% of females) in the dat dataset defined above?
```{r, include=TRUE}
mean(y == "Female") 
```

<br/>

## **Practice with Machine Learning**

We will practice building a machine learning algorithm using a new dataset, iris, that provides multiple predictors for us to use to train. To start, we will remove the setosa species and we will focus on the versicolor and virginica iris species using the following code:

```{r, include=TRUE}
library(caret)
data(iris)
iris <- iris[-which(iris$Species=='setosa'),]
y <- iris$Species
```

The following questions all involve work with this dataset.

<br/>

### **Question 1**

First let us create an even split of the data into train and test partitions using createDataPartition. The code with a missing line is given below:

```{r, include=TRUE}
set.seed(2)
test_index <- createDataPartition(y,times=1,p=0.5,list=FALSE)
test <- iris[test_index,]
train <- iris[-test_index,]
```

Which code should be used in place of # line of code above?

- ```test_index <- createDataPartition(y,times=1,p=0.5)```
- ```test_index <- sample(2,length(y),replace=FALSE)```
- ```test_index <- createDataPartition(y,times=1,p=0.5,list=FALSE)``` [X] 
- ```test_index <- rep(1,length(y))```

<br/>

### **Question 2**

Next we will figure out the singular feature in the dataset that yields the greatest overall accuracy. You can use the code from the introduction and from Q1 to start your analysis.

Using only the train iris data set, which of the following is the singular feature for which a smart cutoff (simple search) yields the greatest overall accuracy?

```{r, include=TRUE}
func <- function(x){
	rangedValues <- seq(range(x)[1],range(x)[2],by=0.1)
	sapply(rangedValues,function(i){
		y_hat <- ifelse(x>i,'virginica','versicolor')
		mean(y_hat==train$Species)
	})
}
predictions <- apply(train[,-5],2,func)
sapply(predictions,max)	
```

- Sepal.Length
- Sepal.Width
- Petal.Length [X]
- Petal.Width

<br/>

### **Question 3**

Using the smart cutoff value calculated on the training data, what is the overall accuracy in the test data?

```{r, include=TRUE}
predictions <- func(train[,3])
rangedValues <- seq(range(train[,3])[1],range(train[,3])[2],by=0.1)
cutoffs <-rangedValues[which(predictions==max(predictions))]
y_hat <- ifelse(test[,3]>cutoffs[1],'virginica','versicolor')
mean(y_hat==test$Species) 
```

<br/>

### **Question 4**

Notice that we had an overall accuracy greater than 96% in the training data, but the overall accuracy was lower in the test data. This can happen often if we overtrain. In fact, it could be the case that a single feature is not the best choice. For example, a combination of features might be optimal. Using a single feature and optimizing the cutoff as we did on our training data can lead to overfitting.

Given that we know the test data, we can treat it like we did our training data to see if the same feature with a different cutoff will optimize our predictions.

Which feature best optimizes our overall accuracy?

- Sepal.Length
- Sepal.Width
- Petal.Length
- Petal.Width [X]

<br/>

### **Question 5**

Now we will perform some exploratory data analysis on the data.

Notice that ```Petal.Length``` and ```Petal.Width``` in combination could potentially be more information than either feature alone.

Optimize the combination of the cutoffs for ```Petal.Length``` and ```Petal.Width``` in the ```train``` data and report the overall accuracy when applied to the test dataset. For simplicity, create a rule that if either the length OR the width is greater than the length cutoff or the width cutoff then virginica or versicolor is called. (Note, the F1 will be similarly high in this example.)

What is the overall accuracy for the test data now?

```{r, include=TRUE}
library(caret)
library(dplyr)
data(iris)
iris <- iris[-which(iris$Species=='setosa'),]
y <- iris$Species
plot(iris,pch=21,bg=iris$Species)
set.seed(2)
test_index <- createDataPartition(y,times=1,p=0.5,list=FALSE)
test <- iris[test_index,]
train <- iris[-test_index,]
petalLengthRange <- seq(range(train[,3])[1],range(train[,3])[2],by=0.1)
petalWidthRange <- seq(range(train[,4])[1],range(train[,4])[2],by=0.1)
cutoffs <- expand.grid(petalLengthRange,petalWidthRange)
id <- sapply(seq(nrow(cutoffs)),function(i){
	y_hat <- ifelse(train[,3]>cutoffs[i,1] | train[,4]>cutoffs[i,2],'virginica','versicolor')
	mean(y_hat==train$Species)
	}) %>% which.max
optimalCutoff <- cutoffs[id,] %>% as.numeric
y_hat <- ifelse(test[,3]>optimalCutoff[1] & test[,4]>optimalCutoff[2],'virginica','versicolor')
mean(y_hat==test$Species)
```
<br/>

## **Conditional Probabilities Review**
### **Question 1**

In a previous module, we covered Bayes' theorem and the Bayesian paradigm. Conditional probabilities are a fundamental part of this previous covered rule.

$\ P(A|B) = P(B|A)\frac{P(A)}{P(B)}$

We first review a simple example to go over conditional probabilities.

Assume a patient comes into the doctor’s office to test whether they have a particular disease.

- The test is positive 85% of the time when tested on a patient with the disease (high sensitivity): $\ P(\text{disease}) = 0.02$
- The test is negative 90% of the time when tested on a healthy patient (high specificity): $\ P(\text{disease}) = 0.02$
- The disease is prevalent in about 2% of the community: $\ P(\text{disease}) = 0.02$

Using Bayes' theorem, calculate the probability that you have the disease if the test is positive.

$\ P(\text{disease} | \text{test}+) = P(\text{test}+ | \text{disease}) \times \frac{P(\text{disease})}{P(\text{test}+)} = \frac{P(\text{test}+ | \text{disease})P(\text{disease})}{P(\text{test}+ | \text{disease})P(\text{disease})+P(\text{test}+ | \text{healthy})P(\text{healthy})]} = \frac{0.85 \times 0.02}{0.85 \times 0.02 + 0.1 \times 0.98} = 0.1478261$

The following 4 questions (Q2-Q5) all relate to implementing this calculation using R.

We have a hypothetical population of 1 million individuals with the following conditional probabilities as described below:

- The test is positive 85% of the time when tested on a patient with the disease (high sensitivity): $\ P(\text{test} + | \text{disease}) = 0.85$
- The test is negative 90% of the time when tested on a healthy patient (high specificity): $\ P(\text{test} - | \text{heathy}) = 0.90$
The disease is prevalent in about 2% of the community: 
- Here is some sample code to get you started: $\ P(\text{disease}) = 0.02$

```{r, include=TRUE}
set.seed(1)
disease <- sample(c(0,1), size=1e6, replace=TRUE, prob=c(0.98,0.02))
test <- rep(NA, 1e6)
test[disease==0] <- sample(c(0,1), size=sum(disease==0), replace=TRUE, prob=c(0.90,0.10))
test[disease==1] <- sample(c(0,1), size=sum(disease==1), replace=TRUE, prob=c(0.15, 0.85))
```

<br/>

### **Question 2**

What is the probability that a test is positive?

```{r, include=TRUE}
mean(test)
```

<br/>

### **Question 3**

What is the probability that an individual has the disease if the test is negative?

```{r, include=TRUE}
mean(disease[test==0])
```

<br/>

### **Question 4**

What is the probability that you have the disease if the test is positive?
Remember: calculate the conditional probability the disease is positive assuming a positive test.

```{r, include=TRUE}
mean(disease[test==1]==1)
```

<br/>

### **Question 5**

If the test is positive, what is the relative risk of having the disease?
First calculate the probability of having the disease given a positive test, then normalize it against the disease prevalence.

```{r, include=TRUE}
mean(disease[test==1]==1)/mean(disease==1)
```

<br/>

## **Conditional Probabilities Practice**
### **Question 1**

We are now going to write code to compute conditional probabilities for being male in the heights dataset. Round the heights to the closest inch. Plot the estimated conditional probability $\ P(x) = \mbox{Pr}(\mbox{Male} | \mbox{height}=x)$ for each $\ x$.

Part of the code is provided here:

```{r, include=TRUE}
library(dslabs)
data("heights")
heights %>% 
	mutate(height = round(height)) %>%
	group_by(height) %>%
	summarize(p = mean(sex == "Male")) %>%
	qplot(height, p, data =.)
```

Which of the following blocks of code can be used to replace MISSING CODE to make the correct plot?

```
heights %>% 
	group_by(height) %>%
	summarize(p = mean(sex == "Male")) %>%
```
```
heights %>% 
	mutate(height = round(height)) %>%
	group_by(height) %>%
	summarize(p = mean(sex == "Female")) %>%
```
```
heights %>% 
	mutate(height = round(height)) %>%
	summarize(p = mean(sex == "Male")) %>%
```
```
heights %>% 
	mutate(height = round(height)) %>%
	group_by(height) %>%
	summarize(p = mean(sex == "Male")) %>% [X]
```

<br/>

### **Question 2**

In the plot we just made in Q1 we see high variability for low values of height. This is because we have few data points. This time use the quantile (\ 0.1,0.2,\dots,0.9 \)and the ```cut``` function to assure each group has the same number of points. Note that for any numeric vector ```x```, you can create groups based on quantiles like this: ```cut(x, quantile(x, seq(0, 1, 0.1)), include.lowest = TRUE)```.

Part of the code is provided here:

```{r, include=TRUE}
ps <- seq(0, 1, 0.1)
heights %>% 
  mutate(g = cut(height, quantile(height, ps), include.lowest = TRUE)) %>%
	group_by(g) %>%
	summarize(p = mean(sex == "Male"), height = mean(height)) %>%
	qplot(height, p, data =.)
```

Which of the following lines of code can be used to replace MISSING CODE to make the correct plot?

```
mutate(g = cut(male, quantile(height, ps), include.lowest = TRUE)) %>%
```
```
mutate(g = cut(height, quantile(height, ps), include.lowest = TRUE)) %>% [X]
```
```
mutate(g = cut(female, quantile(height, ps), include.lowest = TRUE)) %>%
```
```
mutate(g = cut(height, quantile(height, ps))) %>%
```

<br/>

### **Question 3**
You can generate data from a bivariate normal distrubution using the MASS package using the following code.

```{r, include=TRUE}
Sigma <- 9*matrix(c(1,0.5,0.5,1), 2, 2)
dat <- MASS::mvrnorm(n = 10000, c(69, 69), Sigma) %>%
	data.frame() %>% setNames(c("x", "y"))
```	
And make a quick plot using 

```{r, include=TRUE}
plot(dat)
```

Using an approach similar to that used in the previous exercise, let's estimate the conditional expectations and make a plot. Part of the code has been provided for you:

```{r, include=TRUE}
ps <- seq(0, 1, 0.1)
dat %>% 
	mutate(g = cut(x, quantile(x, ps), include.lowest = TRUE)) %>%
group_by(g) %>%
summarize(y = mean(y), x = mean(x)) %>%	
	qplot(x, y, data =.)
```

Which of the following blocks of code can be used to replace MISSING CODE to make the correct plot?

```
mutate(g = cut(x, quantile(x, ps), include.lowest = TRUE)) %>%
group_by(g) %>%
summarize(y = mean(y), x = mean(x)) %>%
```
```
mutate(g = cut(x, quantile(x, ps))) %>%
group_by(g) %>%
summarize(y = mean(y), x = mean(x)) %>%
```
```
mutate(g = cut(x, quantile(x, ps), include.lowest = TRUE)) %>%
summarize(y = mean(y), x = mean(x)) %>%
```
```
mutate(g = cut(x, quantile(x, ps), include.lowest = TRUE)) %>%
group_by(g) %>%
summarize(y =(y), x =(x)) %>%
```

---
title: "03 - Linear Regression for Prediction, Smoothing, and Working with Matrices Comprehension Check"
author: "Alessandro Corradini - Harvard Data Science Professional"
output:
  pdf_document: default
  html_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

<br/>

## **Linear Regression**
### **Question 1**

Create a data set using the following code:

```{r, include=TRUE}
set.seed(1)
n <- 100
Sigma <- 9*matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)
dat <- MASS::mvrnorm(n = 100, c(69, 69), Sigma) %>%
	data.frame() %>% setNames(c("x", "y"))
``` 


Use the ```caret``` package to partition the dataset into test and training sets of equal size. Train a linear model and calculate the RMSE. Repeat this exercise 100 times and report the mean and standard deviation of the RMSEs. (Hint: You can use the code shown in a previous course inside a call to ```replicate``` using a seed of 1.

```{r, include=TRUE}
library(caret)	
set.seed(1)
rmse <- replicate(100, {
	test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
	train_set <- dat %>% slice(-test_index)
	test_set <- dat %>% slice(test_index)
	fit <- lm(y ~ x, data = train_set)
 	y_hat <- predict(fit, newdata = test_set)
	sqrt(mean((y_hat-test_set$y)^2))
})
mean(rmse)
sd(rmse)
```	

- Mean: 2.488661
- SD: 0.1243952

<br/>

### **Question 2**
Now we will repeat the above but using larger datasets. Repeat the previous exercise but for datasets with ```n <- c(100, 500, 1000, 5000, 10000)```. Save the average and standard deviation of RMSE from the 100 repetitions using a seed of 1. Hint: use the sapply or map functions.

```{r, include=TRUE} 
set.seed(1)
n <- c(100, 500, 1000, 5000, 10000)
res <- sapply(n, function(n){
	Sigma <- 9*matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)
	dat <- MASS::mvrnorm(n, c(69, 69), Sigma) %>%
		data.frame() %>% setNames(c("x", "y"))
	rmse <- replicate(100, {
		test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
		train_set <- dat %>% slice(-test_index)
		test_set <- dat %>% slice(test_index)
		fit <- lm(y ~ x, data = train_set)
		y_hat <- predict(fit, newdata = test_set)
		sqrt(mean((y_hat-test_set$y)^2))
	})
	c(avg = mean(rmse), sd = sd(rmse))
})
res
```

- Mean, 100: 2.497754
- SD, 100: 0.1180821
- Mean, 500:2.720951
- SD, 500:0.08002108
- Mean, 1000:2.555545
- SD, 1000: 0.04560258
- Mean, 5000: 2.624828
- SD, 5000: 0.02309673
- Mean, 10000: 2.618442
- SD, 10000: 0.01689205

<br/>

### **Question 3**

What happens to the RMSE as the size of the dataset becomes larger?

- On average, the RMSE does not change much as ```n``` gets larger, but the variability of the RMSE decreases. [X]
- Because of the law of large numbers the RMSE decreases; more data means more precise estimates.
- ```n = 10000``` is not sufficiently large. To see a decrease in the RMSE we would need to make it larger.
- The RMSE is not a random variable.

<br/>

### **Question 4**

Now repeat the exercise from Q1, this time making the correlation between ```x``` and ```y``` larger, as in the following code:

```{r, include=TRUE}
set.seed(1)
n <- 100
Sigma <- 9*matrix(c(1.0, 0.95, 0.95, 1.0), 2, 2)
dat <- MASS::mvrnorm(n = 100, c(69, 69), Sigma) %>%
	data.frame() %>% setNames(c("x", "y"))
```

Note what happens to RMSE - set the seed to 1 as before.

```{r, include=TRUE}
set.seed(1)
rmse <- replicate(100, {
	test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
	train_set <- dat %>% slice(-test_index)
	test_set <- dat %>% slice(test_index)
	fit <- lm(y ~ x, data = train_set)
 	y_hat <- predict(fit, newdata = test_set)
	sqrt(mean((y_hat-test_set$y)^2))
})
mean(rmse)
sd(rmse)
```

- Mean: 0.9099808
- SD:0.06244347

<br/>

### **Question 5**

Which of the following best explains why the RMSE in question 4 is so much lower than the RMSE in question 1?

- It is just luck. If we do it again, it will be larger.
- The central limit theorem tells us that the RMSE is normal.
- When we increase the correlation between x and y, x has more predictive power and thus provides a better estimate of y. [X]
- These are both examples of regression so the RMSE has to be the same.

<br/>

### **Question 6**

Create a data set using the following code.

```{r, include=TRUE}
set.seed(1)
n <- 1000
Sigma <- matrix(c(1.0, 0.75, 0.75, 0.75, 1.0, 0.25, 0.75, 0.25, 1.0), 3, 3)
dat <- MASS::mvrnorm(n = 100, c(0, 0, 0), Sigma) %>%
	data.frame() %>% setNames(c("y", "x_1", "x_2"))
```

Note that ```y``` is correlated with both ```x_1``` and ```x_2``` but the two predictors are independent of each other, as seen by ```cor(dat)```.

Use the ```caret``` package to partition into a test and training set of equal size. Compare the RMSE when using just ```x_1```, just ```x_2``` and both ```x_1``` and ```x_2```. Train a linear model for each.

Which of the three models performs the best (has the lowest RMSE)?


```{r, include=TRUE}
set.seed(1)
test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
train_set <- dat %>% slice(-test_index)
test_set <- dat %>% slice(test_index)
fit <- lm(y ~ x_1, data = train_set)
y_hat <- predict(fit, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))
fit <- lm(y ~ x_2, data = train_set)
y_hat <- predict(fit, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))
fit <- lm(y ~ x_1 + x_2, data = train_set)
y_hat <- predict(fit, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))
``` 

- x_1
- x_2
- x_1 and x_2 [X]

<br/>

### **Question 7**

Report the lowest RMSE of the three models tested in Q6.

- Lowest: 0.3070962

<br/>

### **Question 8**

Repeat the exercise from q6 but now create an example in which x_1 and x_2 are highly correlated.

```
set.seed(1)
n <- 1000
Sigma <- matrix(c(1.0, 0.75, 0.75, 0.75, 1.0, 0.95, 0.75, 0.95, 1.0), 3, 3)
dat <- MASS::mvrnorm(n = 100, c(0, 0, 0), Sigma) %>%
	data.frame() %>% setNames(c("y", "x_1", "x_2"))
```
Use the caret package to partition into a test and training set of equal size. Compare the RMSE when using just x_1, just x_2, and both x_1 and x_2.

Compare the results from q6 and q8. What can you conclude?

```{r, include=TRUE}
set.seed(1)
test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
train_set <- dat %>% slice(-test_index)
test_set <- dat %>% slice(test_index)
fit <- lm(y ~ x_1, data = train_set)
y_hat <- predict(fit, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))
fit <- lm(y ~ x_2, data = train_set)
y_hat <- predict(fit, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))
fit <- lm(y ~ x_1 + x_2, data = train_set)
y_hat <- predict(fit, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))
```

- Unless we include all predictors we have no predictive power.
- Adding extra predictors improves RMSE regardless of whether the added predictors are correlated with other predictors or not.
- Adding extra predictors results in over fitting.
- Adding extra predictors can improve RMSE substantially, but not when the added predictors are highly correlated with other predictors. [X]

<br/>

## **Logistic Regression**
### **Question 1**

Define a dataset using the following code:

```{r, include=TRUE}
set.seed(2)
make_data <- function(n = 1000, p = 0.5, 
				mu_0 = 0, mu_1 = 2, 
				sigma_0 = 1,  sigma_1 = 1){
y <- rbinom(n, 1, p)
f_0 <- rnorm(n, mu_0, sigma_0)
f_1 <- rnorm(n, mu_1, sigma_1)
x <- ifelse(y == 1, f_1, f_0)
  
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
list(train = data.frame(x = x, y = as.factor(y)) %>% slice(-test_index),
	test = data.frame(x = x, y = as.factor(y)) %>% slice(test_index))
}
dat <- make_data()
```

Note that we have defined a variable 
```x``` that is predictive of a binary outcome ```y```: 
```dat$train %>% ggplot(aes(x, color = y)) + geom_density()```.
Generate 25 different datasets changing the difference between the two classes using ```delta <- seq(0, 3, len=25)``` and plot accuracy vs ```mu_1```.
Which is the correct plot?
```{r, include=TRUE}
set.seed(1)
delta <- seq(0, 3, len = 25)
res <- sapply(delta, function(d){
	dat <- make_data(mu_1 = d)
	fit_glm <- dat$train %>% glm(y ~ x, family = "binomial", data = .)
	y_hat_glm <- ifelse(predict(fit_glm, dat$test) > 0.5, 1, 0) %>% factor(levels = c(0, 1))
	mean(y_hat_glm == dat$test$y)
})
qplot(delta, res)
``` 

<br/>

## **Smoothing**
### **Question 1**

In the Wrangling course of this series, PH125.6x, we used the following code to obtain mortality counts for Puerto Rico for 2015-2018:

```{r, include=TRUE}
library(tidyverse)
library(lubridate)
library(purrr)
library(pdftools)
    
fn <- system.file("extdata", "RD-Mortality-Report_2015-18-180531.pdf", package="dslabs")
dat <- map_df(str_split(pdf_text(fn), "\n"), function(s){
	s <- str_trim(s)
	header_index <- str_which(s, "2015")[1]
	tmp <- str_split(s[header_index], "\\s+", simplify = TRUE)
	month <- tmp[1]
	header <- tmp[-1]
	tail_index  <- str_which(s, "Total")
	n <- str_count(s, "\\d+")
	out <- c(1:header_index, which(n==1), which(n>=28), tail_index:length(s))
	s[-out] %>%
		str_remove_all("[^\\d\\s]") %>%
		str_trim() %>%
		str_split_fixed("\\s+", n = 6) %>%
		.[,1:5] %>%
		as_data_frame() %>% 
		setNames(c("day", header)) %>%
		mutate(month = month,
			day = as.numeric(day)) %>%
		gather(year, deaths, -c(day, month)) %>%
		mutate(deaths = as.numeric(deaths))
}) %>%
	mutate(month = recode(month, "JAN" = 1, "FEB" = 2, "MAR" = 3, "APR" = 4, "MAY" = 5, "JUN" = 6, 
                          "JUL" = 7, "AGO" = 8, "SEP" = 9, "OCT" = 10, "NOV" = 11, "DEC" = 12)) %>%
	mutate(date = make_date(year, month, day)) %>%
	filter(date <= "2018-05-01")
```

Use the loess function to obtain a smooth estimate of the expected number of deaths as a function of date. Plot this resulting smooth function. Make the span about two months long.

Which of the following plots is correct?

```{r, include=TRUE}
span <- 60 / as.numeric(diff(range(dat$date)))
fit <- dat %>% mutate(x = as.numeric(date)) %>% loess(deaths ~ x, data = ., span = span, degree = 1)
dat %>% mutate(smooth = predict(fit, as.numeric(date))) %>%
	ggplot() +
	geom_point(aes(date, deaths)) +
	geom_line(aes(date, smooth), lwd = 2, col = 2)
```

<br/>

### **Question 2**

Work with the same data as in Q1 to plot smooth estimates against day of the year, all on the same plot, but with different colors for each year.

Which code produces the desired plot?

```{r, include=TRUE}
span <- 60 / as.numeric(diff(range(dat$date)))
fit <- dat %>% mutate(x = as.numeric(date)) %>% loess(deaths ~ x, data = ., span = span, degree = 1)
dat %>% 
	mutate(smooth = predict(fit, as.numeric(date)), day = yday(date), year = as.character(year(date))) %>%
	ggplot(aes(day, smooth, col = year)) +
	geom_line(lwd = 2)
```
```
dat %>% 
	mutate(smooth = predict(fit), day = yday(date), year = as.character(year(date))) %>%
	ggplot(aes(day, smooth, col = year)) +
	geom_line(lwd = 2)
```
```
dat %>% 
	mutate(smooth = predict(fit, as.numeric(date)), day = mday(date), year = as.character(year(date))) %>%
	ggplot(aes(day, smooth, col = year)) +
	geom_line(lwd = 2)
```
```
dat %>% 
	mutate(smooth = predict(fit, as.numeric(date)), day = yday(date), year = as.character(year(date))) %>%
	ggplot(aes(day, smooth)) +
  	geom_line(lwd = 2)
```
```
dat %>% 
	mutate(smooth = predict(fit, as.numeric(date)), day = yday(date), year = as.character(year(date))) %>%
	ggplot(aes(day, smooth, col = year)) +
	geom_line(lwd = 2) [X]
```

<br/>

### **Question 3**

Suppose we want to predict 2s and 7s in the ```mnist_27``` dataset with just the second covariate. Can we do this? On first inspection it appears the data does not have much predictive power.

In fact, if we fit a regular logistic regression the coefficient for ```x_2``` is not significant!

This can be seen using this code:

```{r, include=TRUE}
library(broom)
library(dslabs)
data(mnist_27)
mnist_27$train %>% glm(y ~ x_2, family = "binomial", data = .) %>% tidy()
```

Plotting a scatterplot here is not useful since y is binary:

```{r, include=TRUE}
qplot(x_2, y, data = mnist_27$train)
```

Fit a loess line to the data above and plot the results. What do you observe?

- There is no predictive power and the conditional probability is linear.
- There is no predictive power and the conditional probability is non-linear.
- There is predictive power and the conditional probability is linear.
- There is predictive power and the conditional probability is non-linear. [X}

<br/>

## **Working with Matrices**
### **Question 1**

Which line of code correctly creates a 100 by 10 matrix of randomly generated normal numbers and assigns it to ```x```?

- ```x <- matrix(rnorm(1000), 100, 100)```
- ```x <- matrix(rnorm(100*10), 100, 10)``` [X]
- ```x <- matrix(rnorm(100*10), 10, 10)```
- ```x <- matrix(rnorm(100*10), 10, 100)```

<br/>

### **Question 2**

Write the line of code that would give you the specified information about the matrix ```x``` that you generated in q1. Do not include any spaces in your line of code.

Dimension of ```x```.

```dim(x)```
Number of rows of ```x```.
```nrow(x)```
Number of columns of ```x```.
```ncol(x)```
<br/>
### **Question 3**
Which of the following lines of code would add the scalar 1 to row 1, the scalar 2 to row 2, and so on, for the matrix ```x```?
- ```x <- x + seq(nrow(x)) ``` [X]
- ```x <- 1:nrow(x)```
- ```x <- sweep(x, 2, 1:nrow(x),"+")```
- ```x <- sweep(x, 1, 1:nrow(x),"+")``` [X]
<br/>
### **Question 4**
Which of the following lines of code would add the scalar 1 to column 1, the scalar 2 to column 2, and so on, for the matrix ```x?```
- ```x <- 1:ncol(x)```
- ```x <- 1:col(x)```
- ```x <- sweep(x, 2, 1:ncol(x), FUN = "+") ``` [X]
- ```x <- -x```
<br/>
### **Question 5**
Which code correctly computes the average of each row of ```x```?
- mean(x)```
- rowMedians(x)```
- sapply(x,mean)```
- rowSums(x)```
- rowMeans(x)``` [X]
 Which code correctly computes the average of each column of ```x```?
 
- ```mean(x)```
- ```sapply(x,mean)```
- ```colMeans(x)``` [X]
- ```colMedians(x)```
- ```colSums(x)```
<br/>
### **Question 6**
For each digit in the mnist training data, compute the proportion of pixels that are in the grey area, defined as values between 50 and 205. (To visualize this, you can make a boxplot by digit class.)
What proportion of pixels are in the grey area overall, defined as values between 50 and 205?
```{r, include=TRUE}
mnist <- read_mnist()
y <- rowMeans(mnist$train$images>50 & mnist$train$images<205)
qplot(as.factor(mnist$train$labels), y, geom = "boxplot")
mean(y)
```



<br/>

### **Question 2**

True or False: In machine learning, we build algorithms that take feature values (X) and train a model using known outcomes (Y) that is then used to predict outcomes when presented with features without known outcomes.

- True [X]
- False

---
title: "04 - Distance, Knn, Cross Validation, and Generative Models Comprehension Check"
author: "Alessandro Corradini - Harvard Data Science Professional"
output:
  html_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

<br/>

## **Distance**
### **Question 1**

Load the following dataset:

```{r, include=TRUE}
library(dslabs)
data("tissue_gene_expression")
```

This dataset includes a matrix ```x```:

```{r, include=TRUE}
dim(tissue_gene_expression$x)
```

This matrix has the gene expression levels of 500 genes from 189 biological samples representing seven different tissues. The tissue type is stored in ```y```:

```{r, include=TRUE}
table(tissue_gene_expression$y)
```

Which of the following lines of code computes the Euclidean distance between each observation and stores it in the object ```d```?

```{r, include=TRUE}
d <- dist(tissue_gene_expression$x)
```

- ```d <- dist(tissue_gene_expression$x, distance='maximum')```
- ```d <- dist(tissue_gene_expression)```
- ```d <- dist(tissue_gene_expression$x)``` [X]
- ```d <- cor(tissue_gene_expression$x)```

<br/>

### **Question 2**

Compare the distances between observations 1 and 2 (both cerebellum), observations 39 and 40 (both colon), and observations 73 and 74 (both endometrium).

Distance-wise, are samples from tissues of the same type closer to each other?

```{r, include=TRUE}
ind <- c(1, 2, 39, 40, 73, 74)
as.matrix(d)[ind,ind]
```

- No, the samples from the same tissue type are not necessarily closer.
- The two colon samples are closest to each other, but the samples from the other two tissues are not.
- The two cerebellum samples are closest to each other, but the samples from the other two tissues are not.
- Yes, the samples from the same tissue type are closest to each other. [X]


### **Question 3**

Make a plot of all the distances using the ```image``` function to see if the pattern you observed in Q2 is general.

Which code would correctly make the desired plot?

- ```image(d)```
- ```image(as.matrix(d))``` [X]
- ```d```
- ```image()```

## **Nearest Neighbors**
### **Question 1**

Previously, we used logistic regression to predict sex based on height. Now we are going to use knn to do the same. Use the code described in these videos to select the 
```F_1``` measure and plot it against ```k```. Compare to the ```F_1``` of about 0.6 we obtained with regression. Set the seed to 1.
- What is the max value of ```F_1```? 0.6019417
- At what value of ```k``` does the max occur? 40
```{r, include=TRUE}
set.seed(1)
data("heights")
library(caret)
library(dplyr)
ks <- seq(1, 101, 3)
F_1 <- sapply(ks, function(k){
	test_index <- createDataPartition(heights$sex, times = 1, p = 0.5, list = FALSE)
	test_set <- heights[test_index, ]
	train_set <- heights[-test_index, ]
	fit <- knn3(sex ~ height, data = train_set, k = k)
	y_hat <- predict(fit, test_set, type = "class") %>% 
		factor(levels = levels(train_set$sex))
	F_meas(data = y_hat, reference = test_set$sex)
})
plot(ks, F_1)
max(F_1)
```

### **Question 2**

Next we will use the same gene expression example used in the Comprehension Check: Distance exercises. You can load it like this:

```{r, include=TRUE}
library(dslabs)
data("tissue_gene_expression")
```

Split the data into training and test sets, and report the accuracy you obtain. Try it for ```k = 1, 3, 5, 7, 9, 11```. Set the seed to 1.

```{r, include=TRUE}
set.seed(1)
library(caret)
y <- tissue_gene_expression$y
x <- tissue_gene_expression$x
train_index <- createDataPartition(y, list = FALSE)
sapply(seq(1, 11, 2), function(k){
	fit <- knn3(x[train_index,], y[train_index], k = k)
	y_hat <- predict(fit, newdata = data.frame(x=x[-train_index,]),
				type = "class")
mean(y_hat == y[-train_index])
})
```

- k=1: 0.980
- k=3: 0.969
- k=5: 0.989
- k=7: 0.968
- k=9: 0.957
- k=11: 0.957

## **Cross-validation**
### **Question 1**

Generate a set of random predictors and outcomes using the following code:

```{r, include=TRUE}
library(dbplyr)
library(dslabs)
library(tidyverse)
library(caret)
set.seed(1996)
n <- 1000
p <- 10000
x <- matrix(rnorm(n*p), n, p)
colnames(x) <- paste("x", 1:ncol(x), sep = "_")
y <- rbinom(n, 1, 0.5) %>% factor()
x_subset <- x[ ,sample(p, 100)]
```

Because ```x``` and ```y``` are completely independent, you should not be able to predict ```y``` using ```x``` with accuracy greater than 0.5. Confirm this by running cross-validation using logistic regression to fit the model. Because we have so many predictors, we selected a random sample ```x_subset```. Use the subset when training the model.

Which code correctly performs this cross-validation?

```{r, include=TRUE}
fit <- train(x_subset, y, method = "glm")
fit$results
```

```
fit <- train(x_subset, y)
fit$results
```
```
fit <- train(x_subset, y, method = "glm")
fit$results [X]
```
```
fit <- train(y, x_subset, method = "glm")
fit$results
```
```
fit <- test(x_subset, y, method = "glm")
fit$results
```

### **Question 2**

Now, instead of using a random selection of predictors, we are going to search for those that are most predictive of the outcome. We can do this by comparing the values for $\ y = 1$ the  group to those in the $\ y = 0$  group, for each predictor, using a t-test. You can do perform this step like this:

```{r, include=TRUE}
library(devtools)
devtools::install_bioc("genefilter")
library(genefilter)
tt <- colttests(x, y)
```

Which of the following lines of code correctly creates a vector of the p-values called ```pvals```?

- ```pvals <- tt$dm```
- ```pvals <- tt$statistic```
- ```pvals <- tt```
- ```pvals <- tt$p.value``` [X]


```{r, include=TRUE}
pvals <- tt$p.value
```

### **Question 3**

Create an index ```ind``` with the column numbers of the predictors that were "statistically significantly" associated with ```y```. Use a p-value cutoff of 0.01 to define "statistically significantly."

```{r, include=TRUE}
ind <- which(pvals <= 0.01)
length(ind)
```

How many predictors survive this cutoff? 108

### **Question 4**

Now re-run the cross-validation after redefinining ```x_subset``` to be the subset of ```x``` defined by the columns showing "statistically significant" association with ```y```.

```{r, include=TRUE}
x_subset <- x[,ind]
fit <- train(x_subset, y, method = "glm")
fit$results
```

What is the accuracy now? 0.7571395



### **Question 5**

Re-run the cross-validation again, but this time using kNN. Try out the following grid ```k = seq(101, 301, 25)``` of tuning parameters. Make a plot of the resulting accuracies.

Which code is correct?

```
fit <- train(x_subset, y, method = "knn", tuneGrid = data.frame(k = seq(101, 301, 25)))
ggplot(fit) [X]
```
```
fit <- train(x_subset, y, method = "knn")
ggplot(fit)
```
```
fit <- train(x_subset, y, method = "knn", tuneGrid = data.frame(k = seq(103, 301, 25)))
ggplot(fit)
```
```
fit <- train(x_subset, y, method = "knn", tuneGrid = data.frame(k = seq(101, 301, 5)))
ggplot(fit)
```

### **Question 6**

In the previous exercises, we see that despite the fact that ```x``` and ```y``` are completely independent, we were able to predict ```y``` with accuracy higher than 70%. We must be doing something wrong then.

What is it?

- The function train estimates accuracy on the same data it uses to train the algorithm.
- We are overfitting the model by including 100 predictors.
- - We used the entire dataset to select the columns used in the model. [X]
The high accuracy is just due to random variability.


### **Question 7**

Use the ```train``` function to predict tissue from gene expression in the ```tissue_gene_expression``` dataset. Use kNN.

```{r, include=TRUE}
data("tissue_gene_expression")
fit <- with(tissue_gene_expression, train(x, y, method = "knn", tuneGrid = data.frame( k = seq(1, 7, 2))))
ggplot(fit)
fit$results
```

What value of ```k``` works best? 1

## **Bootstrap**
### **Question 1**

The ```createResample``` function can be used to create bootstrap samples. For example, we can create 10 bootstrap samples for the ```mnist_27``` dataset like this:


set.seed(1995)
indexes <- createResample(mnist_27$train$y, 10)


How many times do 3, 4, and 7 appear in the first resampled index?

```{r,include=TRUE}
sum(indexes[[1]] == 3)
sum(indexes[[1]] == 4)
sum(indexes[[1]] == 7)
```

- Enter the number of times 3 appears: 1
- Enter the number of times 4 appears: 4
- Enter the number of times 7 appears: 0


### **Question 2**

We see that some numbers appear more than once and others appear no times. This has to be this way for each dataset to be independent. Repeat the exercise for all the resampled indexes.

```{r,include=TRUE}
x=sapply(indexes, function(ind){
	sum(ind == 3)
})
sum(x)
```

What is the total number of times that 3 appears in all of the resampled indexes? 11


### **Question 3**

Generate a random dataset using the following code:

```{r,include=TRUE}
set.seed(1)
y <- rnorm(100, 0, 1)
```

Estimate the 75th quantile, which we know is ```qnorm(0.75)```, with the sample quantile: ```quantile(y, 0.75)```.

Run a Monte Carlo simulation with 10,000 repetitions to learn the expected value and standard error of this random variable. Set the seed to 1.

```{r,include=TRUE}
set.seed(1)
B <- 10000
q_75 <- replicate(B, {
	y <- rnorm(100, 0, 1)
	quantile(y, 0.75)
})
mean(q_75)
sd(q_75)
```

- Expected value 0.6655976
- Standard error 0.1353847

 
### **Question 4**

In practice, we can't run a Monte Carlo simulation. Use 10 bootstrap samples to estimate the standard error using just the initial sample ```y``` . Set the seed to 1.

```{r,include=TRUE}
set.seed(1)
indexes <- createResample(y, 10)
q_75_star <- sapply(indexes, function(ind){
	y_star <- y[ind]
	quantile(y_star, 0.75)
})
mean(q_75_star)
sd(q_75_star)
```

- Expected value 0.7312648
- Standard error 0.07419278

 
### **Question 5**

Repeat the exercise from Q4 but with 10,000 bootstrap samples instead of 10. Set the seed to 1.

```{r,include=TRUE}
set.seed(1)
indexes <- createResample(y, 10000)
q_75_star <- sapply(indexes, function(ind){
	y_star <- y[ind]
	quantile(y_star, 0.75)
})
mean(q_75_star)
sd(q_75_star)
```

- Expected value 0.6737512
- Standard error 0.0930575


### **Question 6**

Compare the SD values obtained using 10 vs 10,000 bootstrap samples.

What do you observe?

- The SD is substantially lower with 10,000 bootstrap samples than with 10.
- The SD is roughly the same in both cases. [X]
- The SD is substantially higher with 10,000 bootstrap samples than with 10. 

## **Generative Models**

In the following exercises, we are going to apply LDA and QDA to the ```tissue_gene_expression``` dataset. We will start with simple examples based on this dataset and then develop a realistic example.

### **Question 1**

Create a dataset of samples from just cerebellum and hippocampus, two parts of the brain, and a predictor matrix with 10 randomly selected columns using the following code:

```{r, include=TRUE}
set.seed(1993)
data("tissue_gene_expression")
ind <- which(tissue_gene_expression$y %in% c("cerebellum", "hippocampus"))
y <- droplevels(tissue_gene_expression$y[ind])
x <- tissue_gene_expression$x[ind, ]
x <- x[, sample(ncol(x), 10)]
```

Use the train function to estimate the accuracy of LDA.

```{r, include=TRUE}
fit_lda <- train(x, y, method = "lda")
fit_lda$results["Accuracy"]
```

What is the accuracy? 0.8707879




### **Question 2**

In this case, LDA fits two 10-dimensional normal distributions. Look at the fitted model by looking at the ```finalModel``` component of the result of ```train```. Notice there is a component called ```means``` that includes the estimated means of both distributions. Plot the mean vectors against each other and determine which predictors (genes) appear to be driving the algorithm.

Which TWO genes appear to be driving the algorithm?

```{r, include=TRUE}
t(fit_lda$finalModel$means) %>% data.frame() %>%
	mutate(predictor_name = rownames(.)) %>%
	ggplot(aes(cerebellum, hippocampus, label = predictor_name)) +
	geom_point() +
	geom_text() +
	geom_abline()
```

- PLCB1
- RAB1B [X]
- MSH4
- OAZ2 [X]
- SPI1
- SAPCD1
- HEMK1


### **Question 3**

Repeat the exercise in Q1 with QDA.

Create a dataset of samples from just cerebellum and hippocampus, two parts of the brain, and a predictor matrix with 10 randomly selected columns using the following code:

```{r, include=TRUE}
set.seed(1993)
data("tissue_gene_expression")
ind <- which(tissue_gene_expression$y %in% c("cerebellum", "hippocampus"))
y <- droplevels(tissue_gene_expression$y[ind])
x <- tissue_gene_expression$x[ind, ]
x <- x[, sample(ncol(x), 10)]
```

Use the train function to estimate the accuracy of QDA.

```{r, include=TRUE}
fit_qda <- train(x, y, method = "qda")
fit_qda$results["Accuracy"]
```

What is the accuracy? 0.8147954

### **Question 4**

Which TWO genes drive the algorithm when using QDA instead of LDA?

```{r, include=TRUE}
t(fit_qda$finalModel$means) %>% data.frame() %>%
	mutate(predictor_name = rownames(.)) %>%
	ggplot(aes(cerebellum, hippocampus, label = predictor_name)) +
	geom_point() +
	geom_text() +
	geom_abline()
```

- PLCB1
- RAB1B [X]
- MSH4
- OAZ2 [X]
- SPI1
- SAPCD1
- HEMK1

### **Question 5**

One thing we saw in the previous plots is that the values of the predictors correlate in both groups: some predictors are low in both groups and others high in both groups. The mean value of each predictor found in ```colMeans(x)``` is not informative or useful for prediction and often for purposes of interpretation, it is useful to center or scale each column. This can be achieved with the ```preProcessing``` argument in train. Re-run LDA with ```preProcessing = "scale"```. Note that accuracy does not change, but it is now easier to identify the predictors that differ more between groups than based on the plot made in Q2.

Which TWO genes drive the algorithm after performing the scaling?

```{r, include=TRUE}
fit_lda <- train(x, y, method = "lda", preProcess = "center")
fit_lda$results["Accuracy"]
t(fit_lda$finalModel$means) %>% data.frame() %>%
	mutate(predictor_name = rownames(.)) %>%
	ggplot(aes(predictor_name, hippocampus)) +
	geom_point() +
	coord_flip()
	
d <- apply(fit_lda$finalModel$means, 2, diff)
ind <- order(abs(d), decreasing = TRUE)[1:2]
plot(x[, ind], col = y)
```	

- C21orf62
- PLCB1
- RAB1B
- MSH4
- OAZ2 [X]
- SPI1 [X]
- SAPCD1
- IL18R1


### **Question 6**

Now we are going to increase the complexity of the challenge slightly: we will consider all the tissue types. Use the following code to create your dataset:

```{r, include=TRUE}
set.seed(1993)
data("tissue_gene_expression")
y <- tissue_gene_expression$y
x <- tissue_gene_expression$x
x <- x[, sample(ncol(x), 10)]
fit_lda <- train(x, y, method = "lda", preProcess = c("center"))
fit_lda$results["Accuracy"]
```	

What is the accuracy using LDA? 0.8194837

---
title: "05 - Classification with More than Two Classes and the Caret Package Comprehension Check"
author: "Alessandro Corradini - Harvard Data Science Professional"
output:
  html_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

<br/>

## **Trees and Random Forests**
### **Question 1**

Create a simple dataset where the outcome grows 0.75 units on average for every increase in a predictor, using this code:

```{r, include=TRUE}
library(rpart)
library(dplyr)
library(ggplot2)
n <- 1000
sigma <- 0.25
x <- rnorm(n, 0, 1)
y <- 0.75 * x + rnorm(n, 0, sigma)
dat <- data.frame(x = x, y = y)
```

Which code correctly uses rpart to fit a regression tree and saves the result to fit?

```{r, include=TRUE}
fit <- rpart(y ~ ., data = dat)
```

- ```fit <- rpart(y ~ .)```
- ```fit <- rpart(y, ., data = dat)```
- ```fit <- rpart(x ~ ., data = dat)```
- ```fit <- rpart(y ~ ., data = dat)``` [X]

<br/>

### **Question 2**

Which of the following plots correctly shows the final tree obtained in Q1?

```{r, include=TRUE}
plot(fit)
text(fit)
```

### **Question 3**

Below is most of the code to make a scatter plot of ```y``` versus ```x``` along with the predicted values based on the fit.

```{r, include=TRUE}
dat %>% 
	mutate(y_hat = predict(fit)) %>% 
	ggplot() +
	geom_point(aes(x, y)) +
	geom_step(aes(x, y_hat), col=2)
```

Which line of code should be used to replace #BLANK in the code above?

- ```geom_step(aes(x, y_hat), col=2)``` [X]
- ```geom_smooth(aes(y_hat, x), col=2)```
- ```geom_quantile(aes(x, y_hat), col=2)```
- ```geom_step(aes(y_hat, x), col=2)```

### **Question 4**

Now run Random Forests instead of a regression tree using randomForest from the ```__randomForest__``` package, and remake the scatterplot with the prediction line. Part of the code is provided for you below.

```{r, include=TRUE}
library(randomForest)
fit <- randomForest(y ~ x, data = dat)
dat %>% 
	mutate(y_hat = predict(fit)) %>% 
	ggplot() +
	geom_point(aes(x, y)) +
	geom_step(aes(x, y_hat), col = 2)
```	

What code should replace #BLANK in the provided code?

- ```randomForest(y ~ x, data = dat)``` [X]
- ```randomForest(x ~ y, data = dat)```
- ```randomForest(y ~ x, data = data)```
- ```randomForest(x ~ y)```

### **Question 5**

Use the plot function to see if the Random Forest from Q4 has converged or if we need more trees.

```{r, include=TRUE}
plot(fit) 
```

### **Question 6**

It seems that the default values for the Random Forest result in an estimate that is too flexible (unsmooth). Re-run the Random Forest but this time with a node size of 50 and a maximum of 25 nodes. Remake the plot.

Part of the code is provided for you below.

```{r, include=TRUE}
library(randomForest)
fit <- randomForest(y ~ x, data = dat, nodesize = 50, maxnodes = 25)
dat %>% 
	mutate(y_hat = predict(fit)) %>% 
	ggplot() +
	geom_point(aes(x, y)) +
	geom_step(aes(x, y_hat), col = 2)
```

What code should replace #BLANK in the provided code?

- ```randomForest(y ~ x, data = dat, nodesize = 25, maxnodes = 25)```
- ```randomForest(y ~ x, data = dat, nodes = 50, max = 25)```
- ```randomForest(x ~ y, data = dat, nodes = 50, max = 25)```
- ```randomForest(y ~ x, data = dat, nodesize = 50, maxnodes = 25)``` [X]
- ```randomForest(x ~ y, data = dat, nodesize = 50, maxnodes = 25)```

## **Caret Package**
The exercises in Q1 and Q2 continue the analysis you began in the last set of assessments.

### **Question 1**

In the exercise in Q6 from Comprehension Check: Trees and Random Forests, we saw that changing ```nodesize``` to 50 and setting ```maxnodes``` to 25 yielded smoother results. Let's use the train function to help us pick what the values of ```nodesize``` and ```maxnodes``` should be.

From the caret description of methods, we see that we can't tune the ```maxnodes``` parameter or the ```nodesize``` argument with ```randomForests```. So we will use the ```__Rborist__``` package and tune the ```minNode``` argument. Use the train function to try values ```minNode <- seq(25, 100, 25)```. Set the seed to 1.

```{r, include=TRUE}
set.seed(1)
library(caret)
fit <- train(y ~ ., method = "Rborist",   
				tuneGrid = data.frame(predFixed = 1, 
									  minNode = seq(25, 100, 25)),
				data = dat)
ggplot(fit)
```

Which value minimizes the estimated RMSE? 50

### **Question 2**

Part of the code to make a scatterplot along with the prediction from the best fitted model is provided below.

```{r, include=TRUE}
library(caret)
dat %>% 
	mutate(y_hat = predict(fit)) %>% 
	ggplot() +
	geom_point(aes(x, y)) +
    geom_step(aes(x, y_hat), col = 2)
```   

Which code correctly can be used to replace #BLANK in the code above?

- ```geom_step(aes(y_hat, x), col = 2)``` 
- ```geom_step(aes(x, y_hat), col = 2)``` [X]
- ```geom_step(aes(x, y), col = 2)```
- ```geom_step(aes(x_hat, y_hat), col = 2)```
- ```geom_smooth(aes(x, y_hat), col = 2)```
- ```geom_smooth(aes(y_hat, x), col = 2)```

### **Question 3**

Use the ```rpart``` function to fit a classification tree to the ```tissue_gene_expression dataset```. Use the train function to estimate the accuracy. Try out ```cp``` values of ```seq(0, 0.1, 0.01)```. Plot the accuracies to report the results of the best model. Set the seed to 1991.

```{r, include=TRUE}
library(caret)
library(dslabs)
set.seed(1991)
data("tissue_gene_expression")
    
fit <- with(tissue_gene_expression, 
                train(x, y, method = "rpart",
                      tuneGrid = data.frame(cp = seq(0, 0.1, 0.01))))
    
ggplot(fit)      
```

Which value of ```cp``` gives the highest accuracy? 0

### **Question 4**

Study the confusion matrix for the best fitting classification tree from the exercise in Q3.

What do you observe happening for the placenta samples?

- Placenta samples are all accurately classified.
- Placenta samples are being classified as two similar tissues.
- Placenta samples are being classified somewhat evenly across tissues. [X]
- Placenta samples not being classified into any of the classes.


### **Question 5**

Note that there are only 6 placentas in the dataset. By default, ```rpart``` requires 20 observations before splitting a node. That means that it is difficult to have a node in which placentas are the majority. Rerun the analysis you did in the exercise in Q3, but this time, allow ```rpart``` to split any node by using the argument ```control = rpart.control(minsplit = 0)```. Look at the confusion matrix again to determine whether the accuracy increases. Again, set the seed to 1991.

```{r, include=TRUE}
set.seed(1991)
data("tissue_gene_expression")
    
fit_rpart <- with(tissue_gene_expression, 
                      train(x, y, method = "rpart",
                            tuneGrid = data.frame(cp = seq(0, 0.10, 0.01)),
                            control = rpart.control(minsplit = 0)))
ggplot(fit_rpart)
confusionMatrix(fit_rpart)
```

What is the accuracy now? 0.9141

### **Question 6**

Plot the tree from the best fitting model of the analysis you ran in Q5.

Which gene is at the first split?

```{r, include=TRUE}
plot(fit_rpart$finalModel)
text(fit_rpart$finalModel)
```

- B3GNT4
- CAPN3
- CES2
- CFHR4
- CLIP3
- GPA33 [X]
- HRH1


### **Question 7**

We can see that with just seven genes, we are able to predict the tissue type. Now let's see if we can predict the tissue type with even fewer genes using a Random Forest. Use the train function and the rf method to train a Random Forest. Try out values of mtry ranging from seq(50, 200, 25) (you can also explore other values on your own). What mtry value maximizes accuracy? To permit small nodesize to grow as we did with the classification trees, use the following argument: nodesize = 1.

Note: This exercise will take some time to run. If you want to test out your code first, try using smaller values with ntree. Set the seed to 1991 again.

```{r, include=TRUE}
set.seed(1991)
library(randomForest)
fit <- with(tissue_gene_expression, 
                train(x, y, method = "rf", 
                      nodesize = 1,
                      tuneGrid = data.frame(mtry = seq(50, 200, 25))))
    
ggplot(fit)
```

What value of mtry maximizes accuracy? 100

### **Question 8**

Use the function varImp on the output of train and save it to an object called imp.

```{r, include=TRUE}
imp <- varImp(fit)
imp
```

What should replace #BLANK in the code above?


### **Question 9**

The ```rpart``` model we ran above produced a tree that used just seven predictors. Extracting the predictor names is not straightforward, but can be done. If the output of the call to train was ```fit_rpart```, we can extract the names like this:

```{r, include=TRUE}
tree_terms <- as.character(unique(fit_rpart$finalModel$frame$var[!(fit_rpart$finalModel$frame$var == "<leaf>")]))
tree_terms
```

Calculate the variable importance in the Random Forest call for these seven predictors and examine where they rank.

```{r, include=TRUE}
data_frame(term = rownames(imp$importance), 
			importance = imp$importance$Overall) %>%
	mutate(rank = rank(-importance)) %>% arrange(desc(importance)) %>%
	filter(term %in% tree_terms)
```

- What is the importance of the CFHR4 gene in the Random Forest call? 35.03253
- What is the rank of the CFHR4 gene in the Random Forest call? 7

---
title: "06 - Model Fitting and Recommendation Systems Comprehension Check"
author: "Alessandro Corradini - Harvard Data Science Professional"
output:
  html_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

<br/>

## **Ensembles**

For these exercises we are going to build several machine learning models for the mnist_27 dataset and then build an ensemble. Each of the exercises in this comprehension check builds on the last.

Use the training set to build a model with several of the models available from the caret package. We will test out all of the following models in this exercise:

```{r, include=TRUE}
models <- c("glm", "lda",  "naive_bayes",  "svmLinear", 
                "gamboost",  "gamLoess", "qda", 
                "knn", "kknn", "loclda", "gam",
                "rf", "ranger",  "wsrf", "Rborist", 
                "avNNet", "mlp", "monmlp",
                "adaboost", "gbm",
                "svmRadial", "svmRadialCost", "svmRadialSigma")
```

We have not explained many of these, but apply them anyway using train with all the default parameters. You will likely need to install some packages. Keep in mind that you will probably get some warnings. Also, it will probably take a while to train all of the models - be patient!

Run the following code to train the various models:

```{r, include=TRUE}
library(caret)
library(dslabs)
set.seed(1)
data("mnist_27")
fits <- lapply(models, function(model){ 
	print(model)
	train(y ~ ., method = model, data = mnist_27$train)
}) 
    
names(fits) <- models
```

### **Question 2**

Now that you have all the trained models in a list, use ```sapply``` or ```map``` to create a matrix of predictions for the test set. You should end up with a matrix with ```length(mnist_27$test$y)``` rows and ```length(models)```.

What are the dimensions of the matrix of predictions?

```{r, include=TRUE}
pred <- sapply(fits, function(object) 
	predict(object, newdata = mnist_27$test))
dim(pred)
```

- Number of rows: 200
- Number of columns: 23


### **Question 3**

Now compute accuracy for each model on the test set. Report the mean accuracy across all models.

```{r, include=TRUE}
acc <- colMeans(pred == mnist_27$test$y)
acc
mean(acc)
```

### **Question 4**

Next, build an ensemble prediction by majority vote and compute the accuracy of the ensemble.

```{r, include=TRUE}
votes <- rowMeans(pred == "7")
y_hat <- ifelse(votes > 0.5, "7", "2")
mean(y_hat == mnist_27$test$y)
``` 

What is the accuracy of the ensemble? 0.84


### **Question 5**

In Q3, we computed the accuracy of each method on the training set and noticed that the individual accuracies varied.

How many of the individual methods do better than the ensemble? 1

Which individual methods perform better than the ensemble?

```{r, include=TRUE}
ind <- acc > mean(y_hat == mnist_27$test$y)
sum(ind)
models[ind]
```

- glm
- lda
- naive_bayes
- svmLinear
- gamboost
- gamLoess
- qda
- knn
- kknn
- loclda [X]
- gam
- rf
- ranger
- wsrf
- Rborist
- avNNet
- mlp
- monmlp
- adaboost
- gbm
- svmRadial
- svmRadialCost
- svmRadialSigma


### **Question 6**

It is tempting to remove the methods that do not perform well and re-do the ensemble. The problem with this approach is that we are using the test data to make a decision. However, we could use the accuracy estimates obtained from cross validation with the training data. Obtain these estimates and save them in an object. Report the mean accuracy of the new estimates.

```{r, include=TRUE}
acc_hat <- sapply(fits, function(fit) min(fit$results$Accuracy))
mean(acc_hat)
```

What is the mean accuracy of the new estimates? 0.811891



### **Question 7**

Now let's only consider the methods with an estimated accuracy of greater than or equal to 0.8 when constructing the ensemble.

```{r, include=TRUE}
ind <- acc_hat >= 0.8
votes <- rowMeans(pred[,ind] == "7")
y_hat <- ifelse(votes>=0.5, 7, 2)
mean(y_hat == mnist_27$test$y)
```

What is the accuracy of the ensemble now? 0.845


## **Dimension Reduction**
### **Question 1**

We want to explore the tissue_gene_expression predictors by plotting them.

```{r, include=TRUE}
library(dplyr)
data("tissue_gene_expression")
dim(tissue_gene_expression$x)
```

We want to get an idea of which observations are close to each other, but, as you can see from the dimensions, the predictors are 500-dimensional, making plotting difficult. Plot the first two principal components with color representing tissue type.

Which tissue is in a cluster by itself?

```{r, include=TRUE}
pc <- prcomp(tissue_gene_expression$x)
data.frame(pc_1 = pc$x[,1], pc_2 = pc$x[,2], 
			tissue = tissue_gene_expression$y) %>%
	ggplot(aes(pc_1, pc_2, color = tissue)) +
	geom_point()
```

- cerebellum
- colon
- endometrium
- hippocampus
- kidney
- liver [X]
- placenta




### **Question 2**

The predictors for each observation are measured using the same device and experimental procedure. This introduces biases that can affect all the predictors from one observation. For each observation, compute the average across all predictors, and then plot this against the first PC with color representing tissue. Report the correlation.

```{r, include=TRUE}
avgs <- rowMeans(tissue_gene_expression$x)
data.frame(pc_1 = pc$x[,1], avg = avgs, 
			tissue = tissue_gene_expression$y) %>%
ggplot(aes(avgs, pc_1, color = tissue)) +
	geom_point()
cor(avgs, pc$x[,1])
```

What is the correlation? 0.5969088


### **Question 3**

We see an association with the first PC and the observation averages. Redo the PCA but only after removing the center. Part of the code is provided for you.

```{r, include=TRUE}
x <- with(tissue_gene_expression, sweep(x, 1, rowMeans(x)))
pc <- prcomp(x)
data.frame(pc_1 = pc$x[,1], pc_2 = pc$x[,2], 
			tissue = tissue_gene_expression$y) %>%
	ggplot(aes(pc_1, pc_2, color = tissue)) +
	geom_point()
```

Which line of code should be used to replace #BLANK in the code block above?

- ```x <- with(tissue_gene_expression, sweep(x, 1, mean(x)))```
- ```x <- sweep(x, 1, rowMeans(tissue_gene_expression$x))```
- ```x <- tissue_gene_expression$x - mean(tissue_gene_expression$x)```
- ```x <- with(tissue_gene_expression, sweep(x, 1, rowMeans(x)))``` [X]

### **Question 4**

For the first 10 PCs, make a boxplot showing the values for each tissue.

For the 7th PC, which two tissues have the greatest median difference?

```{r, include=TRUE}
for(i in 1:10){
	boxplot(pc$x[,i] ~ tissue_gene_expression$y, main = paste("PC", i))
}
```

- cerebellum
- colon
- endometrium
- hippocampus
- kidney
- liver [X]
- placenta [X]


### **Question 5**

Plot the percent variance explained by PC number. Hint: use the ```summary``` function.

```{r, include=TRUE}
plot(summary(pc)$importance[3,])
```

How many PCs are required to reach a cumulative percent variance explained greater than 50%? 3


## **Recommendation Systems**

The following exercises all work with the movielens data, which can be loaded using the following code:

```{r, include=TRUE}
library(dslabs)
data("movielens")
```

### **Question 1**

Compute the number of ratings for each movie and then plot it against the year the movie came out. Use the square root transformation on the counts.

What year has the highest median number of ratings? 1995

```{r, include=TRUE}
movielens %>% group_by(movieId) %>%
	summarize(n = n(), year = as.character(first(year))) %>%
	qplot(year, n, data = ., geom = "boxplot") +
	coord_trans(y = "sqrt") +
	theme(axis.text.x = element_text(angle = 90, hjust = 1))
```


### **Question 2**

We see that, on average, movies that came out after 1993 get more ratings. We also see that with newer movies, starting in 1993, the number of ratings decreases with year: the more recent a movie is, the less time users have had to rate it.

Among movies that came out in 1993 or later, what are the 25 movies with the most ratings per year, and what is the average rating of each of the top 25 movies?

```{r, include=TRUE}
movielens %>% 
	filter(year >= 1993) %>%
	group_by(movieId) %>%
	summarize(n = n(), years = 2018 - first(year),
				title = title[1],
				rating = mean(rating)) %>%
	mutate(rate = n/years) %>%
	top_n(25, rate) %>%
	arrange(desc(rate)) 
```

- What is the average rating for the movie The Shawshank Redemption? 4.49
- What is the average number of ratings per year for the movie Forrest Gump? 14.2



### **Question 3**

From the table constructed in Q2, we can see that the most frequently rated movies tend to have above average ratings. This is not surprising: more people watch popular movies. To confirm this, stratify the post-1993 movies by ratings per year and compute their average ratings. Make a plot of average rating versus ratings per year and show an estimate of the trend.

What type of trend do you observe?

```{r, include=TRUE}
movielens %>% 
	filter(year >= 1993) %>%
	group_by(movieId) %>%
	summarize(n = n(), years = 2017 - first(year),
				title = title[1],
				rating = mean(rating)) %>%
	mutate(rate = n/years) %>%
	ggplot(aes(rate, rating)) +
	geom_point() +
	geom_smooth()
```

- There is no relationship between how often a movie is rated and its average rating.
- Movies with very few and very many ratings have the highest average ratings.
- The more often a movie is rated, the higher its average rating. [X]
- The more often a movie is rated, the lower its average rating.


### **Question 4**

Suppose you are doing a predictive analysis in which you need to fill in the missing ratings with some value.

Given your observations in the exercise in Q3, which of the following strategies would be most appropriate?

- Fill in the missing values with the average rating across all movies.
- Fill in the missing values with 0.
- Fill in the missing values with a lower value than the average rating across all movies. [X]
- Fill in the value with a higher value than the average rating across all movies.
- None of the above. 


### **Question 5**

The ```movielens``` dataset also includes a time stamp. This variable represents the time and data in which the rating was provided. The units are seconds since January 1, 1970. Create a new column ```date``` with the date.

Which code correctly creates this new column?

```{r, include=TRUE}
library(lubridate)
movielens <- mutate(movielens, date = as_datetime(timestamp))
```

- ```movielens <- mutate(movielens, date = as.date(timestamp))```
- ```movielens <- mutate(movielens, date = as_datetime(timestamp))``` [X]
- ```movielens <- mutate(movielens, date = as.data(timestamp))```
- ```movielens <- mutate(movielens, date = timestamp)```


### **Question 6**

Compute the average rating for each week and plot this average against day. Hint: use the round_date function before you group_by.

What type of trend do you observe?

```{r, include=TRUE}
movielens %>% mutate(date = round_date(date, unit = "week")) %>%
	group_by(date) %>%
	summarize(rating = mean(rating)) %>%
	ggplot(aes(date, rating)) +
	geom_point() +
	geom_smooth()
```

- There is strong evidence of a time effect on average rating.
- There is some evidence of a time effect on average rating. [X]
- There is no evidence of a time effect on average rating.

### **Question 7**

Consider again the plot you generated in Q6.

If we define $\ d_{u,i}$ as the day for user's $\ u$ rating of movie $\ i$, which of the following models is most appropriate?

- $\ Y_{u,i} = \mu + b_i + b_u + d_{u,i} + \varepsilon_{u,i}$
- $\ Y_{u,i} = \mu + b_i + b_u + d_{u,i}\beta + \varepsilon_{u,i}$
- $\ Y_{u,i} = \mu + b_i + b_u + d_{u,i}\beta_i + \varepsilon_{u,i}$
- $\ Y_{u,i} = \mu + b_i + b_u + f(d_{u,i}) + \varepsilon_{u,i}$, with $\ f$ a smooth function of  $\ d_{u,i}$ [X]

### **Question 8**

The movielens data also has a genres column. This column includes every genre that applies to the movie. Some movies fall under several genres. Define a category as whatever combination appears in this column. Keep only categories with more than 1,000 ratings. Then compute the average and standard error for each category. Plot these as error bar plots.

```{r, include=TRUE}
movielens %>% group_by(genres) %>%
	summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
	filter(n >= 1000) %>% 
	mutate(genres = reorder(genres, avg)) %>%
	ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
	geom_point() +
	geom_errorbar() + 
	theme(axis.text.x = element_text(angle = 90, hjust = 1))
```

Which genre has the lowest average rating? Comedy


### **Question 9**

The plot you generated in Q8 shows strong evidence of a genre effect. Consider this plot as you answer the following question.

If we define $\ g_{u,i}$ as the day for user's $\ u$ rating of movie $\ i$, which of the following models is most appropriate?

-  $\ Y_{u,i} = \mu + b_i + b_u + g_{u,i} + \varepsilon_{u,i}$
-  $\ Y_{u,i} = \mu + b_i + b_u + g_{u,i}\beta + \varepsilon_{u,i}$
-  $\ Y_{u,i} = \mu + b_i + b_u + \sum{k=1}^K x_{u,i} \beta_k + \varepsilon_{u,i}$, with x^k_{u,i}$ = 1 if $\ g_{u,i}$ is genre $\ k$ [X]
-  $\ Y_{u,i} = \mu + b_i + b_u + f(g_{u,i}) + \varepsilon_{u,i}$, with  $\ f$ a smooth function of  $\ g_{u,i}$


## **Regularization**

The exercises in Q1-Q8 work with a simulated dataset for 100 schools. This pre-exercise setup walks you through the code needed to simulate the dataset.

An education expert is advocating for smaller schools. The expert bases this recommendation on the fact that among the best performing schools, many are small schools. Let's simulate a dataset for 100 schools. First, let's simulate the number of students in each school, using the following code:

```{r, include=TRUE}
set.seed(1986)
n <- round(2^rnorm(1000, 8, 1))
```

Now let's assign a true quality for each school that is completely independent from size. This is the parameter we want to estimate in our analysis. The true quality can be assigned using the following code:

```{r, include=TRUE}
set.seed(1)
mu <- round(80 + 2*rt(1000, 5))
range(mu)
schools <- data.frame(id = paste("PS",1:100),
                      size = n,
                      quality = mu,
                      rank = rank(-mu))
```

We can see the top 10 schools using this code: 

```{r, include=TRUE}
schools %>% top_n(10, quality) %>% arrange(desc(quality))
```

Now let's have the students in the school take a test. There is random variability in test taking, so we will simulate the test scores as normally distributed with the average determined by the school quality with a standard deviation of 30 percentage points. This code will simulate the test scores:

```{r, include=TRUE}
set.seed(1)
scores <- sapply(1:nrow(schools), function(i){
       scores <- rnorm(schools$size[i], schools$quality[i], 30)
       scores
})
schools <- schools %>% mutate(score = sapply(scores, mean))
```

### **Question 1**

What are the top schools based on the average score? Show just the ID, size, and the average score.

Report the ID of the top school and average score of the 10th school.

```{r, include=TRUE}
schools %>% top_n(10, score) %>% arrange(desc(score)) %>% select(id, size, score)
```

- What is the ID of the top school? 67
- What is the average score of the 10th school? 88.09490

 
### **Question 2**

Compare the median school size to the median school size of the top 10 schools based on the score.

```{r, include=TRUE}
median(schools$size)
schools %>% top_n(10, score) %>% .$size %>% median()
``` 

- What is the median school size overall? 261
- What is the median school size of the of the top 10 schools based on the score? 136



### **Question 3**

According to this analysis, it appears that small schools produce better test scores than large schools. Four out of the top 10 schools have 100 or fewer students. But how can this be? We constructed the simulation so that quality and size were independent. Repeat the exercise for the worst 10 schools.

```{r, include=TRUE}
median(schools$size)
schools %>% top_n(-10, score) %>% .$size %>% median()
```
What is the median school size of the bottom 10 schools based on the score? 146


### **Question 4**

From this analysis, we see that the worst schools are also small. Plot the average score versus school size to see what's going on. Highlight the top 10 schools based on the true quality. Use a log scale to transform for the size.

```{r, include=TRUE}
schools %>% ggplot(aes(size, score)) +
	geom_point(alpha = 0.5) +
	geom_point(data = filter(schools, rank<=10), col = 2) 
```

What do you observe?

- There is no difference in the standard error of the score based on school size; there must be an error in how we generated our data.
- The standard error of the score has larger variability when the school is smaller, which is why both the best and the worst schools are more likely to be small. [X]
- The standard error of the score has smaller variability when the school is smaller, which is why both the best and the worst schools are more likely to be small.
- The standard error of the score has larger variability when the school is very small or very large, which is why both the best and the worst schools are more likely to be small.
- The standard error of the score has smaller variability when the school is very small or very large, which is why both the best and the worst schools are more likely to be small.

### **Question 5**

Let's use regularization to pick the best schools. Remember regularization shrinks deviations from the average towards 0. To apply regularization here, we first need to define the overall average for all schools, using the following code:

```{r, include=TRUE}
overall <- mean(sapply(scores, mean))
```

Then, we need to define, for each school, how it deviates from that average.

Write code that estimates the score above the average for each school but dividing by $\ n + \alpha$ instead of $\ n$, with $\ n$ the schools size and $\ \alpha$ a regularization parameters. Try $\ \alpha = 25$.

```{r, include=TRUE}
alpha <- 25
score_reg <- sapply(scores, function(x)  overall + sum(x-overall)/(length(x)+alpha))
schools %>% mutate(score_reg = score_reg) %>%
	top_n(10, score_reg) %>% arrange(desc(score_reg))
```

What is the ID of the top school with regularization? 91
What is the regularized score of the 10th school? 86.90070


### **Question 6**

Notice that this improves things a bit. The number of small schools that are not highly ranked is now lower. Is there a better $\ \alpha$? Find the $\ \alpha$ that minimizes the RMSE = $\ \frac{1}{100} \sum_{i=1}^{100} (\mbox{quality} - \mbox{estimate})^2$.

```{r, include=TRUE}
alphas <- seq(10,250)
rmse <- sapply(alphas, function(alpha){
	score_reg <- sapply(scores, function(x) overall+sum(x-overall)/(length(x)+alpha))
	mean((score_reg - schools$quality)^2)
})
plot(alphas, rmse)
alphas[which.min(rmse)]  
```

What value of  gives the minimum RMSE? 128


### **Question 7**

Rank the schools based on the average obtained with the best $\ \alpha$. Note that no small school is incorrectly included.

```{r, include=TRUE}
alpha <- alphas[which.min(rmse)]  
score_reg <- sapply(scores, function(x)
	overall+sum(x-overall)/(length(x)+alpha))
schools %>% mutate(score_reg = score_reg) %>%
	top_n(10, score_reg) %>% arrange(desc(score_reg))
```

- What is the ID of the top school now? 91
- What is the regularized average score of the 10th school now? 85.35335

 
### **Question 8**

A common mistake made when using regularization is shrinking values towards 0 that are not centered around 0. For example, if we don't subtract the overall average before shrinking, we actually obtain a very similar result. Confirm this by re-running the code from the exercise in Q6 but without removing the overall mean.

```{r, include=TRUE}
alphas <- seq(10,250)
rmse <- sapply(alphas, function(alpha){
	score_reg <- sapply(scores, function(x) sum(x)/(length(x)+alpha))
	mean((score_reg - schools$quality)^2)
})
plot(alphas, rmse)
alphas[which.min(rmse)] 
```

What value of $\ \alpha$ gives the minimum RMSE here? 10


## **Matrix Factorization**

In this exercise set, we will be covering a topic useful for understanding matrix factorization: the singular value decomposition (SVD). SVD is a mathematical result that is widely used in machine learning, both in practice and to understand the mathematical properties of some algorithms. This is a rather advanced topic and to complete this exercise set you will have to be familiar with linear algebra concepts such as matrix multiplication, orthogonal matrices, and diagonal matrices.

The SVD tells us that we can decompose an $\ N\times p$ matrix $\ Y$ with $\ p < N$ as $\ Y = U D V^{\top}$

with $\ U$ and $\ V$ orthogonal of dimensions $\ N\times p$ and $\ p\times p$ respectively and$\  D a$\  p\times p$ diagonal matrix with the values of the diagonal decreasing: 
$\ d_{1,1} \geq d_{2,2} \geq \dots d_{p,p}$

In this exercise, we will see one of the ways that this decomposition can be useful. To do this, we will construct a dataset that represents grade scores for 100 students in 24 different subjects. The overall average has been removed so this data represents the percentage point each student received above or below the average test score. So a 0 represents an average grade (C), a 25 is a high grade (A+), and a -25 represents a low grade (F). You can simulate the data like this:

```{r, include=TRUE}
set.seed(1987)
n <- 100
k <- 8
Sigma <- 64  * matrix(c(1, .75, .5, .75, 1, .5, .5, .5, 1), 3, 3) 
m <- MASS::mvrnorm(n, rep(0, 3), Sigma)
m <- m[order(rowMeans(m), decreasing = TRUE),]
y <- m %x% matrix(rep(1, k), nrow = 1) + matrix(rnorm(matrix(n*k*3)), n, k*3)
colnames(y) <- c(paste(rep("Math",k), 1:k, sep="_"),
                 paste(rep("Science",k), 1:k, sep="_"),
                 paste(rep("Arts",k), 1:k, sep="_"))
```

Our goal is to describe the student performances as succinctly as possible. For example, we want to know if these test results are all just a random independent numbers. Are all students just about as good? Does being good in one subject  imply you will be good in another? How does the SVD help with all this? We will go step by step to show that with just three relatively small pairs of vectors we can explain much of the variability in this 100 x 24 dataset. 

### **Question 1**

You can visualize the 24 test scores for the 100 students by plotting an image:

```{r, include=TRUE}
my_image <- function(x, zlim = range(x), ...){
	colors = rev(RColorBrewer::brewer.pal(9, "RdBu"))
	cols <- 1:ncol(x)
	rows <- 1:nrow(x)
	image(cols, rows, t(x[rev(rows),,drop=FALSE]), xaxt = "n", yaxt = "n",
			xlab="", ylab="",  col = colors, zlim = zlim, ...)
	abline(h=rows + 0.5, v = cols + 0.5)
	axis(side = 1, cols, colnames(x), las = 2)
}
my_image(y)
```

How would you describe the data based on this figure?

- The test scores are all independent of each other.
- The students that are good at math are not good at science.
- The students that are good at math are not good at arts.
- The students that test well are at the top of the image and there seem to be three groupings by subject. [X]
- The students that test well are at the bottom of the image and there seem to be three groupings by subject.

### **Question 2**

You can examine the correlation between the test scores directly like this:

```{r, include=TRUE}
my_image(cor(y), zlim = c(-1,1))
range(cor(y))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)
```

Which of the following best describes what you see?

- The test scores are independent.
- Test scores in math and science are highly correlated but scores in arts are not.
- There is high correlation between tests in the same subject but no correlation across subjects.
- There is correlation among all tests, but higher if the tests are in science and math and even higher within each subject. [X]

### **Question 3**

Remember that orthogonality means that $\ U^{\top}U$ and $\ V^{\top}V$ are equal to the identity matrix. This implies that we can also rewrite the decomposition as

$\ Y V = U D \mbox{ or } U^{\top}Y = D V^{\top}$

We can think of $\ YV$ and $\ U^{\top}V as two transformations of $\ Y$ that preserve the total variability of $\ Y$ since $\ U$ and $\ V$ are orthogonal.

Use the function ```svd``  to compute the SVD of ```y```. This function will return $\ U, V$, and the diagonal entries of $\ D$.
```{r, include=TRUE}
s <- svd(y)
names(s)
```
You can check that the SVD works by typing:
```{r, include=TRUE}
y_svd <- s$u %*% diag(s$d) %*% t(s$v)
max(abs(y - y_svd))
```

Compute the sum of squares of the columns of $\ Y$ and store them in ```ss_y```. Then compute the sum of squares of columns of the transformed YV and store them in ```ss_yv```. Confirm that ```sum(ss_y)``` is equal to ```sum(ss_yv)```.

```{r, include=TRUE}
y_sq <- y*y 
ss_y <- colSums(y_sq)
sum(ss_y) 
```

```{r, include=TRUE}
y_svd_sq <- y_svd*y_svd 
ss_yv <- colSums(y_svd_sq)
sum(ss_yv) 
```

What is the value of ```sum(ss_y)``` (and also the value of ```sum(ss_yv)```)? 175435

 
### **Question 4**

We see that the total sum of squares is preserved. This is because $\ V$ is orthogonal. Now to start understanding how $\ YV$ is useful, plot ```ss_y``` against the column number and then do the same for ```ss_yv```.

What do you observe?

```{r, include=TRUE}
plot(ss_y) 
plot(ss_yv)
```

- ```ss_y``` and ```ss_y```v are decreasing and close to 0 for the 4th column and beyond.
- ```ss_yv``` is decreasing and close to 0 for the 4th column and beyond. [X]
- ```ss_y``` is decreasing and close to 0 for the 4th column and beyond.
- There is no discernible pattern to either ```ss_y``` or ```ss_yv```.


### **Question 5**

Note that we didn't have to compute ```ss_yv``` because we already have the answer. How? Remember that $\ YV = UD$ and because $\ U$ is orthogonal, we know that the sum of squares of the columns of $\ UD$ are the diagonal entries of $\ D$ squared. Confirm this by plotting the square root of ```ss_y```v versus the diagonal entries of $\ D$.

What else is equal to $\ YV$?

```{r, include=TRUE}
plot(sqrt(ss_yv), s$d)
abline(0,1)
```

- D
- U
- UD [X]
- VUD


### **Question 6**

So from the above we know that the sum of squares of the columns of $\ Y$ (the total sum of squares) adds up to the sum of ```s$d^2``` and that the transformation $\ YV$ gives us columns with sums of squares equal to ```s$d^2```. Now compute the percent of the total variability that is explained by just the first three columns of $\ YV$.

```{r, include=TRUE}
sum(s$d[1:3]^2) / sum(s$d^2)
```

What proportion of the total variability is explained by the first three columns of $\ YV$? 0.988

 
### **Question 7**

Before we continue, let's show a useful computational trick to avoid creating the matrix ```diag(s$d)```. To motivate this, we note that if we write out in its columns $\ [U_1, U_2, \dots, U_p]$ then $\ UD$ is equal to $\ UD = [ U_1 d_{1,1}, U_2 d_{2,2}, \dots, U_p d_{p,p}]$

Use the ```sweep``` function to compute $\ UD$ without constructing ```diag(s$d)``` or using matrix multiplication.

Which code is correct?

```{r, include=TRUE}
identical(s$u %*% diag(s$d), sweep(s$u, 2, s$d, FUN = "*"))
```

- ```identical(t(s$u %*% diag(s$d)), sweep(s$u, 2, s$d, FUN = "*"))```
- ```identical(s$u %*% diag(s$d), sweep(s$u, 2, s$d, FUN = "*"))``` [X]
- ```identical(s$u %*% t(diag(s$d)), sweep(s$u, 2, s$d, FUN = "*"))```
- ```identical(s$u %*% diag(s$d), sweep(s$u, 2, s, FUN = "*"))```

### **Question 8**

We know that $\ U_1 d_{1,1}$, the first column of $\ UD$, has the most variability of all the columns of $\ UD$. Earlier we looked at an image of $\ Y$ using ```my_image(y)```, in which we saw that the student to student variability is quite large and that students that are good in one subject tend to be good in all. This implies that the average (across all subjects) for each student should explain a lot of the variability. Compute the average score for each student, plot it against $\ U_1 d_{1,1}$, and describe what you find.

What do you observe?

```{r, include=TRUE}
plot(-s$u[,1]*s$d[1], rowMeans(y))
```

- There is no relationship between the average score for each student and .
- There is a linearly decreasing relationship between the average score for each student and $\ U_1 d_{1,1}$.
- There is a linearly increasing relationship between the average score for each student and $\ U_1 d_{1,1}$. [X]
- There is an exponentially increasing relationship between the average score for each student and $\ U_1 d_{1,1}$.
- There is an exponentially decreasing relationship between the average score for each student and $\ U_1 d_{1,1}$.
Explanation


### **Question 9**

We note that the signs in SVD are arbitrary because:

$\ U D V^{\top} = (-U) D (-V)^{\top}$

With this in mind we see that the first column of $\ UD$ is almost identical to the average score for each student except for the sign.

This implies that multiplying $\ Y$ by the first column of $\ V$ must be performing a similar operation to taking the average. Make an image plot of $\ V$ and describe the first column relative to others and how this relates to taking an average.

How does the first column relate to the others, and how does this relate to taking an average?

```{r, include=TRUE}
my_image(s$v)
```

- The first column is very variable, which implies that the first column of YV is the sum of the rows of Y multiplied by some non-constant function, and is thus not proportional to an average.
- The first column is very variable, which implies that the first column of YV is the sum of the rows of Y multiplied by some non-constant function, and is thus proportional to an average. 
The first column is very close to being a constant, which implies that the - first column of YV is the sum of the rows of Y multiplied by some constant, and is thus proportional to an average. [X]
- The first three columns are all very close to being a constant, which implies that these columns are the sum of the rows of Y multiplied by some constant, and are thus proportional to an average.


### **Question 10**

We already saw that we can rewrite $\ UD$ as $\ U_1 d_{1,1} + U_2 d_{2,2} + \dots + U_p d_{p,p}$

with $\ U_j$ the j-th column of $\ U$. This implies that we can rewrite the entire SVD as:

$\ Y = U_1 d_{1,1} V_1 ^{\top} + U_2 d_{2,2} V_2 ^{\top} + \dots + U_p d_{p,p} V_p ^{\top}$

with $\ V_j $the jth column of $\ V$. Plot $\ U_1$, then plot $\ V_1^{\top}$ using the same range for the y-axis limits, then make an image of $\ U_1 d_{1,1} V_1 ^{\top}$ and compare it to the image of $\ Y$. Hint: use the ```my_image``` function defined above. Use the ```drop=FALSE``` argument to assure the subsets of matrices are matrices.

```{r, include=TRUE}
plot(s$u[,1], ylim = c(-0.25, 0.25))
plot(s$v[,1], ylim = c(-0.25, 0.25))
with(s, my_image((u[, 1, drop=FALSE]*d[1]) %*% t(v[, 1, drop=FALSE])))
my_image(y)
```

### **Question 11**

We see that with just a vector of length 100, a scalar, and a vector of length 24, we can actually come close to reconstructing the a 100 x 24 matrix. This is our first matrix factorization:

$\ Y \approx d_{1,1} U_1 V_1^{\top}$

In the exercise in Q6, we saw how to calculate the percent of total variability explained. However, our approximation only explains the observation that good students tend to be good in all subjects. Another aspect of the original data that our approximation does not explain was the higher similarity we observed within subjects. We can see this by computing the difference between our approximation and original data and then computing the correlations. You can see this by running this code:

```{r, include=TRUE}
resid <- y - with(s,(u[, 1, drop=FALSE]*d[1]) %*% t(v[, 1, drop=FALSE]))
my_image(cor(resid), zlim = c(-1,1))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)
```

Now that we have removed the overall student effect, the correlation plot reveals that we have not yet explained the within subject correlation nor the fact that math and science are closer to each other than to the arts. So let's explore the second column of the SVD.

Repeat the previous exercise (Q10) but for the second column: Plot $\ U_2$, then plot $\ V_2^{\top}$ using the same range for the y-axis limits, then make an image of $\ U_2 d_{2,2} V_2 ^{\top}$ and compare it to the image of ```resid`` .
```{r, include=TRUE}
plot(s$u[,2], ylim = c(-0.5, 0.5))
plot(s$v[,2], ylim = c(-0.5, 0.5))
with(s, my_image((u[, 2, drop=FALSE]*d[2]) %*% t(v[, 2, drop=FALSE])))
my_image(resid)
```
### **Question 12**
The second column clearly relates to a student's difference in ability in math/science versus the arts. We can see this most clearly from the plot of ```s$v[,2]```. Adding the matrix we obtain with these two columns will help with our approximation:
$\ Y \approx d_{1,1} U_1 V_1^{\top} + d_{2,2} U_2 V_2^{\top}$
We know it will explain ```sum(s$d[1:2]^2)/sum(s$d^2) * 100``` percent of the total variability. We can compute new residuals like this:
```{r, include=TRUE}
resid <- y - with(s,sweep(u[, 1:2], 2, d[1:2], FUN="*") %*% t(v[, 1:2]))
my_image(cor(resid), zlim = c(-1,1))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)
```

and see that the structure that is left is driven by the differences between math and science. Confirm this by first plotting $\ U_3$, then plotting $\ V_3^{\top}$ using the same range for the y-axis limits, then making an image of $\ U_3 d_{3,3} V_3 ^{\top}$ and comparing it to the image of ```resid```.


```{r, include=TRUE}
plot(s$u[,3], ylim = c(-0.5, 0.5))
plot(s$v[,3], ylim = c(-0.5, 0.5))
with(s, my_image((u[, 3, drop=FALSE]*d[3]) %*% t(v[, 3, drop=FALSE])))
my_image(resid)
```

### **Question 13**

The third column clearly relates to a student's difference in ability in math and science. We can see this most clearly from the plot of ```s$v[,3]```. Adding the matrix we obtain with these two columns will help with our approximation:

$\ Y \approx d_{1,1} U_1 V_1^{\top} + d_{2,2} U_2 V_2^{\top} + d_{3,3} U_3 V_3^{\top}$

We know it will explain: sum(s$d[1:3]^2)/sum(s$d^2) * 100 percent of the total variability. We can compute new residuals like this:

```{r, include=TRUE}
resid <- y - with(s,sweep(u[, 1:3], 2, d[1:3], FUN="*") %*% t(v[, 1:3]))
my_image(cor(resid), zlim = c(-1,1))
axis(side = 2, 1:ncol(y), rev(colnames(y)), las = 2)
```

We no longer see structure in the residuals: they seem to be independent of each other. This implies that we can describe the data with the following model:

$\ Y =  d_{1,1} U_1 V_1^{\top} + d_{2,2} U_2 V_2^{\top} + d_{3,3} U_3 V_3^{\top} + \varepsilon$

with $\ \varepsilon$ a matrix of independent identically distributed errors. This model is useful because we summarize of 100 x 24 observations with 3 X (100+24+1) = 375 numbers.

Furthermore, the three components of the model have useful interpretations:

1 - the overall ability of a student
2 - the difference in ability between the math/sciences and arts
3 - the remaining differences between the three subjects.

The sizes $\ d_{1,1}, d_{2,2}$ and $\ d_{3,3$} tell us the variability explained by each component. Finally, note that the components $\ d_{j,j} U_j V_j^{\top}$ are equivalent to the jth principal component.

Finish the exercise by plotting an image of $\ Y$, an image of $\ d_{1,1} U_1 V_1^{\top} + d_{2,2} U_2 V_2^{\top} + d_{3,3} U_3 V_3^{\top}$ and an image of the residuals, all with the same ```zlim```.

```{r, include=TRUE}
y_hat <- with(s,sweep(u[, 1:3], 2, d[1:3], FUN="*") %*% t(v[, 1:3]))
my_image(y, zlim = range(y))
my_image(y_hat, zlim = range(y))
my_image(y - y_hat, zlim = range(y))
```

## **Clustering**

These exercises will work with the tissue_gene_expression dataset, which is part of the dslabs package.

### **Question 1**

Load the ```tissue_gene_expression``` dataset. Remove the row means and compute the distance between each observation. Store the result in ```d```.

Which of the following lines of code correctly does this computation?

```{r, include=TRUE}
d <- dist(tissue_gene_expression$x - rowMeans(tissue_gene_expression$x))
```

- ```d <- dist(tissue_gene_expression$x)```
- ```d <- dist(rowMeans(tissue_gene_expression$x))```
- ```d <- dist(rowMeans(tissue_gene_expression$y))```
- ```d <- dist(tissue_gene_expression$x - rowMeans(tissue_gene_expression$x))```[X]

### **Question 2**

Make a hierarchical clustering plot and add the tissue types as labels.

You will observe multiple branches.

Which tissue type is in the branch farthest to the left?

```{r, include=TRUE}
h <- hclust(d)
plot(h)
```
- cerebellum
- colon
- endometrium
- hippocampus
- kidney
- liver [X]
- placenta


### **Question 3**

Run a k-means clustering on the data with $\ K = 7$. Make a table comparing the identified clusters to the actual tissue types. Run the algorithm several times to see how the answer changes.

What do you observe for the clustering of the liver tissue?

```{r, include=TRUE}
cl <- kmeans(tissue_gene_expression$x, centers = 7)
table(cl$cluster, tissue_gene_expression$y)
```

- Liver is always classified in a single cluster.
- Liver is never classified in a single cluster.
- Liver is classified in a single cluster roughly 20% of the time and in more than one cluster roughly 80% of the time. [X]
- Liver is classified in a single cluster roughly 80% of the time and in more than one cluster roughly 20% of the time.


### **Question 4**

Select the 50 most variable genes. Make sure the observations show up in the columns, that the predictor are centered, and add a color bar to show the different tissue types. Hint: use the ColSideColors argument to assign colors. Also, use col = RColorBrewer::brewer.pal(11, "RdBu") for a better use of colors.

Part of the code is provided for you here:

```{r, include=TRUE}
library(RColorBrewer)
sds <- matrixStats::colSds(tissue_gene_expression$x)
ind <- order(sds, decreasing = TRUE)[1:50]
colors <- brewer.pal(7, "Dark2")[as.numeric(tissue_gene_expression$y)]
heatmap(t(tissue_gene_expression$x[,ind]), col = brewer.pal(11, "RdBu"), scale = "row", ColSideColors = colors)
```

Which line of code should replace #BLANK in the code above?

- ```heatmap(t(tissue_gene_expression$x[,ind]), col = brewer.pal(11, "RdBu"), scale = "row", ColSideColors = colors)``` [X]
- ```heatmap(t(tissue_gene_expression$x[,ind]), col = brewer.pal(11, "RdBu"), scale = "row", ColSideColors = rev(colors))```
- ```heatmap(t(tissue_gene_expression$x[,ind]), col = brewer.pal(11, "RdBu"), scale = "row", ColSideColors = sample(colors))```
- ```heatmap(t(tissue_gene_expression$x[,ind]), col = brewer.pal(11, "RdBu"), scale = "row", ColSideColors = sample(colors))```

