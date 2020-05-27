################################
# MovieLens recommendation system
# Harvard Data Science Capstone Project
# Elin Brännström

# This project is bulit on the dslabs Movielens data for modelling a recommendation system. 
# The final model is based on regularization and biases found from data exploration. Full report is attached with the code.
################################


## Create data sets and tidy the data ##

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Examine what columns we have in the data
head(edx)

# Examine data quality and missing values
which(is.na(edx))

# Making data into tidy format by reading out release year
edx <- edx %>% mutate(year_release=str_sub(title,-5,-2))
validation <- validation %>% mutate(year_release=str_sub(title,-5,-2))

#Create new partition for training & testing of edx data, using 80% for training and 20% for testing
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times=1, p=0.2, list=FALSE)
training_set <- edx[-test_index,]
test_set <- edx[test_index,]

#Remove users and movies from test set that are not included in training set
test_set <- test_set %>% semi_join(training_set, by="movieId") %>% semi_join(training_set, by ="userId")

#Put removed lines back into training set
removed_new <- anti_join(training_set,test_set)
training_set <- rbind(training_set,removed_new)

## VISUALIZATION OF THE DATA ## 

# Examine if there is a movie bias
mu_hat <- mean(edx$rating)
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

# Examine if there is a user bias
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# Examine if there is a genre bias - note that genres column is kept as combinations of genres
edx %>% group_by(genres) %>% summarize(b_g = mean(rating)) %>% ggplot(aes(b_g)) + geom_histogram(color = "black")


# Examine if there is a release year bias
edx %>% group_by(year_release) %>% summarize(b_y = mean(rating)) %>% ggplot(aes(b_y)) + geom_histogram(color = "black")

## MODELLING ##

#Firstly, let's define a RMSE-function for calculating the root mean square error between predictions and observations
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))}

#Predict using the mean value
mu_hat <- mean(training_set$rating)
mean_prediction <- RMSE(test_set$rating,mu_hat)

results_rmse <- data_frame(method = "Using the average for prediction" , RMSE=mean_prediction)
results_rmse %>% knitr::kable()

#Adding the b_i term accounting for that different movies are rated differently
movie_avgs <- training_set %>% group_by(movieId) %>% summarize(bi=mean(rating - mu_hat))
test_set_with_bi <- test_set %>% left_join(movie_avgs, by="movieId")
predictions_bi <- test_set_with_bi$bi+mu_hat
bi_prediction <- RMSE(test_set$rating, predictions_bi)
results_rmse <- bind_rows(results_rmse, tibble(method="Adding movie bias", RMSE=bi_prediction))
results_rmse %>% knitr::kable()

#Adding the b_u term to account for difference in user ratings
user_avgs <- training_set %>% left_join(movie_avgs, by="movieId") %>% group_by(userId) %>% summarize(b_u=mean(rating-mu_hat-bi))
test_set_with_bi_bu <- test_set_with_bi %>% left_join(user_avgs, by="userId")
test_set_with_bu <- test_set %>% left_join(user_avgs, by="userId")
predictions_bu <- test_set_with_bi_bu$bi+test_set_with_bu$b_u+mu_hat
bu_predictions <- RMSE(test_set$rating, predictions_bu)
results_rmse <- bind_rows(results_rmse, tibble(method="Adding user bias", RMSE=bu_predictions))
results_rmse %>% knitr::kable()

#Use cross-validation with movie - and user bias to receive even better rmse
lambdas <- seq(3,10,0.5)
sum_movieId <- training_set %>% group_by(movieId) %>% summarize(s=sum(rating-mu_hat), n=n())
rmses <- sapply(lambdas, function(lambda){
  b_i_crossv <- training_set %>% group_by(movieId)%>% summarize(b_i_crossv=sum(rating-mu_hat)/(n()+lambda))
  b_u_crossv <- training_set %>% left_join(b_i_crossv,by="movieId") %>% group_by(userId) %>% summarize(b_u_crossv=sum(rating-mu_hat-b_i_crossv)/(n()+lambda))
  predictions <- test_set %>% left_join(b_i_crossv, by="movieId") %>% left_join(b_u_crossv, by="userId") %>%
    mutate(preds = b_i_crossv+b_u_crossv+mu_hat)
  
  return(RMSE(test_set$rating,predictions$preds))
})
results_rmse <- bind_rows(results_rmse, tibble(method="Cross-validation using movie - and user bias", RMSE=min(rmses)))
results_rmse %>% knitr::kable()

# Include genre effect into cross-validation
lambdas <- seq(3,10,0.5)
sum_movieId <- training_set %>% group_by(movieId) %>% summarize(s=sum(rating-mu_hat), n=n())
rmses <- sapply(lambdas, function(lambda){
  b_i_crossv <- training_set %>% group_by(movieId)%>% summarize(b_i_crossv=sum(rating-mu_hat)/(n()+lambda))
  b_u_crossv <- training_set %>% left_join(b_i_crossv,by="movieId") %>% group_by(userId) %>% summarize(b_u_crossv=sum(rating-mu_hat-b_i_crossv)/(n()+lambda))
  b_g_crossv <- training_set %>% left_join(b_i_crossv,by="movieId") %>% left_join(b_u_crossv, by="userId") %>% group_by(genres) %>% summarize(b_g_crossv=sum(rating-mu_hat-b_i_crossv-b_u_crossv)/(n()+lambda))
  predictions <- test_set %>% left_join(b_i_crossv, by="movieId") %>% left_join(b_u_crossv, by="userId") %>% left_join(b_g_crossv, by="genres") %>%
    mutate(preds = b_i_crossv+b_u_crossv+b_g_crossv+mu_hat)
  
  return(RMSE(test_set$rating,predictions$preds))
})
results_rmse <- bind_rows(results_rmse, tibble(method="Cross-validation using movie, user & genre bias", RMSE=min(rmses)))
results_rmse %>% knitr::kable()

#Include release year into cross-validation
lambdas <- seq(3,10,0.5)
sum_movieId <- training_set %>% group_by(movieId) %>% summarize(s=sum(rating-mu_hat), n=n())
rmses <- sapply(lambdas, function(lambda){
  b_i_crossv <- training_set %>% group_by(movieId)%>% summarize(b_i_crossv=sum(rating-mu_hat)/(n()+lambda))
  b_u_crossv <- training_set %>% left_join(b_i_crossv,by="movieId") %>% group_by(userId) %>% summarize(b_u_crossv=sum(rating-mu_hat-b_i_crossv)/(n()+lambda))
  b_g_crossv <- training_set %>% left_join(b_i_crossv,by="movieId") %>% left_join(b_u_crossv, by="userId") %>% group_by(genres) %>% summarize(b_g_crossv=sum(rating-mu_hat-b_i_crossv-b_u_crossv)/(n()+lambda))
  b_y_crossv <- training_set %>% left_join(b_i_crossv,by="movieId") %>% left_join(b_u_crossv, by="userId") %>% left_join(b_g_crossv, by="genres") %>% group_by(year_release) %>% summarize(b_y_crossv=sum(rating-mu_hat-b_i_crossv-b_u_crossv-b_g_crossv)/(n()+lambda))
  predictions <- test_set %>% left_join(b_i_crossv, by="movieId") %>% left_join(b_u_crossv, by="userId") %>% left_join(b_g_crossv, by="genres") %>% left_join(b_y_crossv, by="year_release") %>%
    mutate(preds = b_i_crossv+b_u_crossv+b_g_crossv+b_y_crossv+mu_hat)
  
  return(RMSE(test_set$rating,predictions$preds))
})
results_rmse <- bind_rows(results_rmse, tibble(method="Cross-validation using movie, user, genre & release year bias", RMSE=min(rmses)))
results_rmse %>% knitr::kable()



#Using validation set for final rmse
lambdas <- lambdas[which.min(rmses)] #SELECT OPTIMAL LAMBDA
rmses <- sapply(lambdas, function(lambda){
  mu_hat <- mean(edx$rating)
  b_i_crossv <- edx %>% group_by(movieId)%>% summarize(b_i_crossv=sum(rating-mu_hat)/(n()+lambda))
  b_u_crossv <- edx %>% left_join(b_i_crossv,by="movieId") %>% group_by(userId) %>% summarize(b_u_crossv=sum(rating-mu_hat-b_i_crossv)/(n()+lambda))
  b_g_crossv <- edx %>% left_join(b_i_crossv,by="movieId") %>% left_join(b_u_crossv, by="userId") %>% group_by(genres) %>% summarize(b_g_crossv=sum(rating-mu_hat-b_i_crossv-b_u_crossv)/(n()+lambda))
  b_y_crossv <- edx %>% left_join(b_i_crossv,by="movieId") %>% left_join(b_u_crossv, by="userId") %>% left_join(b_g_crossv, by="genres") %>% group_by(year_release) %>% summarize(b_y_crossv=sum(rating-mu_hat-b_i_crossv-b_u_crossv-b_g_crossv)/(n()+lambda))
  predictions <- validation %>% left_join(b_i_crossv, by="movieId") %>% left_join(b_u_crossv, by="userId") %>% left_join(b_g_crossv, by="genres") %>% left_join(b_y_crossv, by="year_release") %>%
    mutate(preds = b_i_crossv+b_u_crossv+b_g_crossv+b_y_crossv+mu_hat)
  
  return(RMSE(predictions$rating,predictions$preds))
})
results_rmse <- bind_rows(results_rmse, tibble(method="Cross-validation using movie, user, genre & release year bias on validation set", RMSE=rmses))
results_rmse %>% knitr::kable()
