#################Import Necessary Packages#################


library(jsonlite)
library(ggplot2)
library(stringr)
library(dplyr)
library(tm)
library(Unicode)
library(RCurl)
library(caret)
library(ranger)


#####################Import Data############################


age.profiles <- fromJSON("age_profiles.json", flatten = TRUE)
age.tweets <- fromJSON("age_tweets.json", flatten = TRUE)
friend.profiles <- fromJSON("age_profiles.json", flatten = TRUE)
mention.profiles <- fromJSON("age_tweets.json", flatten = TRUE)
ages.train <- read.csv("ages_train.csv")
ages.test <- read.csv("ages_test.csv")
friends <- read.csv("friends.csv")
mentions <- read.csv("mentions.csv")  


######################Histograms############################


ggplot(age.profiles["friends_count"], aes(friends_count)) +
  geom_histogram()+
  scale_x_log10()

ggplot(age.profiles["followers_count"], aes(followers_count)) +
  geom_histogram()+
  scale_x_log10()

ggplot(age.profiles["favourites_count"], aes(favourites_count)) +
  geom_histogram()+
  scale_x_log10()

ggplot(age.profiles["statuses_count"], aes(statuses_count)) +
  geom_histogram()+
  scale_x_log10()


#################Friend-Follower Relationship################


follower.friend.cor <- cor(as.vector(age.profiles["friends_count"]), as.vector(age.profiles["followers_count"]), method="pearson")

ggplot(age.profiles[c("friends_count", "followers_count")], aes(x = friends_count, y = followers_count)) +
  geom_point() +
  scale_x_log10()+scale_y_log10()+ geom_smooth(method = "lm") 


#################Favourites-Statuses Relationship############


fav.status.cor <- cor(as.vector(age.profiles["favourites_count"]), as.vector(age.profiles["statuses_count"]), method="pearson")

ggplot(age.profiles[c("favourites_count","statuses_count")], aes(x = favourites_count, y = statuses_count)) +
  geom_point() +
  scale_x_log10() + scale_y_log10() + geom_smooth(method = "lm")


#################Major OS by Time Zone#######################


phone.os <- age.profiles[c("time_zone", "status.source")]

os.by.timezone <- age.profiles%>%select(time_zone, status.source)%>%
  mutate(ios = str_detect(status.source, "ipad|iphone|iPad|iPhone"), android = str_detect(status.source, "android|Android"))%>%
  group_by(time_zone)%>%summarise(total.users = n(), ios.users=sum(ios,na.rm = T), android.users = sum(android,na.rm = T))%>%
  mutate(prop.ios = ios.users/total.users, prop.android = android.users/total.users)
  

##################Top Mentions###############################


top.mentions <- unique(mentions)%>%group_by(MentionedHandle)%>%
  summarise(no.of.mentions = n())%>%
  filter(no.of.mentions > 1)%>%
  arrange(desc(no.of.mentions))


######################Emoji Stats############################


em.dict <- read.csv2(text = getURL("https://raw.githubusercontent.com/today-is-a-good-day/emojis/master/emojis.csv"), header = T, stringsAsFactors = F)%>% 
  select(EN, ftu8, unicode) %>% 
  rename(description = EN, r.encoding = ftu8) # Import Emoji Dictionary (Courtesy: Jessica Peterka-Bonetta)

emoji.stats <- age.profiles[c("id_str", "status.text")]

emoji.stats$id_str <- as.numeric(emoji.stats$id_str)

emoji.stats <- emoji.stats%>%inner_join(ages.train, by = c("id_str" = "ID"))  # Note: 23 IDs from ages.train are not present in age.profiles- rest of the analysis ignores these IDs

emoji.stats$status.text <- iconv(emoji.stats[["status.text"]], from = "latin1", to = "ascii", sub = "byte") #convert status.text to workable format

emoji.matrix <- sapply(em.dict$r.encoding, str_count, string = emoji.stats$status.text)

colnames(emoji.matrix) <- em.dict$description

which.max(colSums(emoji.matrix, na.rm = T))  # Find Max Used Emoji (Red heart)

colSums(emoji.matrix, na.rm = T)[1231]  # Number of times red heart emoji was detected using emDict in the sample data tweets 

emoji.stats  <-  cbind(emoji.stats, rowSums(emoji.matrix, na.rm = T))

colnames(emoji.stats)[4] <- "total.emojis.from.dict"

emoji.stats <- emoji.stats%>%
  mutate(emoji.in.tweet = str_detect(status.text, "<.*>"))  # Detect emojis in tweet - caution -based on the observation that emojis are enclosed inside <> 

emoji.stats$age.group <- cut(emoji.stats$Age, breaks = seq(10, 120, by = 10))


##################Age group distribution#####################


barplot(table(cut(emoji.stats$Age, breaks = seq(10, 120, by = 10))), main = "Age Grougs")  # plot Agegroups


##################Emojis by Age-group########################


ggplot(emoji.stats, aes(age.group, as.integer(emoji.in.tweet)))  +
  stat_summary(fun.y = sum, geom = "bar") +ylab("Total Emoji Use Per Tweet")

ggplot(emoji.stats, aes(age.group, total.emojis.from.dict))  +
  stat_summary(fun.y = sum, geom = "bar") 


###################Prediction################################


# I am building an initial prediction model to just get things started
# Featues Used: status.text, emoji.in.tweet, total.emojis.from.dict
# Response: Age

training.set <- emoji.stats[complete.cases(emoji.stats), ]  # Since all these features are derived from status.text, I am only excluding all the (24) IDs  from the training set for just this model.
training.set$age.group = NULL
training.set$id_str = NULL
# Extract Text Features from status.text

CleanCorpus <- function(corpus){
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, c(stopwords("en")))
  return(corpus)
}

vector.corpus <- VCorpus(VectorSource(training.set[["status.text"]]))


vector.corpus <- CleanCorpus(vector.corpus)


status.dtm <- DocumentTermMatrix(vector.corpus)


status.dtm <- removeSparseTerms(status.dtm, sparse = 0.975)


status.df <- as.data.frame(as.matrix(status.dtm ))


training.set$status.text <- NULL

training.set <- cbind(training.set,status.df) 

# Build a prediction model (Random Forest)

model <- train(
  Age ~ .,
  tuneGrid = data.frame(mtry = c(2, 3, 7, 11, 14)),
  data = training.set, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Plot model

plot(model)




