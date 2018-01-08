#################Import Necessary Packages###############
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


age.profiles=fromJSON("age_profiles.json", flatten = TRUE)
age.tweets=fromJSON("age_tweets.json",flatten = TRUE)
friend.profiles=fromJSON("age_profiles.json", flatten = TRUE)
mention.profiles=fromJSON("age_tweets.json",flatten = TRUE)
ages.train=read.csv("ages_train.csv")
ages.test=read.csv("ages_test.csv")
friends=read.csv("friends.csv")
mentions=read.csv("mentions.csv")  


######################Histograms##########################
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


#names(age.profiles)
#names(age.tweets)

#################Friend-Follower Relationship###############

follower.friend.cor=cor(as.vector(age.profiles["friends_count"]),as.vector(age.profiles["followers_count"]),method="pearson")

ggplot(age.profiles[c("friends_count","followers_count")], aes(x = friends_count, y = followers_count)) +
  geom_point() +
  scale_x_log10()+scale_y_log10()+ geom_smooth(method="lm") 


#################Favourites-Statuses Relationship############

fav.status.cor=cor(as.vector(age.profiles["favourites_count"]),as.vector(age.profiles["statuses_count"]),method="pearson")

ggplot(age.profiles[c("favourites_count","statuses_count")], aes(x = favourites_count, y = statuses_count)) +
  geom_point() +
  scale_x_log10()+scale_y_log10()+ geom_smooth(method="lm") 


#################Major OS by Time Zone#######################
phone.OS=age.profiles[c("time_zone","status.source")]

OS.by.Timezone=age.profiles%>%select(time_zone,status.source)%>%
  mutate(IOS=str_detect(status.source,"ipad|iphone|iPad|iPhone"),Android=str_detect(status.source,"android|Android"))%>%
  group_by(time_zone)%>%summarise(Total.users=n(),IOS.users=sum(IOS,na.rm=T),Android.users=sum(Android,na.rm=T))%>%
  mutate(prop.IOS=IOS.users/Total.users,prop.Android=Android.users/Total.users)
  

##################Top Mentions###############################

Top.mentions=unique(mentions)%>%group_by(MentionedHandle)%>%
  summarise(No.of.Mentions=n())%>%
  filter(No.of.Mentions>1)%>%
  arrange(desc(No.of.Mentions))



######################Emoji Stats############################



  
emDict=read.csv2(text=getURL("https://raw.githubusercontent.com/today-is-a-good-day/emojis/master/emojis.csv"), header=T,stringsAsFactors = F)%>% 
  select(EN, ftu8, unicode) %>% 
  rename(description = EN, r.encoding = ftu8) ##########Import Emoji Dictionary- (Courtesy: Jessica Peterka-Bonetta)##
 



Emoji.stats=age.profiles[c("id_str","status.text")]
Emoji.stats$id_str=as.numeric(Emoji.stats$id_str)

Emoji.stats=Emoji.stats%>%inner_join(ages.train,  by = c("id_str"="ID")) #Note: 23 IDs from ages.train are not present in age.profiles- rest of the analysis ignores these IDs

Emoji.stats$status.text<- iconv(Emoji.stats[["status.text"]], from = "latin1", to = "ascii", sub = "byte") #convert status.text to workable format

Emoji.matrix=sapply(emDict$r.encoding, str_count, string=Emoji.stats$status.text)

colnames(Emoji.matrix)=emDict$description

which.max(colSums(Emoji.matrix,na.rm = T)) # Find Max Used Emoji (Red Heart)

colSums(Emoji.matrix,na.rm = T)[1231]      # No of times Red Heart emoji was detected using emDict in the sample data tweets 

Emoji.stats=cbind(Emoji.stats,rowSums(Emoji.matrix,na.rm = T))

colnames(Emoji.stats)[4] <- "Total.Emojis.From.Dict"

Emoji.stats=Emoji.stats%>%
  mutate(Emoji.in.Tweet=str_detect(status.text,"<.*>")) #Emojis.in.tweet - caution -based on the observation that emojis are enclosed inside <> 


Emoji.stats$Age.group=cut(Emoji.stats$Age,breaks = seq(10, 120, by = 10))

##################Age group distribution#####################
barplot(table(cut(Emoji.stats$Age,breaks = seq(10, 120, by = 10))), main = "Age Grougs") #plot Agegroups


##################Emojis by Age-group#######################
ggplot(Emoji.stats, aes(Age.group,as.integer(Emoji.in.Tweet)))  +
  stat_summary(fun.y = sum, geom = "bar") +ylab("Total Emoji Use Per Tweet")

ggplot(Emoji.stats, aes(Age.group,Total.Emojis.From.Dict))  +
  stat_summary(fun.y = sum, geom = "bar") 


###################Prediction##############################

##I am building an initial (basic)model to just get things started
##Featues Used: status.text
##Emoji.in.tweet
##Total Emojis from Dict
##AgeGroup (Classification)

TrainingSet=Emoji.stats[complete.cases(Emoji.stats),]##Since all these features are derived from status.text, I am only excluding all the (24) IDs  from the training set for just this model.

TrainingSet$Age.group=NULL
TrainingSet$id_str=NULL

##################Extract Text Features from status.text###

clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, c(stopwords("en")))
  return(corpus)
}

Vector_corpus=VCorpus(VectorSource(TrainingSet[["status.text"]]))


Vector_corpus = clean_corpus(Vector_corpus)


status_dtm = DocumentTermMatrix(Vector_corpus)


status_dtm = removeSparseTerms(status_dtm, sparse = 0.975)


status_df <- as.data.frame(as.matrix(status_dtm ))


TrainingSet$status.text=NULL

TrainingSet=cbind(TrainingSet,status_df) 

#################Build a predictive Model (Random Forest)###################

# Fit random forest: model
model <- train(
  Age ~ .,
  tuneGrid = data.frame(mtry = c(2, 3, 7, 11, 14)),
  data = TrainingSet, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)


# Plot model
plot(model)



ggplot(Emoji.stats, aes(Emoji.stats$Age)) +
  geom_histogram(binwidth = 1)+
  scale_x_log10()

