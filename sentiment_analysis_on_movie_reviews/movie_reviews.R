library(readr)
library(data.table)
library(plyr)
library(tm)
library(SnowballC)
library(randomForest)
library(nnet)

dat = read_tsv("C:/Users/Yining Cai/Downloads/train.tsv/train.tsv")
dat = data.table(dat)
full_sentence = ddply(dat[, 1:2], .(SentenceId), summarize, PhraseId = min(PhraseId))
dat1 = merge(full_sentence, dat, by = c("SentenceId", "PhraseId"), all = FALSE)

vsource = VectorSource(dat1$Phrase)
x = VCorpus(vsource)

x = tm_map(x, content_transformer(tolower))
x = tm_map(x, stripWhitespace)
x = tm_map(x, removePunctuation)
x = tm_map(x, removeNumbers)
x = tm_map(x, removeWords, stopwords("en"))
x = tm_map(x, stemDocument, language = "english")

rm(dat, vsource, full_sentence)


tdm = TermDocumentMatrix(x)
tdm = t(as.matrix(tdm))
tdm = tdm[, colSums(tdm) >= 20]


#train = as.data.table(tdm)
#train$y = as.factor(dat1$Sentiment)
#sr1 = multinom(y~. , train)


train_x = as.data.table(tdm)
train_y = as.factor(dat1$Sentiment)
rf1 = randomForest(x = train_x, y = train_y, ntree = 100)
table(train_y, rf1$predicted)



