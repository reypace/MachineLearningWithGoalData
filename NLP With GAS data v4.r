#first we need to make our bag of words. We will start with 2-grams (every 2 word combination)
#then we'll process those 2-grams with chi-square, and keep the top 100. I.e., we'll look at each X^2 seperately with the outcome, and keep those with highest relationships
#library(qdap) #'text mining' package
library(tm) #'text mining' package
library(ngram) # used to find our n-grams and make DTMs
library(text2vec) #this is anothe 'text mining' package used in Kwartler (2017)
library(stringr) #useful library for text cleaning, although somewhat redundant with tm
library(e1071) #library for support vector machines (SVMs)


dat0=read.csv('C:/Users/j147p700/OneDrive - The University of Kansas/NLP Resources/NLP With Sheidas data/MD SDLMI Goals Codebook.csv')
dat=dat0[,colnames(dat0)%in%c('Goal.Description','Specific')]
colnames(dat)=c('Goal.Description','Category..Academic.or.Other.')
dat2=dat[,colnames(dat)%in%c('Goal.Description','Category..Academic.or.Other.')] #this is the data we run our analysis on

#our first step is to do a little clean up of the data
  #these functions are self explanatory. Some are base r, others are in the 'tm' package.
  dat2$Goal.Description=tolower(dat2$Goal.Description)
  dat2$Goal.Description=removePunctuation(dat2$Goal.Description)
  dat2$Goal.Description=removeNumbers(dat2$Goal.Description)
  dat2$Goal.Description=stripWhitespace(dat2$Goal.Description)

#removeWords is a shortcut to cut out parts of strings. We'll do this for a list of 'stop words'. It would suffice to do a gsub for each one with ''. But, we'll use the built-in to save time
  dat2$Goal.Description=removeWords(dat2$Goal.Description,stopwords('en'))
  dat2$Goal.Description=stripWhitespace(dat2$Goal.Description)
  
#lets make a holdout set for later cross val
  dat2=dat2[!is.na(dat2$Category..Academic.or.Other.),]
  set.seed(999)
  samples=sample(c(1:nrow(dat2)),200)
  holdout=dat2[samples,]
  datTest=dat2[-samples,]

#now we make our bi-grams and DTM:
  #First step is to extract the vocabulary
  datTest$Goal.Description=str_trim(datTest$Goal.Description)
  tokens <- strsplit(datTest$Goal.Description, split = " ", fixed = T)
  vocab <- create_vocabulary(itoken(tokens),ngram = c(1, 2)) #this extracts all of our 1 and 2 grams
  
  
  #now we can make our DTM matrix with our 1-grams and 2-grams like so
  vecizle=vocab_vectorizer(vocab)
  it = itoken(datTest$Goal.Description, preprocess_function = tolower, tokenizer = word_tokenizer)
  dtm = create_dtm(it, vecizle)
  #note, colnames of DTM are our 1 and 2-grams. 
    #now, we could do this ourselves... WE could loop over our datamatrix and grep for terms. It would be slow, but WE COULD if we really wanted to
  #lets make it a matrix
  dtm2=as.matrix(dtm)
  
#now, let us prune to 100 best features using x^2
  dtm3=cbind(dtm2,datTest$Category..Academic.or.Other.)
  x2=vector(length=ncol(dtm2))
  for(i in 1:ncol(dtm2))
  {
    x2[i]=chisq.test(dtm3[,i],dtm3[,ncol(dtm3)])
  }
  
  #now, what are 100 best features
  orderOfElements=order(unlist(x2),decreasing=TRUE)
  best100=orderOfElements[1:100]
  colnames(dtm3)[best100]
  dtm4=dtm3[,c(best100,ncol(dtm3))]
  colnames(dtm4)[ncol(dtm4)]<-'class'
  
#okay now that feature selection is done, we can do the actual SVM work
  dtm4=as.data.frame(dtm4)
  dtm4$class=as.factor(dtm4$class)
  #first we need to find our best 'cost' parameter, we'll use 10-fold x-val (default)
  tune.out <- tune(svm, class~ . , data = dtm4 , kernel = "radial", ranges = list(cost = c(0.001 , 0.01, 0.1, 1, 5, 10, 100,150,200)),scale=FALSE)
  costOnly=unlist(tune.out$best.parameters)
  #now we run the svm
  svmfit <- svm(class~ . , data = dtm4 , kernel = "radial", cost = costOnly, scale = FALSE,probability=TRUE)


#now we calculate the degree of accuracy. We do this crudely here - simply look at % agreement with actual classification
  #two ways to do this here: get the probabilities and round them, or just get predicted class
  pred <- predict(svmfit, decision.values = TRUE, probability = TRUE,newdata=dtm4)
  probs=attr(pred, "probabilities") #these are the probabilities
  predicted=as.numeric(as.character(predict(svmfit))) #these are the predicted classes.
  actual=as.numeric(as.character(dtm4[,'class']))

  #compare with the actual
  compare=cbind.data.frame(predicted,actual)
  1-length(  which(rowSums(compare)==1))/nrow(dtm4) #% correct!
  
  ##so we get 89% accuracy for our training data. what about for our holdout sample?

#Holdout sample 
  
  #what I need to do somehow is take the dtm items from the training data, and populate it with the holdout data
  colsToKeep=colnames(dtm3)[best100] #these were the 100 1 and 2-grams we found in the training data.
  
  #we repeat the earlier code to extract the 1 & 2 grams in the holdout data
  holdout$Goal.Description=str_trim(holdout$Goal.Description)
  tokens <- strsplit(holdout$Goal.Description, split = " ", fixed = T)
  vocab <- create_vocabulary(itoken(tokens),ngram = c(1, 2))
  vecizle=vocab_vectorizer(vocab)
  it = itoken(holdout$Goal.Description, preprocess_function = tolower, tokenizer = word_tokenizer)
  dtm = create_dtm(it, vecizle)
  dtm2=as.matrix(dtm)
  
  #now we restrict the dtm to have only those terms which were found in the training data (if a term isn't found, its just gone and we have 100-n not found terms to work with)
  length(which(colnames(dtm2)%in%colsToKeep)) #we get to keep 80 of them. not bad really
  dtm3=dtm2[,colnames(dtm2)%in%colsToKeep]
  dtm4=cbind.data.frame(dtm3,holdout$Category..Academic.or.Other.) #get the class var back in there
  colnames(dtm4)[ncol(dtm4)]<-'class'

  
#actual SVM work
 dtm4=as.data.frame(dtm4)
  dtm4$class=as.factor(dtm4$class)
  #first we need to find our best 'cost' parameter, we'll use 10-fold x-val (default)
  tune.out <- tune(svm, class~ . , data = dtm4 , kernel = "radial", ranges = list(cost = c(0.001 , 0.01, 0.1, 1, 5, 10, 100,150,200)),scale=FALSE)
  costOnly=unlist(tune.out$best.parameters)
  #now we run the svm
  svmfit <- svm(class~ . , data = dtm4 , kernel = "radial", cost = costOnly, scale = FALSE,probability=TRUE)
   predicted=as.numeric(as.character(predict(svmfit))) #these are the predicted classes.
  actual=as.numeric(as.character(dtm4[,'class']))

  #compare with the actual
  compare=cbind.data.frame(predicted,actual)
  1-length(  which(rowSums(compare)==1))/nrow(dtm4) #% correct!

  

#out of curiosity, what happens with logistic?
temp=glm(class~ .,family='binomial',data=dtm4)
comp=cbind.data.frame(round(temp$fitted.values),as.numeric(as.character(dtm4[,'class'])))
1-length(  which(rowSums(comp)==1))/nrow(dtm4) #% correct!
  #performance is pretty similar - so we've not yet really tapped into the power of the SVM. Still pretty cool.

  



#the so what: if we can use this is a screening tool we might get better quality data
  #lets look at growth over time between t1 and t2 (t-test with time as predictor) on a) un-screened data, b) screened data

gas=read.csv('S:/SDI_Online_Survey/StatsTeam/MD Project/MD GAS/Year 1/Y1_GAS_WIDE_TH3.csv')


  
  
