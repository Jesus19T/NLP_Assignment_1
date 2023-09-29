import os
import string
import re
import math 


def preprocess (filePath):
    fileWords = []
    parseWords = []
    parsePhrase = []
    cleanCorpus = []

    with open(filePath, 'r', errors = 'ignore') as f:
       lines = f.readlines() #parse words
       
    for x in lines:    
        y = x.split('\n')       #remove return
        lineLower = y[0].lower()        #lowercase
        fileWords.append(("<start>") +" " + lineLower + " " + ("<stop>")) #add start and stop tokens
       
    for x in fileWords:   
        temp = re.sub("[%$&@=`()!\[\]#.,/\"'-:;|?*“”’‘_]","",x)
        temp2 = re.sub("\s+"," ",temp)
        cleanCorpus.append(temp2) #cleaned corpus
        splitVal = temp2.split(' ') 
        
       
        for y in range (0, (len(splitVal))):  
            parseWords.append(splitVal[y]) #combine to large corpus for each word
            if(y != 0): 
                parsePhrase.append(splitVal[y-1]+" "+ splitVal[y]) #create bigram phrases
                    
    return parseWords, parsePhrase, cleanCorpus

def createDictionary(data): 
    dictionary = {}
    for word in data: 
        if(dictionary.get(word)): 
             num = dictionary.get(word)
             dictionary.update({word: (num + 1)}) 
        else:
             dictionary.update({word: 1})
        
    return dictionary

def unigramTraining(dataCount, data): 
    dictionary = {}
    logDict = {}
    for key in dataCount:
        num = dataCount.get(key) #get numerator
        dictionary.update({key: num/len(data)}) #probability regular
        logDict.update({key: -math.log((num/len(data)))}) #probability with log
    
    return dictionary, logDict

def bigramTraining(bigramCount, unigramCount): 
    dictionary = {}
    logDict = {}
    for key in bigramCount:
        prev = key.split(' ') #get denominator key
        num = bigramCount.get(key)
        den = unigramCount.get(prev[0]) #get denominator value
        dictionary.update({key: num/den}) #probability regular
        logDict.update({key: -math.log((num/den))}) #probability with log
    
    return dictionary, logDict

def unigramPerplexityModel (unigramLog, reviews): 
    probabilities = []
    for x in reviews: 
        total = 1
        line = x.split(' ')
        for word in line: 
            val = unigramLog.get(word) #key values 
            #unknown values? 
            print(val)
            total = total+val
        probabilities.append(total)
    
    return probabilities

def main():
    #read files
   # unigramDict = {}
    path = os.getcwd()
    
    newFile = path+"/A1_DATASET/train.txt"    #Uncommment for MAC
    valFile = path+"/A1_DATASET/val.txt"     #validation set
    
    unigramSet, bigramSet, reviews = preprocess(newFile)
 
    #test 
    print(unigramSet[0:30])
    print(bigramSet[0:30])
    
    #create dictionaries
    unigramCount = createDictionary(unigramSet)
    bigramCount = createDictionary(bigramSet)
    
    #training set
    #Section3 - calculate probabilities
    unigramTrainProb, unigramTrainLog = unigramTraining(unigramCount, unigramSet) 
    bigramTrainProb, bigramTrainLog = bigramTraining(bigramCount, unigramCount)
    print("Training Complete")
    
    
    #Section 4 - Smoothing
    #Laplace
    #Add-k
    
    #Section 5 - calculate Perplexity
    valUnigram, valBigram, valReviews = preprocess(valFile)
    reviewUnigramProb = unigramPerplexityModel(unigramTrainLog, valReviews)
    
    #create bigram model
    print("Dictionary")

if __name__ == "__main__":
    main()
    print("Complete")
