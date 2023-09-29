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

def unigramTraining(dataCount, dataLength): 
    dictionary = {}
    logDict = {}
    for key in dataCount:
        num = dataCount.get(key) #get numerator
        dictionary.update({key: num/dataLength}) #probability regular
        logDict.update({key: -math.log((num/dataLength))}) #probability with log
    
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

def PerplexityModel (logDict, review): 
   # probabilities = []
   # probabilities = 0
    total = 0
    for word in review: 
        if(logDict.get(word)):
            val = logDict.get(word)#key values 
            
        else:
            val = logDict.get("<UNK>") #get unknown value if can't find
       
        total = total + val #sum of logs
    
    finalProb = math.exp(total/len(review)) #divide by N and raise to e

   # probabilities.append(finalProb) #probability for each review
    
    return finalProb

def createUnknownList(unigramCount): 

        
    dictionary = {}
    dictionary.update({"<UNK>": 0})
    for word in unigramCount: 
        if(unigramCount.get(word) < 2): 
            num = dictionary.get("<UNK>")
            dictionary.update({"<UNK>": (num + 1) })   #low frequency words = UNK token and add count
        else:
            num = unigramCount.get(word)
            dictionary.update({word: num}) 
                
    return dictionary


def laPlaceUnigram(dictionary, dataLength): 
    
    newDict = {}
    logDict = {}
    V = len(dictionary)
    for key in dictionary: 
        num = dictionary.get(key) #get numerator
        newDict.update({key: ((num+1)/(dataLength+V))}) #probability regular
        logDict.update({key: -math.log(((num+1)/(dataLength+V)))}) #probability regular
    return newDict, logDict   

def laPlaceBigram(bigramCount, unigramCount): 
    dictionary = {}
    logDict = {}
    V = len(dictionary)
    for key in bigramCount:
        prev = key.split(' ') #get denominator key
        num = bigramCount.get(key)
        den = unigramCount.get(prev[0]) #get denominator value
        dictionary.update({key: ((num+1)/(den+V))}) #probability regular
        logDict.update({key: -math.log((num+1)/(den+V))}) #probability with log
    
    return dictionary, logDict

def main():
    #read files
   # unigramDict = {}
    path = os.getcwd()
    
    newFile = path+"/A1_DATASET/train.txt"    #Uncommment for MAC
    valFile = path+"/A1_DATASET/val.txt"     #validation set
    
    unigramSet, bigramSet, reviews = preprocess(newFile)
    
    unigramLength = len(unigramSet)
    bigramLength = len(bigramSet)
     
    #create dictionaries
    unigramCount = createDictionary(unigramSet)
    bigramCount = createDictionary(bigramSet)
    
    #training set
    #Section3 - calculate probabilities
    unigramTrainProb, unigramTrainLog = unigramTraining(unigramCount, unigramLength) 
    bigramTrainProb, bigramTrainLog = bigramTraining(bigramCount, unigramCount)
    print("Training Complete")
    
    
    #Section 4 - Unknown
    unknownUnigramCount = createUnknownList(unigramCount)
    unknownBigramCount = createUnknownList(bigramCount)
    
    unknownUnigramLength = len(unknownUnigramCount)
    
    unigramTrainProb2, unigramTrainLog2 = unigramTraining(unknownUnigramCount, unknownUnigramLength) 
    bigramTrainProb2, bigramTrainLog2 = bigramTraining(unknownBigramCount, unknownUnigramCount)
    print('Unknown Training Complete')
    
    #Section 4 - Smoothing  
    #Laplace
    unigramlaPlaceVal, unigramlaPlaceLog = laPlaceUnigram(unknownUnigramCount, unknownUnigramLength)
    bigramLaPlaceVal, bigramlaPlaceLog = laPlaceBigram(unknownBigramCount, unknownUnigramCount)
    
   
    
   # sorted_footballers_by_goals = sorted(footballers_goals.items(), key=lambda x:x[1])
    #Interpolation
    
    
    #Section 5 - calculate Perplexity
    #preprocess validation set
    valUnigram, valBigram, valReviews = preprocess(valFile)
    
    
    reviewUnigramProb = PerplexityModel(unigramlaPlaceLog, valUnigram) #laplace unigram
    #print("Perplexity using Unigram Model: %d" reviewUnigramProb )
    
    
    reviewBigramProb = PerplexityModel(bigramlaPlaceLog, valBigram) #laplace unigram
   # print("Perplexity using Bigram Model:" + reviewBigramProb )
    
    

    print("Complete")

if __name__ == "__main__":
    main()

