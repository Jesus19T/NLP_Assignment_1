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

def initDictionary(dictionary, data):
    for word in data: 
        dictionary.update({word: 0}) 
            
    return dictionary

def createDictionary(data, dictionary): 
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
        try:
            logDict.update({key: -math.log((num/dataLength))}) #probability with log
        except: 
            logDict.update({key: 0}) #num = 0
    
    return dictionary, logDict

def bigramTraining(bigramCount, unigramCount): 
    dictionary = {}
    logDict = {}
    for key in bigramCount:
        prev = key.split(' ') #get denominator key
        num = bigramCount.get(key)
        den = unigramCount.get(prev[0]) #get denominator value
        
        try:
            dictionary.update({key: num/den}) #probability regular
            logDict.update({key: -math.log((num/den))}) #probability with log
        except: 
            dictionary.update({key: 0}) #probability regular
            logDict.update({key: 0}) #num = 0
    
    return dictionary, logDict

def PerplexityModel (logDict, review , bigramT): 
   # probabilities = []
   # probabilities = 0
    total = 0
    for word in review: 
        if(logDict.get(word)):
            val = logDict.get(word)#key values 
            
        else:
            if(bigramT):
                x = word.split(" ")
                firstVal= x[0]+ " " + "<UNK>"
                secondVal = "<UNK>" + " " + x[1]
                none = "<UNK> <UNK>"
                
                if(logDict.get(firstVal)):
                   val = logDict.get(firstVal)
                elif(logDict.get(secondVal)):
                    val = logDict.get(secondVal)
                else:
                    val = logDict.get(none) # <UNK> <UNK>
                
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

def createUnknownBigramList(bigramCount, unigramCount, unknownUnigramCount): 
    unknownWords = []
    unigramCountList = sorted(unigramCount.items(), key=lambda x:x[1])
    for key,count in unigramCountList:
        if(count < 2):
            unknownWords.append(key)
        
    dictionary = {}
    newBigramCount = []
    dictionary.update({"<UNK> <UNK>": 0})
    for word in bigramCount: 
        x = word.split(' ')
        if((x[0] in unknownWords) and (x[1] in unknownWords)): #both word is unknown
            newKey = "<UNK> <UNK>"
            num = dictionary.get(newKey)
            dictionary.update({newKey: (num+1)})
        elif(x[0] in unknownWords): 
            newKey = "<UNK>"+ " " + x[1]                    #first word is unknown
            num = bigramCount.get(word)
            dictionary.update({newKey: (num) })   #<UNK> token and add count
        elif(x[1] in unknownWords): 
            newKey = x[0]+ " " + "<UNK>"                   #first word is unknown
            num = bigramCount.get(word)
            dictionary.update({newKey: (num) })   #<UNK> token and add count
        else:
            newKey = word
            num = bigramCount.get(word)
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
    V = len(bigramCount)
    for key in bigramCount:
        prev = key.split(' ') #get denominator key
        num = bigramCount.get(key)
        den = unigramCount.get(prev[0]) #get denominator value
        dictionary.update({key: ((num+1)/(den+V))}) #probability regular
        logDict.update({key: -math.log((num+1)/(den+V))}) #probability with log
    
    return dictionary, logDict

def addKUnigram(dictionary, dataLength, review, kVal): 
    newDict = {}
    logDict = {}
    kvalDict = {}
 
    for k in kVal: 
        newDict, logDict = extractUnigramDictionary(dictionary, k, dataLength)
        num = PerplexityModel (logDict, review, False)
        kvalDict.update({k: num})
    
    optK = min(kvalDict, key=kvalDict.get)
       
    return optK, kvalDict   

def extractUnigramDictionary(dictionary, k, dataLength):
    newDict = {}
    logDict = {}
    V = len(dictionary)
    for key in dictionary: 
        num = dictionary.get(key) #get numerator
        newDict.update({key: (num+k)/(dataLength+(k*V))}) #probability regular
        logDict.update({key: -math.log(((num+k)/(dataLength+(k*V))))}) #probability regular
    return newDict, logDict 
        
def addKBigram(bigramDict, unigramDict, review, kVal): 
    newDict = {}
    logDict = {}
    kvalDict = {}

    for k in kVal: 
        newDict, logDict = extractBigramDictionary(bigramDict, unigramDict, k)
        num = PerplexityModel (logDict, review , True)
        kvalDict.update({k: num})
    
    optK = min(kvalDict, key=kvalDict.get)
       
    return optK, kvalDict   

def extractBigramDictionary(bigramDict, unigramDict, k):
    newDict = {}
    logDict = {}
    V = len(bigramDict)
    for key in bigramDict: 
        prev = key.split(' ') #get denominator key
        num = bigramDict.get(key)
        den = unigramDict.get(prev[0]) #get denominator value
        newDict.update({key: ((num+k)/(den+(k*V)))}) #probability regular
        logDict.update({key: -math.log((num+k)/(den+(k*V)))}) #probability with log
        
    return newDict, logDict     
    
def main():
    #read files
   # unigramDict = {}
    path = os.getcwd()
    unigramVocab = {}
    bigramVocab = {}
    
    newFile = path+"/A1_DATASET/train.txt"    #Uncommment for MAC
    valFile = path+"/A1_DATASET/val.txt"     #validation set
    
    unigramSet, bigramSet, reviews = preprocess(newFile)
    #preprocess validation set
    valUnigram, valBigram, valReviews = preprocess(valFile)
    
    unigramLength = len(unigramSet)
    unigramVocab = initDictionary(unigramVocab, unigramSet)
    unigramVocab = initDictionary(unigramVocab, valUnigram)
    
    #bigram vocab
    bigramVocab = initDictionary(bigramVocab, bigramSet)
    bigramVocab = initDictionary(bigramVocab, valBigram)
     
   
    
    #create dictionaries
    unigramCount = createDictionary(unigramSet, unigramVocab)
    bigramCount = createDictionary(bigramSet, bigramVocab)
    
    
    #test
    unigramCountList = sorted(unigramCount.items(), key=lambda x:x[1])
    BigramCountList = sorted(bigramCount.items(), key=lambda x:x[1])
    
    #training set
    #Section3 - calculate probabilities
    unigramTrainProb, unigramTrainLog = unigramTraining(unigramCount, unigramLength) 
    bigramTrainProb, bigramTrainLog = bigramTraining(bigramCount, unigramCount)
    print("Training Complete")
    
    
    #Section 4 - Unknown
    unknownUnigramCount = createUnknownList(unigramCount)
    unknownBigramCount = createUnknownBigramList(bigramCount, unigramCount, unknownUnigramCount)
    
    unknownUnigramLength = len(unknownUnigramCount)
    
    unigramTrainProb2, unigramTrainLog2 = unigramTraining(unknownUnigramCount, unknownUnigramLength) 
    bigramTrainProb2, bigramTrainLog2 = bigramTraining(unknownBigramCount, unknownUnigramCount)
    print('Unknown Words Training Complete')
    
    #Section 4 - Smoothing  
    #Laplace
    unigramlaPlaceVal, unigramlaPlaceLog = laPlaceUnigram(unknownUnigramCount, unknownUnigramLength)
    bigramLaPlaceVal, bigramlaPlaceLog = laPlaceBigram(unknownBigramCount, unknownUnigramCount)
    
    #Add-k
    kVal = [0.5, 0.05, 0.01, 0.001]
    optkUni, kValUniDict = addKUnigram(unknownUnigramCount, unknownUnigramLength, valUnigram, kVal)
    optkBi, kValBiDict = addKBigram(unknownBigramCount, unknownUnigramCount, valBigram, kVal)
    print("Best k for Unigram", optkUni)
    print("Best k for Bigram", optkBi)
    #Interpolation
    
    
    #Section 5 - calculate Perplexity
   
    
    
    reviewUnigramProb = PerplexityModel(unigramlaPlaceLog, valUnigram, False) #laplace unigram
    #print("Perplexity using Unigram Model: %d" reviewUnigramProb )
    
    
    reviewBigramProb = PerplexityModel(bigramlaPlaceLog, valBigram, True) #laplace unigram
   # print("Perplexity using Bigram Model:" + reviewBigramProb )
    
    

    print("Complete")

if __name__ == "__main__":
    main()

