import os
import string
import re


def preprocess (filePath):
    fileWords = []
    parseWords = []
    parsePhrase = []
   # punc ="[\[\]# .,!/\"'-:;|?*“”’‘_]"
    with open(filePath, 'r', errors = 'ignore') as f:
       lines = f.readlines() #parse words
    
 
    for x in lines:    
        y = x.split('\n')       #remove return
        lineLower = y[0].lower()        #lowercase
        fileWords.append(("<start>") +" " + lineLower + " " + ("<stop>")) #add start and stop tokens
       
    for x in fileWords:   
          
        temp = re.sub("[()!\[\]#.,/\"'-:;|?*“”’‘_]","",x)
        temp2 = re.sub("\s+"," ",temp)
        splitVal = temp2.split(' ') 
       

        for y in range (0, (len(splitVal))):  
            parseWords.append(splitVal[y]) #combine to large corpus for each word
            if(y != 0): 
                parsePhrase.append(splitVal[y-1]+" "+ splitVal[y]) #create bigram phrases
            if(y==188): 
                print("x")
                    
    return parseWords, parsePhrase

def unigram(data): 
    dictionary = {}
    for words in data: 
        if(dictionary.get(words)): 
             num = dictionary.get(words)
             dictionary.update({words: num + 1}) 
        else:
             dictionary.update({words: 1})
        
    return dictionary

def main():
    #read files
    unigramDict = {}
    path = os.getcwd()
    
    newFile = path+"/A1_DATASET/train.txt"    #Uncommment for MAC
    
    unigramSet, bigramSet = preprocess(newFile)
    

    #test 
    print(unigramSet[0:30])
    print(bigramSet[0:30])
    
    unigramDict = unigram(unigramSet)
    print("Dictionary")
    
    
    

if __name__ == "__main__":
    main()
    print("Complete")
