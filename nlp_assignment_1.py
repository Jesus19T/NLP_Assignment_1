import os


def preprocess (filePath):
    fileWords = []
    with open(filePath, 'r', errors = 'ignore') as f:
       lines = f.read().split(' ') #parse words 
    
    
    fileWords=[x.lower() for x in lines if x.isalpha()] #remove punctuation
                            
            
    return fileWords

def main():
    #read files
    path = os.getcwd()
    newFile = path+"\\train.txt"
    
    trainingData = preprocess(newFile)
    print(trainingData[0:30])
    
    
    

if __name__ == "__main__":
    main()print("Hello World")
