from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math

domainlist=[]
outputlist=[]

def cal_num(text):
    count=0
    for char in text:
        if '0' <= char <= '9':
            count+=1
    return count

def cal_letter(text):
    count = 0
    for char in text:
        if'a'<= char <='z' or 'A'<= char <= 'Z':
            count +=1
    return count




class Domain:
    def __init__(self,_name,_label,_numOfletter,_numOfnum): 
            self.name=_name
            self.label=_label
            self.numOfletter=_numOfletter
            self.numOfnum=_numOfnum

    def returnData(self):
            return[self.numOfletter,self.numOfnum]

    def returnLabel(self):
            if self.label=="dga":
                    return 0
            else:
                    return 1
                
def labelToString(label):
    if label==0 :
        return "dga"
    if label==1 :
        return "notdga"

def initData(filename):
        with open(filename) as f:
                for line in f:
                        line=line.strip()
                        if line.startswith("#") or line =="":
                                continue
                        tokens = line.split(",")
                        name=tokens[0]
                        label=tokens[1]
                        numOfletter=int(cal_letter(name))
                        numOfnum=int(cal_num(name))
                        domainlist.append(Domain(name,label,numOfletter,numOfnum))
                                          
def outputData(filename,testFileName,clf):
        with open(testFileName) as f:
                for line in f:
                        line=line.strip()
                        if line.startswith("#") or line =="":
                                continue
                        tokens = line.split(",")
                        name=tokens[0]
                        outputlist.append(name)
        with open(filename,'w') as p:
                for item in outputlist:
                    p.write(item+","+labelToString(clf.predict([[cal_letter(item),cal_num(item)]]))+"\n")

def main():
	initData("train.txt")
	featureMatrix=[]
	labelList=[]
	for item in domainlist:
		featureMatrix.append(item.returnData())
		labelList.append(item.returnLabel())



	clf = RandomForestClassifier(random_state=0)
	clf.fit(featureMatrix,labelList)

	outputData("result.txt","test.txt",clf)

if __name__ == '__main__':
	main()

    
                    
                
                         
           
