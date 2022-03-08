import os
import re
import codecs
import numpy as np
import math

cPath = "dataset/"
Gfiles = os.listdir(cPath)
result = []
stopset = set()
listf = []
lista = []
totalwordcount = dict()
regEx = re.compile('\W')
listA = []

with codecs.open("stopwords.txt", mode='r', encoding='Latin1') as f2:
    stopwords = {line.strip('\n').lower() for line in f2}
for w in stopwords:
    stopset.add(w.strip('\r'))

for filename in Gfiles:
    bPath = cPath + filename
    files = os.listdir(bPath)
# open files in a nested dictionary structure

    for item in files:

       f1 = codecs.open(bPath+"\\"+item, mode='r', encoding='Latin1')
       listv = regEx.split(f1.read())
       listw = [re.sub(r'[^a-z]','',w).strip().lower() for w in listv]

       listw = [i for i in listw if i not in stopset]

       from nltk.stem.porter import *
       stemmer = PorterStemmer()
       singles = [stemmer.stem(plural) for plural in listw]
       result.append(singles)


'''construct a word dict'''
for i in range(0, len(result)):
  processedword = dict()
  for word in result[i]:
     if not word in processedword.keys():
      processedword[word] = result[i].count(word) / len(result[i])
     else: continue
  listf.append(processedword)

'''count the frequency with a dict'''
for i in range(0, len(result)):
   for word in result[i]:
      if word in totalwordcount.keys():
         totalwordcount[word] += 1
      else:
         totalwordcount[word] = 1


'''calculate a and store it in a dictionary lista'''
for i in range(0, len(result)):
  analyticalword = dict()
  for word in result[i]:
     if not word in analyticalword.keys():
        analyticalword[word] = listf[i][word] * math.log(len(result)/totalwordcount[word])
     else: continue
  lista.append(analyticalword)

max_k = 0
for i in range(0, len(result)):
  if max_k < len(result[i]):
    max_k = len(result[i])

listA = np.zeros((len(result),max_k))
for i in range(0, len(result)):
  Aword = []
  for word in result[i]:
   '''initialize and calculate
      the divisior for each words
   '''
   k = result[i].index(word)
   squaresum = 0
   for j in result[i]:
     squaresum += math.sqrt(math.pow(lista[i][j],2))
   listA[i][k] = lista[i][word]/squaresum
   print("working...",end=" ")
np.savez('train-20ng.npz',listA)
print("\ndone!")
