"""
this is a python file you can simply copy paste it in your jupyter notebook and run it
Don't forget to upload the csv file that contains all the data needed
I recomand that you put it in the same directory with this python file
and don't modify its name(iris.csv)
(otherwise you have to change it in the open function)
"""
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

LData=[]
with open('iris.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        #print(row)
        LData.append(row)
        

Data=np.array(LData)
#print(Data.shape)
#print("{}".format( Data))

#ens d'apprentissage
LA = Data[0:100] #print(LA.shape)

#ens de validation
LV = Data[100:125] #print(LV.shape)

#ens de test
LT=Data[125:150] #print(LT.shape)

#extraction de x(features) et y(labels)
x_train=LA[:,0:4]
y_train=LA[:,4:5]
x_test=LV[:,0:4] #pour moi ça doit etre LT mais selon l'énoncé ce sont les données de validation qu'on va utiliser
y_test=LV[:,4:5]

x_train = x_train.astype(np.float64)
x_test = x_test.astype(np.float64)
#y_train = y_train.reshape((1,100)) why it doesn't work ?
y_train=np.ravel(y_train)
y_test=np.ravel(y_test)


#KNN algorithm for validation set
score_max=0
neighbor_max=0
for i in range(1,31):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(x_train, y_train) #training
    y_pred = clf.predict(x_test) #testing
    score = accuracy_score(y_test, y_pred)#calculate accuracy
    if score>score_max:
        score_max = score
        neighbor_max=i
print("Sur l'ensemble de validation : ")
print ("le score max est : {} pour la valeur de voisin(s) : {}".format(score_max , neighbor_max))

#KNN algorithm for test set
x_test=LT[:,0:4] 
y_test=LT[:,4:5]

x_test = x_test.astype(np.float64)
y_test=np.ravel(y_test)

score_max=0
neighbor_max=0
for i in range(1,31):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(x_train, y_train) #training
    y_pred = clf.predict(x_test) #testing
    score = accuracy_score(y_test, y_pred)#calculate accuracy
    if score>score_max:
        score_max = score
        neighbor_max=i
print("Sur l'ensemble de test : ")
print ("le score max est : {} pour la valeur de voisin(s) : {}".format(score_max , neighbor_max))
