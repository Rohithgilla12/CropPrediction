import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("FinalFinal.csv")
#print(dataset.shape)
print(dataset.head())

#dataset = pd.get_dummies(dataset)
data2 = dataset[['msp','climate_conditions','rainfall','nitrogen','phosphorus','potassium','Season','soil_conditions']]
dataset2 = pd.get_dummies(data2)
X = dataset2.iloc[:,0:12].values

Y = dataset['Type']

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0)

clf = SVC(kernel='linear')
clf.fit(X_train,Y_train)

climate = int(input("Enter the average temperature in celcius"))
soiltype = int(input(('Enter the soil type: 0 for alluvial soil, 1 for black soil,2 for laterite soil,3 for marshy soil,4 for red soil') ))
rainfall = int(input("Enter the rainfall in mm in your region"))
potas_nutri = int(input("Enter the potassium content in kg/ha"))
phosphor_nutri = int(input("Enter the phosphorous content in kg/ha"))
nitrogen_nutri = int(input("Enter the nitrogen content in kg/ha"))
season = int(input("Enter the season, 0 for kharif, 1 for rabi,2 for yearlong season"))
if (season==0):
    if(soiltype==0):
                print(clf.predict([[climate,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,1,0,1,0,0,0,0]]))
    elif(soiltype==1):
                print(clf.predict([[climate,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,1,0,0,1,0,0,0]])) 
    elif(soiltype==2):
                print(clf.predict([[climate,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,1,0,0,0,1,0,0]]))
    elif(soiltype==3):
                print(clf.predict([[climate,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,1,0,0,0,0,1,0]]))
    elif(soiltype==4):
                print(clf.predict([[climate,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,1,0,0,0,0,0,1]]))
else:
    if(soiltype==0):
                print(clf.predict([[climate,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,0,1,1,0,0,0,0]]))
    elif(soiltype==1):
                print(clf.predict([[climate,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,0,1,0,1,0,0,0]]))
    elif(soiltype==2):
                print(clf.predict([[climate,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,0,1,0,0,1,0,0]]))
    elif(soiltype==3):
                print(clf.predict([[climate,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,0,1,0,0,0,1,0]]))
    elif(soiltype==4):
                print(clf.predict([[climate,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,0,1,0,0,0,0,1]]))


#Y_pred = svclassifier.predict(X_test)


#print(confusion_matrix(Y_test,Y_pred))  
#print(classification_report(Y_test,Y_pred))
