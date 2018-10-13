import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from django.shortcuts import render
from django.http import JsonResponse
import json

dataset = pd.read_csv("datasets/FinalFinal.csv")
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
joblib.dump(clf, 'models/model.joblib')

def index(request):
    if request.method == "POST":
        print("Posting")
        return render(request,'index.html',{"context":"Blahblah"})
    return render(request,"index.html",{})

def get_resp(request,*args,**kwargs):
    clf = joblib.load('models/model.joblib') 
    climate=int(kwargs['temp'])
    rainfall=int(kwargs['rainfall'])
    pottasium=int(kwargs['pottasium'])
    phosphorus=int(kwargs['phosporus'])
    nitrogen=int(kwargs['nitrogen'])
    season=int(kwargs['season'])
    soiltype=int(kwargs['soiltype'])
    if (season==0):
        if(soiltype==0):
            return JsonResponse(
                {'crop':json.dumps(clf.predict(
                [[climate,rainfall,nitrogen,phosphorus,pottasium,1,0,1,0,0,0,0]]).tolist())}
            )                    
        elif(soiltype==1):
            return JsonResponse({'crop':json.dumps(clf.predict([[climate,rainfall,nitrogen,phosphorus,pottasium,1,0,0,1,0,0,0]]).tolist())})                    
        elif(soiltype==2):
            return JsonResponse({'crop':json.dumps(clf.predict([[climate,rainfall,nitrogen,phosphorus,pottasium,1,0,0,0,1,0,0]]).tolist())})
        elif(soiltype==3):
            return JsonResponse({'crop':json.dumps(clf.predict([[climate,rainfall,nitrogen,phosphorus,pottasium,1,0,0,0,0,1,0]]).tolist())})
        elif(soiltype==4):
            return JsonResponse({'crop':json.dumps(clf.predict([[climate,rainfall,nitrogen,phosphorus,pottasium,1,0,0,0,0,0,1]]).tolist())})
    else:
        if(soiltype==0):
            return JsonResponse({'crop':json.dumps(clf.predict([[climate,rainfall,nitrogen,phosphorus,pottasium,0,1,1,0,0,0,0]]).tolist())})
        elif(soiltype==1):
            return JsonResponse({'crop':json.dumps(clf.predict([[climate,rainfall,nitrogen,phosphorus,pottasium,0,1,0,1,0,0,0]]).tolist())})
        elif(soiltype==2):
            return JsonResponse({'crop':json.dumps(clf.predict([[climate,rainfall,nitrogen,phosphorus,pottasium,0,1,0,0,1,0,0]]).tolist())})
        elif(soiltype==3):
            return JsonResponse({'crop':json.dumps(clf.predict([[climate,rainfall,nitrogen,phosphorus,pottasium,0,1,0,0,0,1,0]]).tolist())})
        elif(soiltype==4):
            return JsonResponse({'crop':json.dumps(clf.predict([[climate,rainfall,nitrogen,phosphorus,pottasium,0,1,0,0,0,0,1]]).tolist())})