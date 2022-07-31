import pickle
import numpy as np
from django.shortcuts import render


def home(request):
    return render(request,'home.html')



def result(request):
    model=pickle.load(open('iri.pkl','rb'))
    data1 = request.POST['a']
    data2 = request.POST['b']
    data3 = request.POST['c']
    data4 = request.POST['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return  render(request,'result.html',{'ans':pred})


