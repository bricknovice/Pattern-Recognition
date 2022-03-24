import csv
import numpy as np 
import random
import math
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    with open('wine.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        x = list(reader)
        
        x = x[1:]
        y = []
        for item in x:
            temp = item.pop(0)
            y.append(temp)
        class1_item = []
        class2_item = []
        class3_item = []

        row = 178
        for i in range(len(y)):
            if(y[i]=='1'):
                class1_item.append(x[i])
            elif(y[i]=='2'):
                class2_item.append(x[i])
            elif(y[i]=='3'):
                class3_item.append(x[i])
        P1 = len(class1_item)/row
        P2 = len(class2_item)/row
        P3 = len(class3_item)/row
        class1_item = np.asarray(class1_item).astype(np.float)
        class2_item = np.asarray(class2_item).astype(np.float)
        class3_item = np.asarray(class3_item).astype(np.float)
        
        mean1 = class1_item.mean(0)
        cov1 = np.cov(np.transpose(class1_item))

        mean2 = class2_item.mean(0)
        cov2 = np.cov(np.transpose(class2_item))
        
        mean3 = class3_item.mean(0)
        cov3 = np.cov(np.transpose(class3_item))

        comp1 = 1/8 * np.dot(np.dot(np.transpose(mean2 - mean1),np.linalg.inv((cov1+cov2)/2)),mean2-mean1) + 1/2 * np.log(np.linalg.det((cov1+cov2)/2)/math.sqrt(np.linalg.det(cov1)*np.linalg.det(cov2))) 
        error12 = math.sqrt(P1 * P2 ) * math.exp(-1*comp1)
        comp2 = 1/8 * np.dot(np.dot(np.transpose(mean3 - mean2),np.linalg.inv((cov2+cov3)/2)),mean3-mean2) + 1/2 * np.log(np.linalg.det((cov2+cov3)/2)/math.sqrt(np.linalg.det(cov2)*np.linalg.det(cov3))) 
        error23 = math.sqrt(P2 * P3 )  * math.exp(-1*comp2)
        comp3 = 1/8 * np.dot(np.dot(np.transpose(mean3 - mean1),np.linalg.inv((cov1+cov3)/2)),mean3-mean1) + 1/2 * np.log(np.linalg.det((cov1+cov3)/2)/math.sqrt(np.linalg.det(cov1)*np.linalg.det(cov3))) 
        error13 = math.sqrt(P1 * P3 ) * math.exp(-1*comp3) 
        print(error12)
        print(error23)
        print(error13)