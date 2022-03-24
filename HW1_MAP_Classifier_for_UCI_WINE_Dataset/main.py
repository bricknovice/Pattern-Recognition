import csv
import numpy as np 
import random
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
        X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.5)
        class1_item = []
        class2_item = []
        class3_item = []
        X_test = np.asarray(X_test).astype(np.float)
        y_test = np.asarray(y_test).astype(np.float)

        row = 178
        for i in range(len(y_train)):
            if(y_train[i]=='1'):
                class1_item.append(X_train[i])
            elif(y_train[i]=='2'):
                class2_item.append(X_train[i])
            elif(y_train[i]=='3'):
                class3_item.append(X_train[i])
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
        correct = 0
        for i in range(len(X_test)):
            comp1 = -1 * np.log(P1) + (1 / 2) * np.dot(np.dot(np.transpose((np.transpose(X_test[i]) - mean1)) , np.linalg.inv(cov1)) , (np.transpose(X_test[i]) - mean1)) + 1 / 2 *np.log(np.linalg.det(cov1))
            comp2 = -1 * np.log(P2) + (1 / 2) * np.dot(np.dot(np.transpose((np.transpose(X_test[i]) - mean2)) , np.linalg.inv(cov2)) , (np.transpose(X_test[i]) - mean2)) + 1 / 2 *np.log(np.linalg.det(cov2))
            comp3 = -1 * np.log(P3) + (1 / 2) * np.dot(np.dot(np.transpose((np.transpose(X_test[i]) - mean3)) , np.linalg.inv(cov3)) , (np.transpose(X_test[i]) - mean3)) + 1 / 2 *np.log(np.linalg.det(cov3))
            if( (min(comp1,comp2,comp3) == comp1 and y_test[i]==1) or (min(comp1,comp2,comp3) == comp2 and y_test[i]==2) or (min(comp1,comp2,comp3) == comp3 and y_test[i]==3)):
                correct += 1
        print(str(correct/(row/2))+"%")
    