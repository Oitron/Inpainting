import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
from random import shuffle

from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error as MSE


class Preambule:
    def __init__(self, method="LR", alpha=0.01):
        self.method = method
        self.alpha = alpha
        self.models = []
        self.coefs = []

    #Convert labels to [-1,1]
    def classification_binaire(self, liste, label):
        y = list(map(lambda i: int(i == label), liste))
        return np.array(y)*2-1 

    #régression linéaire
    def start_fit(self, x_train, y_train):
        labels = sorted(set(y_train))
        for label in labels:
            x = x_train
            y = self.classification_binaire(y_train, label)
            x_y = list(zip(x,y))
            plus_zero = list(filter(lambda i: i[1]==1, x_y))
            moins_zero = list(filter(lambda i: i[1]==-1, x_y))
            shuffle(plus_zero)
            shuffle(moins_zero)
            size = min(len(plus_zero),len(moins_zero))
            plus_zero = plus_zero[:size]
            moins_zero = moins_zero[:size]
            x_y = plus_zero+moins_zero
            shuffle(x_y)
            x,y = zip(*x_y)
            x = np.array(x)
            y = np.array(y)
            if(self.method == "LR"):
                model = LR().fit(x,y)
            elif(self.method == "Ridge"):
                model = Ridge(alpha = self.alpha).fit(x,y)
            elif(self.method == "Lasso"):
                model = Lasso(alpha = self.alpha).fit(x,y)
            self.models.append(model)
        for model in self.models:
            self.coefs.append(model.coef_)

    def get_coefs_mean(self):
        return np.mean(np.abs(self.coefs))
    
    def get_nb_coefs_zero(self):
        return np.sum(np.array(self.coefs)==0)

    def predict(self, x_test):
        predicts = []
        for model in self.models:
            predict = model.predict(x_test)
            predicts.append(predict)
        return (np.array(predicts)>0).argmax(0)

    def score(self, y_test, y_predict):
        #return f1_score(y_test, y_predict, average="micro")
        return f1_score(y_test, y_predict, average=None).mean()


    def MSE_score(self,y_test,y_predict):
        return MSE(y_predict, y_test)

    def L1_score(self,y_test,y_predict):
        self.coefs = np.array(self.coefs)
        nb_labels = self.coefs.shape[0]
        #size = self.coefs.shape[0]*self.coefs.shape[1]
        all_coefs = []
        for i in range(self.coefs.shape[0]):
            coef = 0
            for j in range(self.coefs.shape[1]):
                coef += self.coefs[i][j]
            coef /= nb_labels
            all_coefs.append(coef)
        all_coefs = np.array(all_coefs)
        p = self.alpha*np.sum(all_coefs**2)
        return (MSE(y_predict, y_test)+p)

    def L2_score(self,y_test,y_predict):
        self.coefs = np.array(self.coefs)
        nb_labels = self.coefs.shape[0]
        #size = self.coefs.shape[0]*self.coefs.shape[1]
        all_coefs = []
        for i in range(self.coefs.shape[0]):
            coef = 0
            for j in range(self.coefs.shape[1]):
                coef += self.coefs[i][j]
            coef /= nb_labels
            all_coefs.append(coef)
        all_coefs = np.array(all_coefs)
        p = self.alpha*np.sum(np.abs(all_coefs))
        return (MSE(y_predict, y_test)+p)

    def display_poids(self, h, w):
        i = 1
        for coef in self.coefs:
            plt.figure('poids display',figsize = (30,10)).add_subplot(2, 5, i)
            title = "label " + str(i-1)
            plt.title(title)
            plt.xlabel("coef")
            plt.ylabel("poids")
            plt.bar(range(len(coef)), coef)
            #sns.heatmap(data=coef.reshape(h,w),square=True) 
            i+=1
        plt.show()