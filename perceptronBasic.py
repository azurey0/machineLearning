import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class perceptron(object):

    def __init__(self, learningRate=0.025, iters=5):
        self.learningRate=learningRate
        self.iters = iters
        self.theta1=[]
        self.theta2=[]
        self.theta3 = []


    def predict(self, xi):
        print('self.wb[1:]', self.wb[1:],'xi', xi)
        wxPlusBias= np.dot(self.wb[1:], xi) + self.wb[0]
        return np.where(wxPlusBias <= 0.0, -1 ,1)


    def fit(self, X, y):
        #print(X.shape[1])
        self.wb = np.zeros(1+ X.shape[1])
        self.errors = []
        for i in range(self.iters):
            errors = 0
            print( i ,"th iteration: ")
            for xi, yi in zip(X,y): #对每个训练样本
                update = self.learningRate * (yi - self.predict(xi))
                #如果update不为0，则下列式子不为0，进行更新。权重变化为正常的2倍。
                self.wb[1: ] += update * xi
                self.wb[0] += update
                errors += int(update != 0.0)#误分类样本数

            self.theta1.append(self.wb[1])
            self.theta2.append(self.wb[2])

            if errors == 0: #如果没有误分类，训练终止
                print('accuracy =1. return')
                return self

            print('accuracy: ', (1 - errors / 100))
        return self

def plotDataSet( dataset):
    plt.scatter(dataset[:50, 0], dataset[:50, 1], color = 'green', marker = 'o', label = 'Iris-Setosa')
    plt.scatter(dataset[50:100, 0], dataset[50:100, 1], color='yellow', marker='x', label='Iris-Versicolour')
    plt.scatter(dataset[100:150, 0], dataset[100:150, 1], color='blue', marker='v', label='Iris-Virginica')
    plt.xlabel('sepal length in cm')
    plt.ylabel('sepal width in cm')
    plt.legend(loc='upper left')
    plt.show()

from matplotlib.colors import ListedColormap

def main():
        iris = load_iris()
        print('data attributes: ', dir(iris))
        print('data description：　', iris.DESCR )

        X_all = iris.data[:]
        plotDataSet(X_all)
        #画图可以看出第二种和第三种iris线性不可分，以下只采用前两种iris的数据

        X = iris.data[:100]

        y = iris.target[:100]
        y = np.where(y == 1, 1, -1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        ppn = perceptron(learningRate=0.025, iters=5)
        result = ppn.fit(X_train, y_train)
        print('train done')
        #print("result:\n theta1: " +str(result.theta1))
        #plot_decision_regions(X, y, clf=ppn, legend=2)



if __name__  == '__main__':
    main()