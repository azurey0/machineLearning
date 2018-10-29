import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class perceptron(object):

    def __init__(self, learningRate=0.000025, iters=50):
        self.learningRate=learningRate
        self.iters = iters
        self.theta1=[]
        self.theta2=[]
        self.theta3 = []

    def fit(self, X, y):
        # X = m * n: m is num of item, n is num of dimension
        self.wb = 0.2 * np.random.rand(1 + X.shape[1]) - 0.1
        for i in range(self.iters):
            print(i, "th iteration: ")
            a = np.dot(self.wb[1:], X.T)+self.wb[0] #a=wx+b
            lost = - np.sum(y * a)
            predict = np.where(a <= 0.0, -1, 1)
            acc = np.sum(y == predict) / y.shape[0]
            update = self.learningRate * (y - a)
            self.wb[1:] += np.sum(update * X.T, axis=1)
            self.wb[0] += np.sum(update)


            self.theta1.append(self.wb[1])
            self.theta2.append(self.wb[2])

            print('lost ', lost)
            print('accuracy', acc)
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
        '''
        X_all = iris.data[:]
        plotDataSet(X_all)
        #画图可以看出第二种和第三种iris线性不可分，以下只采用前两种iris的数据
        '''
        X = iris.data[:100]
        y = iris.target[:100]
        y = np.where(y == 1, 1, -1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        ppn = perceptron(learningRate=0.000025, iters=50)
        result = ppn.fit(X_train, y_train)
        print('train done')
        #print("result:\n theta1: " +str(result.theta1))
        #plot_decision_regions(X, y, clf=ppn, legend=2)



if __name__  == '__main__':
    main()