import functools

import numpy as np
import tensorflow as tf

from datetime import datetime
from tqdm import tqdm
import os

import csv

import json

from Graph import Graph
from GrowingNeuralGasPlotter import GrowingNeuralGasPlotter

class GrowingNeuralGas(object):

    def __init__(self, epsilon_a=.1, epsilon_n=.05, a_max=10, eta=5, alpha=.1, delta=.1, maxNumberUnits=1000, maxNumberCC=10):
        self.A = None
        self.N = []
        self.error_ = None
        self.epsilon_a = epsilon_a
        self.epsilon_n = epsilon_n
        self.a_max = a_max
        self.eta = eta
        self.alpha = alpha
        self.delta = delta
        self.maxNumberUnits = maxNumberUnits
        self.maxNumberCC = maxNumberCC

    def incrementAgeNeighborhood(self, indexNearestUnit):
        self.N[indexNearestUnit].incrementAgeNeighborhood(1.0)
        for indexNeighbour in self.N[indexNearestUnit].neighborhood:
            self.N[indexNeighbour].incrementAgeNeighbour(indexNearestUnit, 1.0)

    def findNearestUnit(self, xi, A):
        return tf.math.argmin(tf.math.reduce_sum(tf.math.pow(A - xi, 2), 1))

    def findSecondNearestUnit(self, xi, A):
        indexNearestUnit = self.findNearestUnit(xi, A)
        error_ = tf.constant(tf.math.reduce_sum(tf.math.pow(A - xi, 2), 1), dtype=tf.float32).numpy()
        error_[indexNearestUnit] = np.Inf
        return tf.math.argmin(tf.constant(error_))

    def findIndexNeighbourMaxError(self, indexUnitWithMaxError_):
        index = tf.squeeze(tf.math.argmax(tf.gather(self.error_, self.N[indexUnitWithMaxError_].neighborhood)), 0)
        indexNeighbourMaxError = self.N[indexUnitWithMaxError_].neighborhood[index]
        return indexNeighbourMaxError

    def pruneA(self):
        indexToNotRemove = [index for index in tf.range(self.N.__len__()) if self.N[index].neighborhood.__len__() > 0]
        self.A = tf.Variable(tf.gather(self.A, indexToNotRemove, axis=0))

        for graphIndex in reversed(range(self.N.__len__())):
            if self.N[graphIndex].neighborhood.__len__() == 0:
                for pivot in range(graphIndex + 1, self.N.__len__()):
                    self.N[pivot].id -= 1
                    for indexN in range(self.N.__len__()):
                        for indexNeighbothood in range(self.N[indexN].neighborhood.__len__()):
                            if self.N[indexN].neighborhood[indexNeighbothood] == pivot:
                                self.N[indexN].neighborhood[indexNeighbothood] -= 1
                self.N.pop(graphIndex)

    def save(self, path, matlab=False):
        if not os.path.exists(path):
            os.mkdir(path)
        if matlab:
            np.savetxt(path + "/matlab_a.csv", self.A, delimiter=',')
            gruposProfe = self.getGraphConnectedComponents()[1]
            gruposAlumno = self.groups()
            allProfe = [[[j.id, [k.numpy().tolist() for k in j.neighborhood]] for j in i] for i in gruposProfe]
            allAlumno = [[[self.N[j].id, [k.numpy().tolist() for k in self.N[j].neighborhood]] for j in i] for i in gruposAlumno]
            ids = [[j.id for j in i] for i in gruposProfe]
            vecinos = [[[k.numpy().tolist() for k in j.neighborhood] for j in i] for i in gruposProfe]

            with open(path + "/matlab_grupos_alumno.json", "wt") as f:
                f.write(json.dumps(gruposAlumno))
            with open(path + "/matlab_all_profe.json", "wt") as f:
                f.write(json.dumps(allProfe))
            with open(path + "/matlab_all_alumno.json", "wt") as f:
                f.write(json.dumps(allAlumno))
            with open(path + "/matlab_ids.json", "wt") as f:
                f.write(json.dumps(ids))
            with open(path + "/matlab_grupos_profe.json", "wt") as f:
                f.write(json.dumps(vecinos))

        np.save(path + "/a_save.npy", self.A.numpy())
        np.save(path + "/error_save.npy", self.error_.numpy())

        with open(path + "/n.json", "wt") as f:
            f.write(json.dumps([[i.ageNeighborhood, i.id, [j.numpy().tolist() for j in i.neighborhood]] for i in self.N]))

        dic = {
            "epsilon_a":self.epsilon_a,
            "epsilon_n":self.epsilon_n,
            "a_max":self.a_max,
            "eta":self.eta,
            "alpha":self.alpha,
            "delta":self.delta,
            "maxNumberUnits":self.maxNumberUnits,
        }
        with open(path + "/variables.json", "wt") as f:
            f.write(json.dumps(dic))

    def load(self, path):
        if os.path.exists(path):
            self.A = tf.Variable(np.load(path + "/a_save.npy"))
            self.error_ = tf.Variable(np.load(path + "/error_save.npy"))

            n = json.loads(open(path + "/n.json", "r").read())
            #self.N = [[i[0], i[1], [tf.constant(j, dtype=tf.int64) for j in i[2]]] for i in n]
            self.N = [Graph(i[1], [tf.constant(j, dtype=tf.int64) for j in i[2]], i[0]) for i in n]

            dic = json.loads(open(path + "/variables.json", "r").read())
            self.epsilon_n = dic["epsilon_n"]
            self.a_max = dic["a_max"]
            self.eta = dic["eta"]
            self.alpha = dic["alpha"]
            self.delta = dic["delta"]
            self.maxNumberUnits = dic["maxNumberUnits"]
        else:
            #except Exception as e:
            print("[Load] \"" + path + "\" does not exist.")

    def groups(self):
        groups = [[y.numpy().tolist() for y in x.neighborhood] for x in self.N]

        visited = []
        res = []
        for i in range(len(groups)):
            if i not in visited:
                visiting = [i]
                for j in visiting:
                    visiting.extend([x for x in groups[j] if x not in visiting])
                res.append(visiting)
                visited.extend(visiting)
        return res

    def getGraphConnectedComponents(self):
        connectedComponentIndeces = list(range(self.N.__len__()))
        for graphIndex in range(self.N.__len__()):
            for neighbourIndex in self.N[graphIndex].neighborhood:
                if connectedComponentIndeces[graphIndex] <= connectedComponentIndeces[neighbourIndex]:
                    connectedComponentIndeces[neighbourIndex] = connectedComponentIndeces[graphIndex]
                else:
                    aux = connectedComponentIndeces[graphIndex]
                    for pivot in range(graphIndex, self.N.__len__()):
                        if connectedComponentIndeces[pivot] == aux:
                            connectedComponentIndeces[pivot] = connectedComponentIndeces[neighbourIndex]
        uniqueConnectedComponentIndeces = functools.reduce(lambda cCI, index: cCI.append(index) or cCI if index not in cCI else cCI, connectedComponentIndeces, [])
        connectedComponents = []
        for connectedComponentIndex in uniqueConnectedComponentIndeces:
            connectedComponent = []
            for index in range(connectedComponentIndeces.__len__()):
                if connectedComponentIndex == connectedComponentIndeces[index]:
                    connectedComponent.append(self.N[index])
            connectedComponents.append(connectedComponent)
        return uniqueConnectedComponentIndeces.__len__(), connectedComponents

    def predict(self, X):
        # indexNearestUnit es usado como id pero no sabemos si coincide
        indexNearestUnit = self.findNearestUnit(X, self.A)
        _, groups = self.getGraphConnectedComponents()
        for i, e in enumerate(groups):
            if indexNearestUnit in [j.id for j in e]:
                return i

    def fit(self, trainingX, numberEpochs, date, dataset):

        path = os.getcwd() + '//output//' + dataset + "-" + date + "//"
        pathConn = os.getcwd() + '//outputConnections//' + dataset + "-" + date + "//"

        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(pathConn):
            os.mkdir(pathConn)
        if not os.path.exists(pathConn + "epochs"):
            os.mkdir(pathConn + "epochs")

        self.A = tf.Variable(tf.random.normal([2, trainingX.shape[1]], 0.0, 1.0, dtype=tf.float32))
        self.N.append(Graph(0))
        self.N.append(Graph(1))
        self.error_ = tf.Variable(tf.zeros([2, 1]), dtype=tf.float32)
        epoch = 0
        numberProcessedRow = 0
        count = 0
        plotStep = 10
        #while epoch < numberEpochs and (self.A.shape[0] < self.maxNumberUnits or self.getGraphConnectedComponents()[0] < self.maxNumberCC):
        while epoch < numberEpochs:
                shuffledTrainingX = tf.random.shuffle(trainingX)
                for row_ in tqdm(tf.range(shuffledTrainingX.shape[0])):
                    xi = shuffledTrainingX[row_]
                    indexNearestUnit = self.findNearestUnit(xi, self.A)
                    self.incrementAgeNeighborhood(indexNearestUnit)
                    indexSecondNearestUnit = self.findSecondNearestUnit(xi, self.A)

                    self.error_[indexNearestUnit].assign(self.error_[indexNearestUnit] + tf.math.reduce_sum(tf.math.squared_difference(xi, self.A[indexNearestUnit])))

                    self.A[indexNearestUnit].assign(self.A[indexNearestUnit] + self.epsilon_a * (xi - self.A[indexNearestUnit]))
                    for indexNeighbour in self.N[indexNearestUnit].neighborhood:
                        self.A[indexNeighbour].assign(self.A[indexNeighbour] + self.epsilon_n * (xi - self.A[indexNeighbour]))

                    if indexSecondNearestUnit in self.N[indexNearestUnit].neighborhood:
                        self.N[indexNearestUnit].setAge(indexSecondNearestUnit, 0.0)
                        self.N[indexSecondNearestUnit].setAge(indexNearestUnit, 0.0)
                    else:
                        self.N[indexNearestUnit].addNeighbour(indexSecondNearestUnit, 0.0)
                        self.N[indexSecondNearestUnit].addNeighbour(indexNearestUnit, 0.0)

                    for graph in self.N:
                        graph.pruneGraph(self.a_max)

                    self.pruneA()


                    #print("GrowingNeuralGas::numberUnits: {} - GrowingNeuralGas::numberGraphConnectedComponents: {}".format(self.A.shape[0], numberGraphConnectedComponents))
                    count = (count + 1) % plotStep
                    if count == 0:
                        numberGraphConnectedComponents, _ = self.getGraphConnectedComponents()
                        GrowingNeuralGasPlotter.plotConnection(pathConn,
                                                               'graphConnectedComponents_' + '{}_{}'.format(
                                                                   self.A.shape[0],
                                                                   len(self.groups())),
                                                               self.A,
                                                               self.N)

                    if not (numberProcessedRow + 1) % self.eta:
                        indexUnitWithMaxError_ = tf.squeeze(tf.math.argmax(self.error_), 0)
                        indexNeighbourWithMaxError_ = self.findIndexNeighbourMaxError(indexUnitWithMaxError_)

                        self.A = tf.Variable(tf.concat([self.A, tf.expand_dims(0.5 * (self.A[indexUnitWithMaxError_] + self.A[indexNeighbourWithMaxError_]), 0)], 0))

                        self.N.append(Graph(self.A.shape[0] - 1, [indexUnitWithMaxError_, indexNeighbourWithMaxError_], [0.0, 0.0]))
                        self.N[indexUnitWithMaxError_].removeNeighbour(indexNeighbourWithMaxError_)
                        self.N[indexUnitWithMaxError_].addNeighbour(tf.constant(self.A.shape[0] - 1, dtype=tf.int64), 0.0)
                        self.N[indexNeighbourWithMaxError_].removeNeighbour(indexUnitWithMaxError_)
                        self.N[indexNeighbourWithMaxError_].addNeighbour(tf.constant(self.A.shape[0] - 1, dtype=tf.int64), 0.0)

                        self.error_[indexUnitWithMaxError_].assign(self.error_[indexUnitWithMaxError_] * self.alpha)
                        self.error_[indexNeighbourWithMaxError_].assign(self.error_[indexNeighbourWithMaxError_] * self.alpha)
                        self.error_ = tf.Variable(tf.concat([self.error_,  tf.expand_dims(self.error_[indexUnitWithMaxError_], 0)], 0))

                    self.error_.assign(self.error_ * self.delta)
                    numberProcessedRow += 1
                epoch += 1
                GrowingNeuralGasPlotter.plotConnection(pathConn + "epochs//",
                                                       'graphConnectedComponents_' + '{}_{}_{}'.format(
                                                           epoch,
                                                           self.A.shape[0],
                                                           numberGraphConnectedComponents),
                                                       self.A,
                                                       self.N)

