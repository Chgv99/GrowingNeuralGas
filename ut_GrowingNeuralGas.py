import unittest
import tensorflow as tf

from GrowingNeuralGas import GrowingNeuralGas
from Graph import Graph

class ut_GrowingNeuralGas(unittest.TestCase):
    def test_findNearestUnit_and_findSecondNearestUnit(self):
        A = tf.constant([[0., 0., 0.], [0., 1., 0.], [0.5, 0.1, 0.], [2., 3., 0.5], [0.1, 0.5, 0.5]], dtype=tf.float32)
        xi = tf.expand_dims(tf.constant([0.45, 0.15, 0.05]), 0)

        growingNeuralGas = GrowingNeuralGas()
        indexNearestUnit = growingNeuralGas.findNearestUnit(xi, A)
        indexSecondNearestUnit = growingNeuralGas.findSecondNearestUnit(xi, A)

        self.assertEqual(indexNearestUnit, 2)
        self.assertEqual(indexSecondNearestUnit, 0)

    def test_pruneA(self):
        aBase = tf.constant([[0., 0., 0.], [0.5, 0.1, 0.], [0.1, 0.5, 0.5]], dtype=tf.float32)
        nBase = []
        nBase.append(Graph(0, [1, 2], [71, 31]))
        nBase.append(Graph(1, [0, 2], [21, 41]))
        nBase.append(Graph(2, [0, 1], [11, 32]))

        aTest = tf.Variable([[0., 0., 0.], [0., 1., 0.], [0.5, 0.1, 0.], [2., 3., 0.5], [0.1, 0.5, 0.5]], dtype=tf.float32)
        nTest = []
        nTest.append(Graph(0, [2, 4], [71, 31]))
        nTest.append(Graph(1))
        nTest.append(Graph(2, [0, 4], [21, 41]))
        nTest.append(Graph(3))
        nTest.append(Graph(4, [0, 2], [11, 32]))

        growingNeuralGasTest = GrowingNeuralGas()
        growingNeuralGasTest.A = aTest
        growingNeuralGasTest.N = nTest

        growingNeuralGasTest.pruneA()

        self.assertTrue(tf.math.reduce_all(aBase == growingNeuralGasTest.A))
        for graphBase, graphTest in zip(nBase, growingNeuralGasTest.N):
            self.assertEqual(graphBase, graphTest)

    def test_getGraphConnectedComponents(self):
        connectedComponentBase_0 = [Graph(0, [4, 9], [0.0, 0.0]), Graph(4, [0, 8, 11], [0.0, 0.0, 0.0]), Graph(8, [4, 9], [0.0, 0.0]), Graph(9, [0, 8], [0.0, 0.0]),
                                    Graph(11, [4], [0.0])]
        connectedComponentBase_1 = [Graph(1, [6], [0.0]), Graph(2, [5, 6], [0.0, 0.0]), Graph(5, [2, 6], [0.0, 0.0]), Graph(6, [1, 2, 5], [0.0, 0.0, 0.0])]
        connectedComponentBase_2 = [Graph(3, [7, 10], [0.0, 0.0]), Graph(7, [3, 10], [0.0, 0.0]), Graph(10, [3, 7], [0.0, 0.0])]

        nTest = [Graph(0, [4, 9], [0.0, 0.0]), Graph(1, [6], [0.0]), Graph(2, [5, 6], [0.0, 0.0]), Graph(3, [7, 10], [0.0, 0.0]), Graph(4, [0, 8, 11], [0.0, 0.0, 0.0]),
                 Graph(5, [2, 6], [0.0, 0.0]), Graph(6, [1, 2, 5], [0.0, 0.0, 0.0]), Graph(7, [3, 10], [0.0, 0.0]), Graph(8, [4, 9], [0.0, 0.0]), Graph(9, [0, 8], [0.0, 0.0]),
                 Graph(10, [3, 7], [0.0, 0.0]), Graph(11, [4], [0.0])]

        growingNeuralGasTest = GrowingNeuralGas()
        growingNeuralGasTest.N = nTest

        numberConnectedComponent, connectedComponents = growingNeuralGasTest.getGraphConnectedComponents()
        self.assertTrue(numberConnectedComponent == 3)
        for graphBase, graphTest in zip(connectedComponentBase_0, connectedComponents[0]):
            self.assertEqual(graphBase, graphTest)
        for graphBase, graphTest in zip(connectedComponentBase_1, connectedComponents[1]):
            self.assertEqual(graphBase, graphTest)
        for graphBase, graphTest in zip(connectedComponentBase_2, connectedComponents[2]):
            self.assertEqual(graphBase, graphTest)


if __name__ == '__main__':
    unittest.main()
