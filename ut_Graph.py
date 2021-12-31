import unittest

from Graph import Graph

class ut_Graph(unittest.TestCase):
    def test_addNeighborhood(self):
        graphBase = Graph(7, [1, 2, 3], [10.0, 20.0, 30.0])

        graphTest = Graph(7)
        graphTest.addNeighbour(1, 10.0)
        graphTest.addNeighbour(2, 20.0)
        graphTest.addNeighbour(3, 30.0)

        self.assertEqual(graphBase, graphTest)

    def test_removeNeighborhood(self):
        graphBase = Graph(7, [1, 3], [10.0, 30.0])

        graphTest = Graph(7, [1, 2, 3], [10.0, 20.0, 30.0])
        graphTest.removeNeighbour(2)

        self.assertEqual(graphBase, graphTest)

    def test_incrementAgeNeighborhood(self):
        graphBase = Graph(7, [1, 2, 3], [11.0, 21.0, 31.0])

        graphTest = Graph(7, [1, 2, 3], [10.0, 20.0, 30.0])
        graphTest.incrementAgeNeighborhood(1.0)

        self.assertEqual(graphBase, graphTest)

    def test_incrementAgeNeighbour(self):
        graphBase = Graph(7, [1, 2, 3], [10.0, 21.0, 30.0])

        graphTest = Graph(7, [1, 2, 3], [10.0, 20.0, 30.0])
        graphTest.incrementAgeNeighbour(2, 1.0)

        self.assertEqual(graphBase, graphTest)


    def test_incrementAgeNeighborhood_when_graph_have_not_neighborhood(self):
        graphBase = Graph(7, [], [])

        graphTest = Graph(7, [], [])
        graphTest.incrementAgeNeighborhood(1)

        self.assertEqual(graphBase, graphTest)

    def test_pruneGraph(self):
        graphBase_30 = Graph(7, [1, 2, 4, 5], [25.0, 20.0, 25.0, 20.0])
        graphBase_25 = Graph(7, [2, 5], [20.0, 20.0])
        graphBase_20 = Graph(7)

        graphTest = Graph(7, [1, 2, 3, 4, 5], [25.0, 20.0, 30.0, 25.0, 20.0])

        graphTest.pruneGraph(30.0)
        self.assertEqual(graphTest, graphBase_30)

        graphTest.pruneGraph(25.0)
        self.assertEqual(graphTest, graphBase_25)

        graphTest.pruneGraph(20.0)
        self.assertEqual(graphTest, graphBase_20)


if __name__ == '__main__':
    unittest.main()