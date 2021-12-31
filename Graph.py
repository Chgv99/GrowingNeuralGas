class Graph(object):
    def __init__(self, id, neighborhood=None, ageNeighborhood=None):
        if neighborhood is None:
            neighborhood = []
        if ageNeighborhood is None:
            ageNeighborhood = []
        self.__id = id
        self.__neighborhood = neighborhood
        self.__ageNeighborhood = ageNeighborhood

    def addNeighbour(self, neighbour, ageNeighbour):
        self.__neighborhood.append(neighbour)
        self.__ageNeighborhood.append(ageNeighbour)

    def removeNeighbour(self, neighbour):
        self.__ageNeighborhood.pop(self.__neighborhood.index(neighbour))
        self.__neighborhood.remove(neighbour)

    def incrementAgeNeighborhood(self, increment):
        self.__ageNeighborhood = [ageNeighbour + increment for ageNeighbour in self.__ageNeighborhood]

    def incrementAgeNeighbour(self, neighbour, increment):
        self.__ageNeighborhood[self.__neighborhood.index(neighbour)] += increment

    def setAge(self, neighbour, age):
        self.__ageNeighborhood[self.__neighborhood.index(neighbour)] = age

    def pruneGraph(self, a_max):
        neighborhoodToRemove = [neighbour for neighbour, ageNeighbour in zip(self.__neighborhood, self.__ageNeighborhood) if ageNeighbour >= a_max]
        for neighbour in neighborhoodToRemove:
            self.removeNeighbour(neighbour)

    @property
    def id(self):
        return self.__id

    @property
    def neighborhood(self):
        return self.__neighborhood

    @property
    def ageNeighborhood(self):
        return self.__ageNeighborhood

    @id.setter
    def id(self, id):
        self.__id = id

    @neighborhood.setter
    def neighborhood(self, neighborhood):
        self.__neighborhood = neighborhood

    @ageNeighborhood.setter
    def ageNeighborhood(self, ageNeighborhood):
        self.__ageNeighborhood = ageNeighborhood

    def __eq__(self, other):
        if isinstance(other, self.__class__):

            if other.id != self.id:
                return False

            if other.neighborhood.__len__() != self.neighborhood.__len__():
                return False
            else:
                if other.neighborhood.__len__() and self.neighborhood.__len__():
                    _otherNeighborhood = other.neighborhood.copy()
                    _selfNeighborhood = self.neighborhood.copy()
                    _otherNeighborhood.sort()
                    _selfNeighborhood.sort()
                    for elementOtherNeighborhood, elementSelfNeighborhood in zip(_otherNeighborhood, _selfNeighborhood):
                        if elementOtherNeighborhood != elementSelfNeighborhood:
                            return False

            if other.ageNeighborhood.__len__() != self.ageNeighborhood.__len__():
                return False
            else:
                if other.ageNeighborhood.__len__() and self.ageNeighborhood.__len__():
                    _otherAgeNeighborhood = other.ageNeighborhood.copy()
                    _selfAgeNeighborhood = self.ageNeighborhood.copy()
                    _otherAgeNeighborhood.sort()
                    _selfAgeNeighborhood.sort()
                    for elementOtherAgeNeighborhood, elementSelfAgeNeighborhood in zip(_otherAgeNeighborhood, _selfAgeNeighborhood):
                        if elementOtherAgeNeighborhood != elementSelfAgeNeighborhood:
                            return False

            return True

        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)