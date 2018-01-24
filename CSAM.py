# -*-coding:utf-8 -*
import numpy
import sys
import time
import math
import pickle
from numba import jit
from graphviz import Graph



#   Parameters
modeT = 0                   #Building mode for set T : 0, all elements from TSO's control area;
                            #1, only most influenced by a j contingency elements; 2, only elements influenced a j contingency above setJTh
modeI = 0                   #Building mode for set I : 0, all grid elements; 1, contingencies with IF > setITh;
                            #2, contingencies with normalized IF > setINTh; 3, contingencies satisfying both conditions
modeR = 0                   #Ring by ring approach : 0, complete assessment; 1, ring-by-ring approach with non-normalized parameter setRTh
                            #2, ring-by-ring approach with normalized parameter setRTh
setJTh = 0.05               #Threshold for the influence of a external contingency j on element t for t to be in set T
setITh = 0.1               #Threshold for the influence of an external contingency j on element t for j to be in set Iext
setINTh = 0.1              #Threshold for the normalized influence of an external contingency j on element t for j to be in set Iext
setRTh = 0.05               #Threshold for the N-2 influence factor of a r element to assess next ring
#Parameters used to build the Observability area
modeOA = 0                  #Building mode for OA : 0, all the external elements; 1, all elements with N-2 IF >= setOATh; 
                            #2, all elements with normalized N-2 IF >= setOAnTh; #3, all elements with N-2 IF >= setOATh OR normalized N-2 IF >= setOAnTh; 
                            #4, all elements with N-2 IF >= setOATh AND normalized N-2 IF > setOAnTh
setOATh = 0.05
setOAnTh = 0.05

#countryCode = "F"           #Used to define the control area (can be 1- or 2-characters long)
epsilon = 0.001              #As we are working on numeric value, values below epsilon are rounded to 0 and values between 1 - epsilon and 1 + epsilon are rounded to 1
iatlTH = 5000               #Threshold above which IATL are regarded as infinite.
ringChars = 7               #Significant characters used to determined rings : 8, one ring per node; 7, one ring per voltage level; 6, one ring per substation (all voltage)
fileUCT = "Winter Peak 20170118_1030_RE3_UX5.uct"

#pre-processing parameters
pMergeCouplers = True
pMergeXNodes = True
pMergeEquivalents = False


sys.setrecursionlimit(12000)

class Branch:
    nbBranches = 0
    def __init__(self, nameFrom, nameTo, order, impedance, IATL, type):
        #Description of the branch
        self.index = Branch.nbBranches
        self.nameFrom = nameFrom
        self.nameTo = nameTo
        self.nameBranch = nameFrom + " " + nameTo + " " + order
        #Characteristics of the branch
        self.nodeFrom = None
        self.nodeTo = None
        self.ring = 99
        self.connected = False
        self.tieLine = False
        self.type = type
        VBase = dictVBase[int(nameFrom[6:7])]
        self.impedance = impedance * SBase / (VBase * VBase)
        if IATL < iatlTH:
            self.PATL = IATL * math.sqrt(3) * VBase/1000
        else :
            self.PATL = 0
        self.PTDF = 1.0
        Branch.nbBranches +=1
    
    def coupleNodes(self, node1, node2):
        if self.nameFrom == node1:
            self.nameFrom = node2
        elif self.nameTo == node1:
            self.nameTo = node2
        else:
            print("Error while merging branch : " + self.nameBranch + " node:" + node1 + " was not found among " + self.nameFrom + " and " + self.nameTo)

    def __str__(self):
        return str(self.index) + "," + self.nameBranch + "," + self.nodeFrom.name + "," + self.nodeTo.name + "," + str(self.impedance) + "," + str(self.PATL) + "," + str(self.ring) + "," + str(self.tieLine)

    def insertInCA(self):
        if not self.isTieLine():
            for node in [self.nodeFrom, self.nodeTo]:
                node.insertInCA()

    def connectToGrid(self):
        self.connected = True
        for node in [self.nodeFrom, self.nodeTo]:
            node.connectToGrid()

    def increaseRing(self,ringIndex, dictNodes):
        for (node1, node2) in [(self.nodeFrom, self.nodeTo), (self.nodeTo, self.nodeFrom)]:
            if node1.ring == ringIndex and node2.ring == 99:
                equivalentNodeName = node2.getEquivalentNodeName()
                for elt in dictNodes[equivalentNodeName]:
                    if elt.isXNode():
                        elt.ring = ringIndex
                        for branch in elt.branches:
                            branch.increaseRing(ringIndex, dictNodes)
                    else:
                        elt.ring = ringIndex + 1
        self.ring = min((self.nodeFrom.ring, self.nodeTo.ring))

    def updateRing(self):
        self.ring = min((self.nodeFrom.ring, self.nodeTo.ring))

    def header():
        return "Index,Name,Node From,Node To,Impedance,PATL,Ring,Tie-Line"
    header = staticmethod(header)

    def applyCouplers(self, dictCouplers):
        if self.nameFrom in dictCouplers: self.nameFrom = dictCouplers[self.nameFrom]
        if self.nameTo in dictCouplers: self.nameTo = dictCouplers[self.nameTo]

    def setCountry(self):
        if self.nodeFrom.isXNode():
            self.country = self.nodeTo.country
        else:
            self.country = self.nodeFrom.country

    def isTieLine(self):
        if self.tieLine:
            return True
        else:
            return self.nodeFrom.isXNode() or self.nodeTo.isXNode()

class resultIF:
    def __init__(self, eltR, IFN1, nIFN1, IFN2, nIFN2, eltI, eltT, eltIn, eltTn, LODFit, LODFti):
        """
            Generates a result with :
            -eltR the element whose influence is assessed
            -IFN1 : the N-1 IF
            -nIFN1 : the normalized N-1 IF
            -IFN2 : the N-2 IF (according to CSAM)
            -nIFN2 : the normalized N-2 IF
            -eltI : a contingency i for which IFN2 is reached
            -eltT : an element from the CA for which IFN2 is reached
            -eltIn : a contingency i for which nIFN2 is reached
            -eltTn : an element from the CA for which nIFN2 is reached
        """
        self.eltR = eltR
        self.IFN1 = IFN1
        self.nIFN1 = nIFN1
        self.IFN2 = IFN2
        self.nIFN2 = nIFN2
        self.eltI = eltI
        self.eltT = eltT
        self.eltIn = eltIn
        self.eltTn = eltTn
        self.LODFit = LODFit
        self.LODFti = LODFti

    def header():
        return "name;ring;PATL;N-1 IF; N-1 nIF;IF;i;t;Tie-line;nIF;i;t;Tie-line;PATL;LODF i-t" + '\n'
    header = staticmethod(header)

    def __str__(self):
        resultat = self.eltR.nameBranch + ";" + str(self.eltR.ring) + ";" + str(self.eltR.PATL) + ";" + str(round(self.IFN1,4)).replace(".",",") + ";"
        resultat += str(round(self.nIFN1,4)).replace(".",",") + ";" + str(round(self.IFN2,4)).replace(".",",") + ";"
        resultat += self.eltI.nameBranch + ";" + self.eltT.nameBranch + ";" + str(self.eltT.tieLine) + ";" + str(round(self.nIFN2,4)).replace(".",",") + ";"
        resultat += self.eltIn.nameBranch + ";" + self.eltTn.nameBranch + ";" + str(self.eltTn.tieLine) + ";" + str(self.eltTn.PATL).replace(".",",") + ";"
        resultat += str(round(self.LODFit,4)).replace(".", ",") + ";" + str(round(self.LODFti,4)).replace(".", ",") + '\n'
        return resultat     

class finalResult:

    def __init__(self, branches, nodes, results):
        self.listBranches = branches
        self.listNodes = nodes
        self.listResults = results

    def getBranches(self):
        return self.listBranches

    def getNodes(self):
        return self.listNodes

    def getResults(self):
        return self.listResults

class GenerationUnit:
    nbGenerators = 0
    def __init__(self, name, power):
        self.index = GenerationUnit.nbGenerators
        self.node = None
        self.nodeName = name
        self.name = name
        self.power = power
        self.country = ""
        self.connected = False
        GenerationUnit.nbGenerators += 1

    def applyCouplers(self, dictCouplers):
        if self.nodeName in dictCouplers:
            self.nodeName = dictCouplers[self.nodeName]

class Node:
    nbNodes = 0
    def __init__(self, name):
        self.index = Node.nbNodes
        self.country = Node.getCountry(name)
        self.name = name
        self.branches = []
        self.generators = []
        self.ring = 99
        self.connected = False
        Node.nbNodes += 1

    def __str__(self,):
        return str(self.index) + "," + self.name + "," + str(self.ring) + "," + str(self.connected) + "," + str([elt.index for elt in self.branches])

    def insertInCA(self):
        if self.ring == 99:
            self.ring = 0
            for branch in self.branches:
                branch.insertInCA()

    def connectToGrid(self):
        if not self.connected:
            self.connected = True
            for branch in self.branches:
                branch.connectToGrid()

    def isXNode(self):
        if self.name[0:1] == "X" or self.name[0:1] == "D" and not self.name[1:2].isdigit():
            return True
        else:
            return False

    def isBorder(self):
        if len([branch for branch in self.branches if branch.tieLine]) > 0:
            return True
        else:
            return False

    def header():
        return "Index,Name,Ring,Connected,Branches"
    header = staticmethod(header)

    def getCountry(countryName):
        if countryName[0:1] == "D":
            if countryName[1:2].isdigit():
                return countryName[0:2]
            else:
                return "X"
        else:
            return countryName[0:1]
    getCountry = staticmethod(getCountry)

    def remove(self, setOfElements, setOfNodes):
        for branch in self.branches:
            setOfElements.remove(branch)
            [node for node in [branch.nodeFrom, branch.nodeTo] if node != self][0].branches.remove(branch)
        setOfNodes.remove(self)
        for i in range(len(setOfElements)):
            setOfElements[i].index = i
        for i in range(len(setOfNodes)):
            setOfNodes[i].index = i

    def getEquivalentNodeName(self):
        if self.isXNode():
            return self.name
        else:
            return self.name[:ringChars]

class geography:
    def __init__(self, fileName):
        self.nodes = {}
        with open(fileName, 'r') as fileGeography:
            for lineFields in [line.split(';') for line in fileGeography.read().split('\n')]:
                if len(lineFields) > 4:
                    if lineFields[0] == lineFields[0].upper():
                        nodeCode = lineFields[0]
                        nodeLatitude = float(lineFields[3])
                        nodeLongitude = float(lineFields[4])
                        self.nodes[nodeCode] = coordinates(nodeLatitude, nodeLongitude, True)
        with open('fileGeography.csv', 'w') as fileGeo:
            for elt in self.nodes.keys():
                fileGeo.write(elt + ';' + str(self.getPosition(elt)) + '\n')
        print('Geography built with ' + str(len(self.nodes.keys())) + ' nodes.')

    def getPosition(self, nodeName):
        maxLatitude = max([coordinate.latitude for coordinate in self.nodes.values()])
        minLatitude = min([coordinate.latitude for coordinate in self.nodes.values()])
        maxLongitude = max([coordinate.longitude for coordinate in self.nodes.values()])
        minLongitude = min([coordinate.longitude for coordinate in self.nodes.values()])
        if nodeName in self.nodes.keys():
            nodeLatitude = self.nodes[nodeName].latitude
            nodeLongitude = self.nodes[nodeName].longitude
            relativeLatitude = (nodeLatitude-minLatitude)/(maxLatitude-minLatitude)
            relativeLongitude = (nodeLongitude-minLongitude)/(maxLongitude-minLongitude)
            return coordinates(relativeLongitude, relativeLatitude, True)
        else:
            return coordinates(.5, .5, False)

    def isInGeography(self, nodeName):
        return (nodeName in self.nodes.keys())

class coordinates:
    def __init__(self, longitude, latitude, fixed):
        self.longitude = longitude
        self.latitude = latitude
        self.fixed = fixed

    def __str__(self):
        if self.fixed:
            return str(self.longitude) + ';' + str(self.latitude) + '!'
        else:
            return str(self.longitude) + ';' + str(self.latitude)

def matrixReduction(setHor, setVer, arrayToReduce):
    """
    This function returns the values from arrayToReduce for
    -columns from setHor
    -lines from setVer
    in a new array.
    """
    listTemp = []
    for i in range(len(setVer)):
        listTemp.append(arrayToReduce[setVer[i].index,:])
    arrayTemp = numpy.array(listTemp)
    listTemp = []
    for i in range(len(setHor)):
        listTemp.append(arrayTemp[:,setHor[i].index])
    result = numpy.transpose(numpy.array(listTemp))

    return result
  
#Function defined to avoid computations of N-k-k, required for computation on GPU.
def excludeAB(setA, setB):
    results = -1 * numpy.ones(len(setA), dtype = numpy.int32)
    for i in range(len(setA)):
        try:
            results[i] = setB.index(setA[i])
        except:
            pass
    return results

#@jit('void(float64[:,:], float64[:,:], int32)')
def compareMatrix(matA, matB, result):
    result = 0
    for (i,j) in numpy.ndindex(matA.shape):
        if matA[i,j] != matB[i,j]: 
            result +=1
            print(str(i) + "," + str(j))


#Function defined to compute N-2 IF on CPU
@jit('void(int32[:], float64[:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], int32[:,:], float64[:,:], int32[:], int32[:], int32[:], float64[:], float64[:], int32[:], float64[:,:])')
def computeIFCPU(setsSize, vectorKii, vectorKrr, matrixKir, matrixKit, matrixKri, matrixKrt, resultT, resultIF, excludeIR, excludeRT, excludeTI, matrixNrt, resultNIF, resultNT, resultNIFNN):
    epsilon = 0.00001
    for (r, i) in numpy.ndindex((setsSize[0], setsSize[1])):
        Kir = matrixKir[r, i]
        Kri = matrixKri[i, r]
        Kii = vectorKii[i]
        Krr = vectorKrr[r]
        denominator = (1 - Kii) * (1 - Krr) - Kir * Kri
        if abs(denominator) > epsilon:
            for t in range(setsSize[2]):
                if excludeIR[i] != r and excludeRT[r] != t and excludeTI[t] != i:
                    """
                    values = str(i) + ',' + str(r) + ',' + str(t)
                    if setI[i].index == setR[r].index:
                        print('error for values ' + values + ' element i = element r')
                    if setI[i].index == setT[t].index:
                        print('error for values ' + values + ' element i = element t')
                    if setR[r].index == setT[t].index:
                            print('error for values ' + values + ' element r = element t')
                    """
                    Kit = matrixKit[t,i]
                    Krt = matrixKrt[t,r]
                    Nrt = matrixNrt[t, r]
                    numerator = Kit*Kri + (1- Kii) * Krt
                    IF = numerator / denominator
                    if abs(IF) > resultIF[i,r]:
                        resultIF[i,r] = abs(IF)
                        resultT[i,r] = t
                    NIF = Nrt * abs(IF)
                    if NIF > resultNIF[i,r]:
                        resultNIF[i, r] = NIF
                        resultNIFNN[i,r] = abs(IF)
                        resultNT[i, r] = t

#Function defined to get IF, t and i from 2-D matrices previously computed (CPU compiled)
@jit('void(int32[:,:], float64[:,:], int32[:,:], float64[:,:], float64[:,:], int32[:], int32[:], int32[:], int32[:], float64[:], float64[:], float64[:])')
def getResults(resultsT, resultsIF, resultsNT, resultsNIF, resultsNIFNN, finalResultsT, finalResultsNT, finalResultsI, finalResultsNI, finalResultsIF, finalResultsNIF, finalResultsNIFNN):
    for (i,r) in numpy.ndindex(resultsT.shape):
        if resultsIF[i, r] > finalResultsIF[r]:
            finalResultsIF[r] = resultsIF[i, r]
            finalResultsT[r] = resultsT[i, r]
            finalResultsI[r] = i
        if resultsNIF[i, r] > finalResultsNIF[r]:
            finalResultsNIF[r] = resultsNIF[i, r]
            finalResultsNT[r] = resultsNT[i, r]
            finalResultsNI[r] = i
            finalResultsNIFNN[r] = resultsNIFNN[i, r]

#Function defined to build the normalization matrix
#@jit('float64[:](int32[:])')
def buildNormMatrix(PATL):
    sizeP = len(PATL)
    matrixTemp = []
    for i in range(sizeP):
        if PATL[i] > 0:
            matrixTemp.append(PATL/PATL[i])
        else:
            matrixTemp.append(numpy.array([1.0] * sizeP))
    return numpy.array(matrixTemp)

def buildNormGenerators(PATL, GenPower):
    sizeP = len(PATL)
    matrixTemp = []
    for i in range(sizeP):
        if PATL[i] > 0:
            matrixTemp.append(GenPower/PATL[i])
        else:
            matrixTemp.append(GenPower*0)
    return numpy.array(matrixTemp)

#Function defined to read line elements from the .uct file
def readLines(fileRead):
    branches = []
    #looking for  ##L line.
    i = 0
    while i < len(fileRead) and fileRead[i] !="##L":
        i += 1
    if i<len(fileRead):
        ## line i is "##L"
        i += 1
        while i < len(fileRead) and fileRead[i][0:2] != "##":
            if int(fileRead[i][20]) < 2:
                nodeNameFrom = fileRead[i][0:8]
                nodeNameTo = fileRead[i][9:17]
                branchOrder = fileRead[i][18]
                impedance = float(fileRead[i][29:35])
                IATL = float(fileRead[i][45:51])
                branches.append(Branch(nodeNameFrom, nodeNameTo, branchOrder, impedance, IATL, "Line"))
            i += 1
    else:
        print("No line ##L was found in the UCT file")
        sys.exit()
    print("Lines read")
    return branches

#Function defined to read transformers elements from the .uct file
def readTransformers(fileRead, setOfElements):
    #looking for  ##T line.
    i = 0
    while i < len(fileRead) and fileRead[i] !="##T":
        i += 1
    if i<len(fileRead):
        ## line i is "##T"
        i += 1
        while i < len(fileRead) and fileRead[i][0:2] != "##":
            if int(fileRead[i][20]) < 2:
                nodeNameFrom = fileRead[i][0:8]
                nodeNameTo = fileRead[i][9:17]
                branchOrder = fileRead[i][18]
                impedance = float(fileRead[i][47:53])
                IATL = float(fileRead[i][70:76])
                setOfElements.append(Branch(nodeNameFrom, nodeNameTo, branchOrder, impedance, IATL, "Transformer"))
            i += 1
    else:
        print("No line ##T was found in the UCT file")
        sys.exit()
    print("Transformers read")

def readGenerators(fileRead):
    fileLog.write("Reading generators" + '\n')
    generators = []
    #looking for  ##N line.
    i = 0
    while i < len(fileRead) and fileRead[i] !="##N":
        i += 1
    if i<len(fileRead):
        i+=1
        while i < len(fileRead) and (fileRead[i][0:2] != "##" or fileRead[i][0:3] == "##Z"):
            if len(fileRead[i]) > 80:
                nodeName = fileRead[i][0:8]
                try:
                    generatorPower = float(fileRead[i][73:80])
                    if generatorPower >= 0.0:
                        fileLog.write("     Generator " + nodeName + " has negative or zero maximum generation power" + '\n')
                    else:
                        generators.append(GenerationUnit(nodeName, -generatorPower))
                except:
                    fileLog.write("     Generator " + nodeName + " maximum permissible generation could not be read." + '\n')
            i+=1
    else:
        print("No line ##N was found in the UCT file")
        sys.exit()
    print("Generators read")
    return generators

#Function defined to read couplers
def readCouplers(fileRead, setOfElements):
    """

    :param fileRead: UCT file open
    :param setOfElements:
    :return: nothing
    """

    #looking for  ##L line.

    i = 0
    while i < len(fileRead) and fileRead[i] !="##L":
        i += 1
    if i<len(fileRead):
        ## line i is "##L"
        i += 1
        while i < len(fileRead) and fileRead[i][0:2] != "##":
            if int(fileRead[i][20]) == 2:
                if pMergeCouplers:
                    nodeNameFrom = fileRead[i][0:8]
                    nodeNameTo = fileRead[i][9:17]
                    if nodeNameFrom in dictCouplers.values():
                        for key in [key for key in dictCouplers if dictCouplers[key] == nodeNameFrom]:
                            dictCouplers[key] = nodeNameTo
                    if nodeNameFrom in dictCouplers.keys():
                        if nodeNameFrom in dictCouplers.values():
                            print("Erreur with coupler " + nodeNameFrom + " -> " + nodeNameTo)
                        else:
                            dictCouplers[nodeNameTo] = dictCouplers[nodeNameFrom]
                    else:
                        dictCouplers[nodeNameFrom] = nodeNameTo
                else:
                    nodeNameFrom = fileRead[i][0:8]
                    nodeNameTo = fileRead[i][9:17]
                    branchOrder = fileRead[i][18]
                    impedance = 0.03
                    IATL = 0.0
                    branches.append(Branch(nodeNameFrom, nodeNameTo, branchOrder, impedance, IATL, "Line"))
            i += 1
    if pMergeCouplers:
        for i in range(len(setOfElements)):
            setOfElements[i].applyCouplers(dictCouplers)

def removeLoopElements(setOfElements):
    eltToRemove = []
    for elt in setOfElements:
        if elt.nameFrom == elt.nameTo: eltToRemove.append(elt)
    for elt in eltToRemove:
        setOfElements.remove(elt)
    for i in range(len(setOfElements)):
        setOfElements[i].index = i

def attachGenerators(setOfNodes, setOfElements):
    fileLog.write("Attaching generators" + '\n')
    for elt in setOfElements:
        if elt.nodeName in dictCouplers.keys():
            elt.nodeName = dictCouplers[elt.nodeName]
        attachedNode = [node for node in setOfNodes if node.name == elt.nodeName]
        if len(attachedNode) == 1:
            elt.node = attachedNode[0]
            elt.country = attachedNode[0].country
            elt.connected = True
            attachedNode[0].generators.append(elt)
        else:
            fileLog.write("     Generator " + elt.name + " could not be attached to node " + elt.nodeName + " : " + str(len(attachedNode)) + " matches found." + '\n')
    generatorsToRemove = []
    for elt in setOfElements:
        if elt.node == None: generatorsToRemove.append(elt)
    for elt in generatorsToRemove:
        setOfElements.remove(elt)
    for i in range(len(setOfElements)):
        setOfElements[i].index = i
    print(str(len(setOfElements)) + " generators are in operation in the system. " + str(len(generatorsToRemove)) + " generators are out of operation.")

#Function defined to merge 3-windings transformers in a 2-windings one.
def mergeEquivalents(setOfElements, countryCode):
    fileLog.write("Merging 3-windings transformers for country " + countryCode + '\n')
    dictNodes = {}
    for elt in setOfElements:
        try:
            dictNodes[elt.nameFrom].append([elt, True])
        except:
            dictNodes[elt.nameFrom]=[[elt, True]]
        try:
            dictNodes[elt.nameTo].append([elt, False])
        except:
            dictNodes[elt.nameTo]=[[elt, False]]
    eltToMerge = [elt for elt in branches if elt.impedance < 0 and elt.nameFrom[0:len(countryCode)] == countryCode and elt.nameTo[0:len(countryCode)] == countryCode]
    eltToRemove = []
    print(str(len(eltToMerge)) + " transformers to merge in control area " + countryCode)
    for eltEq in eltToMerge:
        for (nodeEq, nodeReal) in [(eltEq.nameFrom, eltEq.nameTo), (eltEq.nameTo, eltEq.nameFrom)]:
            if int(nodeEq[6:7]) < int(nodeReal[6:7]):
                eltReals = [elt for elt in dictNodes[nodeEq] if elt[0].index != eltEq.index]
                if len(eltReals) == 0 :
                    print(eltEq.nameBranch + " has negative impedance but there are " + str(len(eltReals)) + " real transformers found.")
                else:
                    for elt in eltReals:
                        eltReal = elt[0]
                        if eltReal.nameFrom[6:7] == nodeEq[6:7] and not elt[1] and eltReal.PATL > 0:
                            fileLog.write("     " + eltEq.nameBranch + " merged with " + eltReal.nameBranch + '\n')
                            eltReal.nameTo = nodeReal
                            eltReal.impedance += eltEq.impedance
                            eltToRemove.append(eltEq)
                        elif eltReal.nameTo[6:7] == nodeEq[6:7] and elt[1] and eltReal.PATL > 0:
                            fileLog.write("     " + eltEq.nameBranch + " merged with " + eltReal.nameBranch + '\n')
                            eltReal.nameFrom = nodeReal
                            eltReal.impedance += eltEq.impedance
                            eltToRemove.append(eltEq)
                        else:
                            eltToRemove.append(eltReal)              
    for elt in eltToRemove:
        try:
            setOfElements.remove(elt)
        except ValueError:
            fileLog.write(elt.nameBranch + " could not be removed from the list." + '\n')
    for i in range(len(setOfElements)):
        setOfElements[i].index = i
    eltToMerge = [elt for elt in branches if elt.impedance < 0 and elt.nameFrom[0:len(countryCode)] == countryCode and elt.nameTo[0:len(countryCode)] == countryCode]
    print("Negative impedance transformers merged in control area " + countryCode + ": " + str(len(eltToMerge)) + " such elements remaining")

#Function defined to build the list of nodes from the list of branches
def determineNodes(setOfElements):
    nodes = []
    dictNodes = {}
    for i in range(len(setOfElements)):
        nodeFrom = setOfElements[i].nameFrom
        if nodeFrom in dictNodes.keys():
            setOfElements[i].nodeFrom = dictNodes[nodeFrom]
            dictNodes[nodeFrom].branches.append(setOfElements[i])
        else:
            newNode = Node(nodeFrom)
            newNode.branches.append(setOfElements[i])
            nodes.append(newNode)
            dictNodes[newNode.name] = newNode
            setOfElements[i].nodeFrom = newNode    
        
        nodeTo = setOfElements[i].nameTo
        if nodeTo in dictNodes.keys():
            setOfElements[i].nodeTo = dictNodes[nodeTo]
            dictNodes[nodeTo].branches.append(setOfElements[i])
        else:
            newNode = Node(nodeTo)
            newNode.branches.append(setOfElements[i])
            nodes.append(newNode)
            dictNodes[newNode.name] = newNode
            setOfElements[i].nodeTo = newNode

    for elt in setOfElements:
        if elt.nodeFrom == None or elt.nodeTo == None:
            print(elt.nameBranch + " has no nodes declared")
        else:
            elt.setCountry()
    print("List of nodes built")

    return nodes

def mergeTieLines(setOfElements, setOfNodes):
    fileLog.write("Merging tie-lines" + '\n')
    for nodeToMerge in [node for node in setOfNodes if node.isXNode()]:
        if len(nodeToMerge.branches) != 2:
            fileLog.write("     X-node " + nodeToMerge.name + " could not be merged : incorrect number of branches connected, " + str(len(nodeToMerge.branches)) + '\n')
            if len(nodeToMerge.branches) == 1:
                nodeToMerge.remove(setOfElements, setOfNodes)
                fileLog.write("     Node " + nodeToMerge.name + " and branch " + nodeToMerge.branches[0].nameBranch + " removed." + '\n')
        else:
            branchA = nodeToMerge.branches[0]
            branchB = nodeToMerge.branches[1]
            nodesA = [node for node in [branchA.nodeFrom, branchA.nodeTo] if node != nodeToMerge]
            nodesB = [node for node in [branchB.nodeFrom, branchB.nodeTo] if node != nodeToMerge]
            if len(nodesA) != 1: 
                fileLog.write("     Error while merging " + nodeToMerge.name + " : incorrect number of nodes for branch " + branchA.nameBranch + '\n')
                continue
            else:
                nodeA = nodesA[0]
            if len(nodesB) != 1: 
                fileLog.write("     Error while merging " + nodeToMerge.name + " : incorrect number of nodes for branch " + branchB.nameBranch + '\n')
                continue
            else:
                nodeB = nodesB[0]
            orderA = branchA.nameBranch[18]
            orderB = branchB.nameBranch[18]
            if orderA != orderB:
                fileLog.write("     Error while merging " + nodeToMerge.name + " : order could not be determined for lines " + branchA.nameBranch + " and " + branchB.nameBranch + '\n')
                mergedOrder = "X"
            else:
                mergedOrder = orderA
            
            mergedPATL = min(branchA.PATL, branchB.PATL)
            mergedImpedance = branchA.impedance + branchB.impedance
            mergedBranch = Branch(nodeA.name, nodeB.name, mergedOrder, 1, 0, "Line")
            mergedBranch.impedance = mergedImpedance
            setOfElements.append(mergedBranch)
            mergedBranch.PATL = mergedPATL
            mergedBranch.nodeFrom = nodeA
            mergedBranch.nodeTo = nodeB
            mergedBranch.tieLine = True
            nodeA.branches.append(mergedBranch)
            nodeB.branches.append(mergedBranch)
            nodeToMerge.remove(setOfElements, setOfNodes)
            fileLog.write("     X-node " + nodeToMerge.name + " merged : " + branchA.nameBranch + " and " + branchB.nameBranch + " merged in " + mergedBranch.nameBranch + '\n')
            fileLog.write("     Branch A impedance : " + str(branchA.impedance) +  ", branch B impedance : " + str(branchB.impedance) + ", merged branch impedance : " + str(mergedBranch.impedance) + '\n')
    print("X-nodes remaining in system : " + str(len([node for node in setOfNodes if node.isXNode()])))
         
#Function defined to set nodes from the investigated control area as ring 0
def initializeRingAndConnection(setOfNodes, countryCode):
    #Finding the most connex node in country
    maxConnection = max([len(elt.branches) for elt in setOfNodes if elt.name[0:len(countryCode)] == countryCode])
    mostConnectedNode = [elt for elt in setOfNodes if elt.name[0:len(countryCode)] == countryCode and len(elt.branches) == maxConnection][0]
    print("most connected node in " + countryCode + " is " + mostConnectedNode.name)
    #Setting nodes from the country in 0-ring, starting from the most connex node
    mostConnectedNode.connected = True
    connectionSteps = 0
    connectableBranches = [branch for branch in branches if branch.nodeTo.connected != branch.nodeFrom.connected]
    while len(connectableBranches) > 0:
        connectionSteps += 1
        for branch in connectableBranches:
            branch.nodeTo.connected = True
            branch.nodeFrom.connected = True
        connectableBranches = [branch for branch in branches if branch.nodeTo.connected != branch.nodeFrom.connected]
    print("Connectivity established in " + str(connectionSteps) + " steps.")
    mostConnectedNode.insertInCA()
    print("Ring 0 initialised with " + str(len([node for node in setOfNodes if node.ring == 0])) + " nodes.")
    #Listing nodes which are not connex to the assessed control area
    for node in [node for node in setOfNodes if node.name[0:len(countryCode)] == countryCode and node.ring == 99]:        
        print("Warning : " + node.name + " is not connected to " + countryCode)
    print("Initial connectivity initialised")
    #Listing nodes which are not connex to the assessed control area
    print("Among " + str(len(setOfNodes)) + " nodes, " + str(len([elt for elt in nodes if elt.connected == False])) + " nodes are not connected to " + countryCode) 
    #Checking consistency
    for elt in [node.name for node in setOfNodes if node.ring == 0 and not node.connected]:
        print("Node " + elt + " is in " + countryCode + " but is not connected")

#Function defined to determine rings
def determineRings(setOfNodes):
    dictNodes = {}
    for node in [node for node in setOfNodes if node.connected]:
        equivalentNodeName = node.getEquivalentNodeName()
        if equivalentNodeName in dictNodes.keys():
            dictNodes[equivalentNodeName].append(node)
        else:
            dictNodes[equivalentNodeName] = [node]

    currentRing = 0
    nodesInRing = [node for node in setOfNodes if node.ring == currentRing]
    while len(nodesInRing) >0:
        for node in nodesInRing:
            for branch in node.branches:
                branch.increaseRing(currentRing, dictNodes)
        currentRing +=1
        nodesInRing = [node for node in setOfNodes if node.ring == currentRing]
    #Checking consistency
    for elt in [node.name for node in setOfNodes if node.ring == 99 and node.connected]:
        print("Node " + elt + " is connected but has no ring")
    for elt in [node.name for node in setOfNodes if node.ring < 99 and not node.connected]:
        print("Node " + elt + " is in a ring but is not connected")
    print("Rings determined. Maximum ring is #" + str(currentRing - 1) + ".")
    
#Function defined to determine inverse inner-rings (starting from 0 at X-nodes)
def determineInverseRings(setOfNodes):
    controlArea = [node for node in setOfNodes if node.ring == 0]
    for node in [node for node in controlArea if node.isBorder() == False]:
        node.ring = 99
    determineRings(controlArea)
    branchesArea = [branch for branch in branches if branch.nodeTo in controlArea and branch.nodeFrom in controlArea]
    for branch in branchesArea:
        branch.updateRing
    #Storing internal branches
    with open("Branches-Internal-" + countryCode + ".csv", "w") as file:
        file.write(Branch.header() + '\n')
        for elt in branchesArea:
            file.write(str(elt) + '\n')
    with open("Nodes-Internal-" + countryCode + ".csv", "w") as file:
        file.write(Node.header() + '\n')
        for elt in controlArea:
            file.write(str(elt) + '\n')
    #storing external branches
    with open("Branches-External-" + countryCode + ".csv", "w") as file:
        file.write(Branch.header() + '\n')
        for elt in [branch for branch in branches if branch not in branchesArea]:
            file.write(str(elt) + '\n')
    with open("Nodes-External-" + countryCode + ".csv", "w") as file:
        file.write(Node.header() + '\n')
        for elt in [node for node in nodes if node not in controlArea]:
            file.write(str(elt) + '\n')
    for node in controlArea:
        node.ring = 0
    for branch in branchesArea:
        branch.updateRing()

def mainComponentRestriction(setOfNodes, setOfElements):
    connexNodes = []
    connexBranches = []
    for node in setOfNodes:
        if node.connected:
            connexNodes.append(node)
            for branch in node.branches:
                if not branch in connexBranches:
                    connexBranches.append(branch)
    #checking consistency
    for node in connexNodes:
        if not node.connected:
            print("Node " + node.name + " should not be in main component")
    for branch in connexBranches:
        if not branch.nodeFrom.connected:
            print("Branch " + branch.nameBranch + " should not be in main component")
        if not branch.nodeTo.connected:
            print("Branch " + branch.nameBranch + " should not be in main component")
    #Rebuilding index
    for i in range(len(connexNodes)):
        connexNodes[i].index = i
    for i in range(len(connexBranches)):
        connexBranches[i].index = i
    print("System restricted to main connected component with " + str(len(connexNodes)) + " nodes and " + str(len(connexBranches)) + " elements")
    return connexNodes, connexBranches

def computeISF(setOfNodes, setOfElements):
    t1 = time.clock()
    #selecting slack node
    maxConnection = max([len(node.branches) for node in setOfNodes if node.name[0:len(countryCode)]== countryCode and node.name[6:7] == "1"])
    slackNode = [node for node in setOfNodes if len(node.branches) == maxConnection and node.name[0:len(countryCode)] == countryCode][0]
    print("Slack node for the system is " + slackNode.name)
    #Susceptance matrix construction
    sizeN = len(setOfNodes)
    matrixB = numpy.zeros((sizeN, sizeN))
    for elt in setOfElements:
        i = elt.nodeFrom.index
        j = elt.nodeTo.index
        matrixB[i,i] += -1 / elt.impedance
        matrixB[j,j] += -1 / elt.impedance
        matrixB[i,j] += 1 / elt.impedance
        matrixB[j,i] += 1 / elt.impedance
    matrixB = numpy.delete(matrixB, slackNode.index, axis = 0)
    matrixB = numpy.delete(matrixB, slackNode.index, axis = 1)
    print("Susceptance matrix B built in " + str(round(time.clock() - t1, 2)) + " seconds.")
    #Susceptance matrix inversion
    t1 = time.clock()
    inverseB = numpy.linalg.inv(matrixB)
    print("Susceptance matrix B inverted in " + str(round(time.clock() - t1, 2)) + " seconds.")
    t1 = time.clock()
    #Injection Shift Factors computation
    ISFBis = []
    for elt in setOfElements:
        i = elt.nodeFrom.index
        j = elt.nodeTo.index
            
        if i < slackNode.index:
            BFrom = inverseB[i,:]
        elif i > slackNode.index:
            BFrom = inverseB[i-1,:]
        else: 
            BFrom = numpy.zeros(sizeN-1)

        if j < slackNode.index:
            BTo = inverseB[j,:]
        elif j > slackNode.index:
            BTo = inverseB[j-1,:]
        else: 
            BTo = numpy.zeros(sizeN-1)
            
        ISFBis.append(-1/elt.impedance * numpy.array((BFrom-BTo)))
    matrixISF = numpy.array(ISFBis)
    matrixISF = numpy.insert(matrixISF, slackNode.index, 0, axis = 1)
    print("ISF matrix computed in " + str(round(time.clock() - t1,1)) + " seconds.")
    return matrixISF

def computeLODF(setOfElements, PTDF):
    """
    This function computes a LODF matrix from a PTDF matrix. It is assumed that the PTDF matrix is order provided by the
    'index' property of each branch.
    :param setOfElements:  a list on n branches
    :param PTDF: a square matrix of size n*n of Power Transfer Distribution Factors
    :return: a square matrix of size n*n of Line Outage Distribution Factors.
    """
    LODF = []
    for elt in setOfElements:
        if elt.PTDF < 1 - epsilon:
            column = numpy.array(PTDF[:,elt.index] / ( 1 - elt.PTDF))
            column[elt.index] = 0.0
        else:
            column = numpy.zeros(PTDF.shape[0])
        LODF.append(column)
    arrayLODF = numpy.transpose(numpy.array(LODF))
    return arrayLODF

def computeLODFg(setOfElements, ISF):
    fileLog.write("computing IF for SGU")
    LODF = []
    for elt in setOfElements:
        balancingGenerators = [eltGenerator for eltGenerator in generators if eltGenerator.country == elt.country and 
                               eltGenerator != elt]
        if len(balancingGenerators) == 0:
            fileLog.write("No generators found to balance the contingency of " + elt.nameBranch)
            column = numpy.zeros(ISF.shape[0])
        else:
             balancingPower = sum([elt.power for elt in balancingGenerators])
             column = numpy.zeros(ISF.shape[0])
             for eltGenerator in balancingGenerators:
                    generatorPTDF = ISF[:,eltGenerator.node.index] - ISF[:, elt.node.index]
                    column += eltGenerator.power / balancingPower * (numpy.array(generatorPTDF))
        LODF.append(column)
    arrayLODFg = numpy.transpose(numpy.array(LODF))
    arrayLODFgn = arrayLODFg * arrayNormg
    return arrayLODFg, arrayLODFgn

def computePTDF(setOfElements, ISF):
    t1 = time.clock()
    PTDF = []
    for elt in setOfElements:
        column = numpy.array((ISF[:, elt.nodeFrom.index] - ISF[:, elt.nodeTo.index]))
        PTDF.append(column)
    arrayPTDF = numpy.transpose(numpy.array(PTDF))
    #PTDF computation for elements
    for elt in setOfElements:
        elt.PTDF = arrayPTDF[elt.index, elt.index]
        if elt.PTDF<-epsilon:
            print(elt.nameBranch + "has negative self-PTDF :" + str(elt.PTDF))
        if elt.PTDF>1+epsilon:
            print(elt.nameBranch + "has self-PTDF higher than 1 :" + str(elt.PTDF))
    print("PTDF computed in " + str(round(time.clock() - t1, 1)) + " seconds.")
    return arrayPTDF

def determineIext(setOfElements, setT, inputLODF, normalisationMatrix):
    setIext = []
    currentRing = 1
    setJ = [elt for elt in setOfElements if elt.ring == currentRing]
    fileI = open("setIext-" + countryCode + ".csv", "w")
    fileI.write("External contingencies determinated with mode " + str(modeI) + " with non-normalized thresold " + str(setITh) + " (mode1, 3, 4) and normalized threshold " + str(setINTh) + " (mode 2, 3, 4)" + '\n')
    fileI.write("Element,Ring,self-PTDF,max IF (non-normalized),max IF (normalized)" + '\n')
    while len(setJ) > 0:
        LODF = numpy.absolute(matrixReduction(setJ, setT, inputLODF))
        LODFn = LODF * matrixReduction(setJ, setT, normalisationMatrix)
        for i in range(len(setJ)):
            eltI = setJ[i]
            #Include any element from the ring
            if modeI < 2:
                setIext.append(eltI)
                fileI.write(eltI.nameBranch + "," + str(eltI.ring) + "," + str(eltI.PTDF) + "," + str(numpy.amax(LODF[:,i])) + "," + str(numpy.amax(LODFn[:,i])) + '\n')
            elif modeI == 2 and numpy.amax(LODF[:,i]) > setITh:
                setIext.append(eltI)
                fileI.write(eltI.nameBranch + "," + str(eltI.ring) + "," + str(eltI.PTDF) + "," + str(numpy.amax(LODF[:,i])) + "," 
                            + str(numpy.amax(LODFn[:,i])) + "," + str([setT[k].nameBranch for k in range(len(setT)) if LODF[k,i] == numpy.amax(LODF[:,i])]) + '\n')
            elif modeI == 3 and numpy.amax(LODFn[:,i]) > setINTh:
                setIext.append(setJ[i])
                fileI.write(setJ[i].nameBranch + "," + str(setJ[i].ring) + "," + str(setJ[i].PTDF) + "," + str(numpy.amax(LODF[:,i])) + "," + str(numpy.amax(LODFn[:,i])) + '\n')
            elif modeI == 4 and (numpy.amax(LODF[:,i]) > setITh or numpy.amax(LODFn[:,i]) > setINTh):
                setIext.append(setJ[i])
                fileI.write(setJ[i].nameBranch + "," + str(setJ[i].ring) + "," + str(setJ[i].PTDF) + "," + str(numpy.amax(LODF[:,i])) + "," + str(numpy.amax(LODFn[:,i])) + '\n')
            elif modeI == 5 and numpy.amax(LODF[:,i]) > setITh and numpy.amax(LODFn[:,i]) > setINTh:
                setIext.append(setJ[i])
                fileI.write(setJ[i].nameBranch + "," + str(setJ[i].ring) + "," + str(setJ[i].PTDF) + "," + str(numpy.amax(LODF[:,i])) + "," + str(numpy.amax(LODFn[:,i])) + '\n')
                """
        if numpy.amax(LODF) > setITh and modeI >0 or modeI == 0:
            currentRing += 1
            setJ = [elt for elt in setOfElements if elt.ring == currentRing]
        else:
            setJ = []
            """
        currentRing += 1
        setJ = [elt for elt in setOfElements if elt.ring == currentRing]
    fileI.close()
    print("External contingencies determinated : " + str(len(setIext)) + " elements selected within maximum ring #" + str(currentRing))
    return setIext

def determineT1(setT, setIext, inputLODF, normalisationMatrix):
    setT1 = []
    LODF = numpy.absolute(matrixReduction(setIext, setT, inputLODF))
    if modeT == 0:
        setT1 = setT
    elif modeT == 1:
        for i in range(len(setIext)):
            for branch in [setT[j] for j in range(len(setT)) if LODF[j,i] == numpy.amax(LODF[:,i]) and not setT[j] in setT1]:
               setT1.append(branch)
    elif modeT == 2:
        for i in range(len(setIext)):
            for branch in [setT[j] for j in range(len(setT)) if LODF[j,i] > setITh and not setT[j] in setT1]:
               setT1.append(branch)
    print("Influenced Internal elements determinated : " + str(len(setT1)) + " elements selected.")
    return setT1

def determineIint(setT1, setT, inputLODF, normalisationMatrix):
    """
    if modeI > 0:
        setIint = []
        LODF = numpy.absolute(matrixReduction(setT, setT1, inputLODF))
        LODFn = numpy.absolute(matrixReduction(setT, setT1, inputLODFn))
        for i in range(len(setT)):
            if numpy.amax(LODF[:,i]) > setJTh:
                setIint.append(setT[i])
    else:
    """
    setIint = list(setT)
    print("Internal contingencies determinated : " + str(len(setIint)) + " elements selected.")
    return setIint

def excludeRadialI(setJ):
    result = []
    for eltI in setJ:
        if eltI.PTDF > 1 - epsilon:
            pass
        else:
            result.append(eltI)
    print("Radial elements which do not lead to disconnection of a generetor are excluded : " + str(len(result)) + "/" + str(len(setJ)) + " kept.")
    return result

def computeIF(inputLODF, normalizationMatrix):
    results = []

    sizeI = len(setI)
    sizeT = len(setT)

    with open("setI-" + countryCode + ".csv", "w") as fileOut:
        for elt in setI:
            fileOut.write(str(elt) + '\n')
    with open("setT-" + countryCode + ".csv", "w") as fileOut:
        for elt in setT:
            fileOut.write(str(elt) + '\n')

    print("Internal elements monitored : " + str(sizeT))
    print("Contingencies : " + str(sizeI))
    vectorKii = [elt.PTDF for elt in setI]
    matrixKit = matrixReduction(setI, setT, arrayPTDF)
    excludeTI = excludeAB(setT, setI)

    currentRing = 1
    setR = excludeRadialI([elt for elt in branches if elt.ring == currentRing])
    #setR = excludeRadialI([elt for elt in branches if elt.ring >0])
    while len(setR)>0:
        sizeR = len(setR)
        setsSize = numpy.array([sizeR,sizeI,sizeT], dtype=numpy.int32)
        print("Assessing IF for ring #" + str(currentRing) + " with " + str(sizeR) + " elements.")
        LODF = matrixReduction(setR, setT, inputLODF)
        LODFn = LODF * matrixReduction(setR, setT, normalizationMatrix)
        vectorKrr = [elt.PTDF for elt in setR]
        matrixKir = matrixReduction(setI, setR, arrayPTDF)
        matrixKri = matrixReduction(setR, setI, arrayPTDF)
        matrixKrt = matrixReduction(setR, setT, arrayPTDF)
        
        matrixNrt = matrixReduction(setR, setT, normalizationMatrix)

        resultsT = numpy.zeros((sizeI, sizeR), dtype = numpy.int32) #2D-Matrix of most influenced t element in N-i-r situation
        resultsNT = numpy.zeros((sizeI, sizeR), dtype = numpy.int32) #2D-Matrix of most normalized influenced t element in N-i-r situation
        resultsIF = numpy.zeros((sizeI, sizeR)) #2D-Matrix of influence factor on the most influenced t element in N-i-r situation
        resultsNIF = numpy.zeros((sizeI, sizeR)) #2D-Matrix of normalized influence factor on the most normalized influenced t element in N-i-r situation
        resultsNIFNN = numpy.zeros((sizeI, sizeR)) #2D-Matrix of non-normalized influence factor on the most normalized influenced t element in N-i-r situation
        excludeIR = excludeAB(setI, setR) #coordinates of elements i in R set to avoid i = r situation
        excludeRT = excludeAB(setR, setT) #coordinates of elements r in T set to avoid r = t situation
        finalResultsT = numpy.zeros(sizeR, dtype = numpy.int32) #1D-Vector of most influenced t element
        finalResultsNT = numpy.zeros(sizeR, dtype = numpy.int32) #1D-Vector of most normalized influenced t element
        finalResultsI = numpy.zeros(sizeR, dtype = numpy.int32)
        finalResultsNI = numpy.zeros(sizeR, dtype = numpy.int32)
        finalResultsIF = numpy.zeros(sizeR)
        finalResultsNIF = numpy.zeros(sizeR)
        finalResultsNIFNN = numpy.zeros(sizeR)
        computeIFCPU(setsSize, vectorKii, vectorKrr, matrixKir, matrixKit, matrixKri, matrixKrt, resultsT, resultsIF,
                     excludeIR, excludeRT, excludeTI, matrixNrt, resultsNIF, resultsNT, resultsNIFNN)

        getResults(resultsT, resultsIF, resultsNT, resultsNIF, resultsNIFNN, finalResultsT, finalResultsNT, finalResultsI, finalResultsNI, finalResultsIF, finalResultsNIF, finalResultsNIFNN)
        for r in range(len(setR)):
            #Template : "name,N-1 IF, N-1 nIF,IF,i,t,nIF,i,t,NNnIF"
            eltR = setR[r]
            IFN1 = max(numpy.absolute(LODF[:,r]))
            nIFN1 = max(numpy.absolute(LODFn[:,r]))
            IFN2 = finalResultsIF[r]
            nIFN2 = finalResultsNIF[r]
            eltI = setI[finalResultsI[r]]
            eltT = setT[finalResultsT[r]]
            eltIn = setI[finalResultsNI[r]]
            eltTn = setT[finalResultsNT[r]]
            LODFit = inputLODF[eltTn.index, eltIn.index]
            LODFir = inputLODF[eltR.index, eltIn.index]
            results.append(resultIF(eltR, IFN1, nIFN1, IFN2, nIFN2, eltI, eltT, eltIn, eltTn, LODFit, LODFir))
        if modeR == 1 and numpy.amax(numpy.absolute(resultsIF)) > setRTh or modeR == 2 and numpy.amax(numpy.absolute(resultsNIF)) > setRTh or modeR == 0:
            currentRing += 1
            setR = [elt for elt in branches if elt.ring == currentRing]
        else:
            setR = []
    return results

def storeResults(results):
    with open("resultsIF-" + countryCode + ".csv", "w") as fileOut:
        #Writing results in French format
        fileOut.write(resultIF.header())
        for elt in results:
            fileOut.write(str(elt))         

def storeTopology(setOfElements, setOfNodes):
    with open("branches.csv", "w") as fileOut:
        fileOut.write(Branch.header() + '\n')
        for elt in setOfElements:
            fileOut.write(str(elt) + '\n')
    with open("nodes.csv", "w") as fileOut:
        fileOut.write(Node.header() + '\n')
        for elt in setOfNodes:
            fileOut.write(str(elt) + '\n')

def storeN1(setR, setT, inputLODF, inputLODFn):
    LODF = numpy.absolute(matrixReduction(setR, setT, inputLODF))
    LODFn = numpy.absolute(matrixReduction(setR, setT, inputLODFn))
    with open("N-1IF.csv", "w") as file:
        file.write("i;ring;IF;t;norm IF;t" + '\n')
        for i in range(len(setR)):
            eltR = setR[i]
            stringToWrite = eltR.nameBranch + ";" + str(eltR.ring)
            IF = numpy.amax(LODF[:,i])
            stringToWrite += ";" + str(IF).replace(".",",")
            if IF > 0.0:
                stringToWrite += ";" + str([setT[k].nameBranch for k in range(len(setT)) if LODF[k,i] == IF])
            else:
                stringToWrite += ";"
            IFn = numpy.amax(LODFn[:,i])
            stringToWrite += ";" + str(IFn).replace(".",",")
            if IFn > 0.0:
                stringToWrite += ";" + str([setT[k].nameBranch for k in range(len(setT)) if LODFn[k,i] == IFn])
            else:
                stringToWrite += ";"
            file.write(stringToWrite + '\n')

def establishOA(results):
    #Building mode for OA : 0, all the external elements; 1, all elements with N-2 IF >= setOATh; 
    #2, all elements with normalized N-2 IF >= setOAnTh; #3, all elements with N-2 IF >= setOATh OR normalized N-2 IF >= setOAnTh; 
    #4, all elements with N-2 IF >= setOATh AND normalized N-2 IF > setOAnTh
    if modeOA == 0:
        setOA = [elt.eltR for elt in results]
    elif modeOA == 1:        
        setOA = [elt.eltR for elt in results if elt.IFN2 >= setOATh]
    elif modeOA == 2:
        setOA = [elt.eltR for elt in results if elt.nIFN2 >= setOAnTh]
    elif modeOA == 3:
        setOA = [elt.eltR for elt in results if elt.IFN2 >= setOATh or elt.nIFN2 >= setOAnTh]
    elif modeOA == 4:
        setOA = [elt.eltR for elt in results if elt.IFN2 >= setOATh and elt.nIFN2 >= setOAnTh]
    else:
        print("Incorrect parameter for modeOA")
        sys.exit()
    return setOA

def computeIFSGU():
    results = []
    tempLODF = []
    tempLODFn = []
    for elt in setT:
        tempLODF.append(LODFg[elt.index,:])
        tempLODFn.append(LODFgn[elt.index,:])
    rLODFg = numpy.array(tempLODF)
    rLODFgn = numpy.array(tempLODFn)
    rLODF = matrixReduction(setI,setT,LODF)
    rLODFn = rLODF * matrixReduction(setI,setT,arrayNorm)
    for r in range(len(setR)):
        eltR = setR[r]
        results.append([eltR.name,eltR.power,0.0,[],[],0.0,[],[]])
        for i in range(len(setI)):
            vectorLODF = rLODFg[:,r] + rLODF[:,i]*LODFg[setI[i].index,r]
            IF = numpy.max(numpy.abs(vectorLODF))
            if IF > results[r][2]:
                results[r][2] = IF
                results[r][3] = [setI[i].nameBranch]
                results[r][4] = [setT[k].nameBranch for k in range(len(setT)) if abs(vectorLODF[k]) == IF]
            elif IF == results[r][2]:
                results[r][3].append(setI[i].nameBranch)
                results[r][4].append([setT[k].nameBranch for k in range(len(setT)) if abs(vectorLODF[k]) == IF and not setT[k] in results[r][4]])
            vectorLODFn = rLODFgn[:,r] + rLODFn[:,i]*LODFgn[setI[i].index,r]
            IFn = numpy.max(numpy.abs(vectorLODFn))
            if IFn > results[r][5]:
                results[r][5] = IFn
                results[r][6] = [setI[i].nameBranch]
                results[r][7] = [setT[k].nameBranch for k in range(len(setT)) if abs(vectorLODFn[k]) == IFn]
            elif IFn == results[r][5]:
                results[r][6].append(setI[i].nameBranch)
                results[r][7].append([setT[k].nameBranch for k in range(len(setT)) if abs(vectorLODFn[k]) == IFn and not setT[k] in results[r][7]])
    return results

def storeResultsSGU(results):
    with open("resultsSGU-" + countryCode + ".csv", "w") as fileOut:
        fileOut.write("SGU;Power;IF;i;t;IFn;i;t" + '\n')
        for elt in resultsSGU:
            fileOut.write(elt[0] + ";")
            fileOut.write(str(elt[1]).replace(".",",") + ";")
            fileOut.write(str(round(elt[2],4)).replace(".",",") + ";")
            fileOut.write(str(elt[3]).replace("[","").replace("]","") + ";")
            fileOut.write(str(elt[4]).replace("[","").replace("]","") + ";")
            fileOut.write(str(round(elt[5],4)).replace(".",",") + ";")
            fileOut.write(str(elt[6]).replace("[","").replace("]","") + ";")
            fileOut.write(str(elt[7]).replace("[","").replace("]","") + '\n')


def drawOA(nodesOfOA):
    graphSize = (50, 80)

    # Initializing graph
    dot = Graph(comment='Observability areas', format='svg', engine='sfdp')
    dot.attr('graph', overlap='false')
    dot.attr('graph', label=fileUCT + '\n' + 'Normalized threshold : ' + str(
        setOAnTh) + '\n' + 'Not-Normalized threshold : ' + str(setOATh))
    dot.attr('graph', size=str(graphSize[0]) + ',' + str(graphSize[1]))
    # dot.attr('graph', splines='true')
    # dot.attr('graph', rotate='180')

    # Determining graph structures
    nodeEdges = {}
    graphEdges = []
    # for branch in [branch for branch in branches if (branch.nodeFrom.name[6] == '1' or branch.nodeFrom.name[6] == '2') and (branch.nodeTo.name[6] == '1' or branch.nodeTo.name[6] == '2')]:
    for branch in branches:
        for nodeName in [branch.nodeFrom.name[:7], branch.nodeTo.name[:7]]:
            if nodeName not in nodeEdges.keys() and nodeName[0] in ['F','E','P','B','N','S','D','O','I','L']:
                nodeEdges[nodeName] = 0

        for nodeName1, nodeName2 in [(branch.nodeFrom.name[:7], branch.nodeTo.name[:7]),
                                     (branch.nodeTo.name[:7], branch.nodeFrom.name[:7])]:
            if nodeName1 < nodeName2:
                if len([edge for edge in graphEdges if edge[0] == nodeName1 and edge[1] == nodeName2]) == 0 and \
                                nodeName1 in nodeEdges.keys() and nodeName2 in nodeEdges.keys():
                    graphEdges.append([nodeName1, nodeName2])
                    nodeEdges[nodeName1] += 1
                    nodeEdges[nodeName2] += 1

    currentOA = []
    nodesNotInUCT = []
    with open('currentOA.csv', 'r') as fileOA:
        for node in fileOA.read().split('\n'):
            shortNode = node[:7]
            if shortNode in nodeEdges.keys():
                currentOA.append(shortNode)
            else:
                nodesNotInUCT.append(shortNode)
    print('The following ' + str(len(nodesNotInUCT)) + ' are in the current OA of RTE but not in the .uct file' + '\n')
    print(str(nodesNotInUCT))

    # Determining nodes characteristics
    dictOAs = []
    # Translating inputs in node names
    for node in nodesOfOA:
        nodeName = node.name[:7]
        if not nodeName in dictOAs:
            dictOAs.append(nodeName)

    dictGeography = geography('graphNodes-Geography.csv')

    for node in sorted(nodeEdges.keys()):
        if nodeEdges[node] > 1:
            # Drawing Y-nodes
            if node[0] == "F" and node[1] == "Z":
                dot.node(node, shape='point')
            elif node[0] == "D" and node[2] == "Y":
                dot.node(node, shape='point')
            # Drawing real nodes
            else:
                nodeColor = getNodeColor(node, dictOAs, dictGeography, currentOA)
                nodePosition = getNodePosition(node, dictGeography, graphSize)
                nodeShape = getNodeShape(node, dictGeography)
                dot.node(node, shape=nodeShape, style='filled', color=nodeColor, pos=nodePosition)
                # Building mode for OA : 0, all the external elements; 1, all elements with N-2 IF >= setOATh;
                # 2, all elements with normalized N-2 IF >= setOAnTh; #3, all elements with N-2 IF >= setOATh OR normalized N-2 IF >= setOAnTh;
                # 4, all elements with N-2 IF >= setOATh AND normalized N-2 IF > setOAnTh

    for edge in [edge for edge in graphEdges if nodeEdges[edge[0]] > 1 and nodeEdges[edge[1]] > 1]:
        dot.edge(edge[0], edge[1])

    dot.render('test-output/OA-' + countryCode + '-' + str(setOAnTh) + '-' + str(setOATh))


def getNodePosition(nodeName, geography, size):
    maxX = size[0]
    maxY = size[1]
    nodePosition = geography.getPosition(nodeName)
    stringPosition = str(nodePosition.longitude * maxX) + ',' + str(nodePosition.latitude * maxY)
    if nodePosition.fixed:
        return stringPosition + '!'
    else:
        return stringPosition

"""
def getNodeColor(nodeName, externalOAs, geography):
    if nodeName in externalOAs[0]:
        return 'gray'
    # Other nodes
    # AND condition
    elif nodeName in externalOAs[4]:
        return 'skyblue'
    # Detected by not-normalized threshold but not by the normalized one
    elif nodeName in externalOAs[1] and not nodeName in externalOAs[2]:
        return 'forestgreen'
    # Detected by normalized threshold but not by the non-normalized one
    elif nodeName not in externalOAs[1] and nodeName in externalOAs[2]:
        return 'deeppink'
    elif nodeName in externalOAs[1] and nodeName in externalOAs[2] and not nodeName in externalOAs[3]:
        return 'orange'
    elif nodeName in externalOAs[1] and not nodeName in externalOAs[2] and nodeName in externalOAs[3]:
        return 'yellow'
    elif not nodeName in externalOAs[1] and nodeName in externalOAs[2] and nodeName in externalOAs[3]:
        return 'red'
    elif nodeName in externalOAs[1] or nodeName in externalOAs[2] or nodeName in externalOAs[3]:
        return 'khaki3'
    elif geography.isInGeography(nodeName):
        return 'brown'
    else:
        return 'white'
"""
def getNodeColor(nodeName, newOA, geography, currentOA):
    if nodeName[0] == 'F':
        return 'gray'
    # Other nodes
    # AND condition
    elif nodeName in newOA and nodeName in currentOA:
        return 'forestgreen'
    elif nodeName in newOA and not nodeName in currentOA:
        return 'orange'
    elif not nodeName in newOA and nodeName in currentOA:
        return 'skyblue'
    else:
        return 'white'

def getNodeShape(nodeName, geography):
    if geography.isInGeography(nodeName):
        return 'circle'
    else:
        return 'ellipse'

"""
def drawOA(internalOAs, externalOAs):

    # Initializing graph
    dot = Graph(comment = 'Observability areas', format='svg', engine='sfdp')
    dot.attr('graph', overlap='prism')
    dot.attr('graph', label=fileUCT + '\n' + 'Normalized threshold : ' + str(setOAnTh) + '\n' + 'Not-Normalized threshold : ' + str(setOATh))

    # Determining graph structures
    graphNodes = {}     #Nodes to draw on the map
    nodeEdges = {}      #Number of edges linked to each node
    graphEdges = []     #Edges to draw on the map
    for branch in branches:
        shortNodeNameFrom = branch.nodeFrom.name[:6]
        shortNodeNameTo = branch.nodeTo.name[:6]
        for nodeName in [shortNodeNameFrom, shortNodeNameTo]:
            if nodeName not in nodeEdges.keys():
                nodeEdges[nodeName] = 0
                graphNodes[nodeName] = {}
        for nodeName1, nodeName2 in [(shortNodeNameFrom, shortNodeNameTo),(shortNodeNameFrom, shortNodeNameTo)]:
            if nodeName1 < nodeName2:
                if len([edge for edge in graphEdges if edge[0] == nodeName1 and edge[1] == nodeName2]) == 0:
                    graphEdges.append([nodeName1, nodeName2])
                    nodeEdges[nodeName1] += 1
                    nodeEdges[nodeName2] += 1
    # Simplifying graph : tbd


    # Determining nodes characteristics
    dictInternalOAs = translateOANodesInGraphNodes(internalOAs)
    dictExternalOAs = translateOANodesInGraphNodes(externalOAs)

    # Translating inputs in node names

    for node in sorted(graphNodes, reverse=True):
        if nodeEdges[node] > 1:
            #Drawing Y-nodes
            if node[0] == "F" and node[1] == "Z":
                dot.node(node, shape='point')
            elif node[0] == "D" and node[2] == "Y":
                dot.node(node, shape='point')
            #Drawing real nodes
            else:
                #Control area
                if node in dictInternalOAs[0]:
                    colors = ['orangered', 'orange', 'red', 'yellow', 'gray']
                    nodeColor = getNodeColor(node, dictInternalOAs, colors)
                #External nodes
                else:
                    colors = ['limegreen', 'lawngreen', 'forestgreen', 'olivedrab', 'white']
                    nodeColor = getNodeColor(node, dictExternalOAs, colors)
                dot.node(node, style='filled', fillcolor=nodeColor)
        #Building mode for OA : 0, all the external elements; 1, all elements with N-2 IF >= setOATh; 
        #2, all elements with normalized N-2 IF >= setOAnTh; #3, all elements with N-2 IF >= setOATh OR normalized N-2 IF >= setOAnTh; 
        #4, all elements with N-2 IF >= setOATh AND normalized N-2 IF > setOAnTh
    for edge in [edge for edge in graphEdges if nodeEdges[edge[0]] > 1 and nodeEdges[edge[1]] >1]:
        dot.edge(edge[0], edge[1])
    
    
    dot.render('test-output/OA-' + countryCode + '-' + str(setOAnTh) + '-' + str(setOATh) + '.gv')
"""

# Utilitarian functions defined for a cleaner code
def translateOANodesInGraphNodes(dictOAs):
    """
    Function defined to build a dictionary of graph nodes names from a dictionary of OA nodes
    :param dictOAs: a dictionary with
    as keys the modes used to determine an OA
    as values the list of nodes belonging to the determined OA
    :return: a dictionary with
    as keys the modes used to determine an OA
    as values the list of graph nodes names belonging to the determined OA
    """

    tempDict = {}
    for i in [j for j in range(5) if j in dictOAs.keys()]:
        tempDict[i] = []
        for nodeName in [node.name[:6] for node in dictOAs[i]]:
            if not nodeName in tempDict[i]:
                tempDict[i].append(nodeName)
    return tempDict


#Constants (not to be modified by user)
SBase = 1.0 #MVA, for p.u conversion.
dictVBase = {}
dictVBase[0] = 750.0
dictVBase[1] = 380.0
dictVBase[2] = 220.0
dictVBase[3] = 150.0
dictVBase[4] = 120.0
dictVBase[5] = 110.0
dictVBase[6] = 70.0
dictVBase[7] = 27.0
dictVBase[8] = 330.0
dictVBase[9] = 500.0


if __name__ == '__main__':
    countries = ['A','B','C','D2','D4','D7','D8','E','F','G','H','I','J','L','M','N','O','P','Q','R','S','T',
                 'U','V','W','Y','Z','0']
    #countries = ['K']
    dictResults = {}
    for countryCode in countries:
        t0 = time.clock()
        tt = time.clock()
        fileLog = open("logs-" + countryCode + ".txt", "w")
        print("Required functions compiled ! Processing " + fileUCT)
            #Opening .uct file.
        with open(fileUCT, "r") as file:
            content = file.read().split('\n')
            branches = readLines(content)
            readTransformers(content, branches)
            dictCouplers = {}
            readCouplers(content, branches)
            generators = readGenerators(content)
            removeLoopElements(branches)


        if pMergeEquivalents:
            for mergedCountry in ['N', 'F', 'S', 'Z']:
                mergeEquivalents(branches, mergedCountry)

        nodes = determineNodes(branches)

        #Merging tie-lines
        if pMergeXNodes:    mergeTieLines(branches, nodes)

        print(str(len([branch for branch in branches if branch.impedance < 0])) + " branches have negative impedance.")
        print("System read from " + fileUCT + " in " + str(round(time.clock() - t0, 3)) + " seconds")
    
        t0 = time.clock()
        initializeRingAndConnection(nodes, countryCode)
    
        determineRings(nodes)
    
        #Restriction to main connex component
        nodes, branches = mainComponentRestriction(nodes, branches)

        attachGenerators(nodes, generators)
    
        storeTopology(branches, nodes)
        print("Topology determined in " + str(round(time.clock() - t0, 3)) + " seconds")

        t0 = time.clock()
        #Determination of set T
        setT = [branch for branch in branches if branch.ring == 0]
        sizeT = len(setT)
        # Reverse ring determination of set T
        # determineInverseRings(nodes)
        print("Control area contains " + str(sizeT) + " elements")
        # print("Control area inner rings determined in " + str(round(time.clock() - t0)) + " seconds")
    
        # Normalization matrix
        t0 = time.clock()
        arrayPATL = numpy.array([elt.PATL for elt in branches])
        arrayNorm = buildNormMatrix(arrayPATL)
        print("Normalization matrix built in " + str(round(time.clock() - t0,3)) + " seconds.")


        # PTDF matrix computation
        t0 = time.clock()
        arrayISF = computeISF(nodes, branches)
        arrayPTDF = computePTDF(branches, arrayISF)
        print("ISF and PTDF computed.")



        #N-1 IF computation
        t0 = time.clock()
        setR = [branch for branch in branches if branch.ring >0]
        LODF = computeLODF(branches, arrayPTDF)
        #storeN1(setR, setT, LODF, LODFn)   
        print("N-1 IF computed in " + str(round(time.clock() - t0, 1)) + " seconds.")
    
        #External contingencies selection
        t0 = time.clock()
        setIext = determineIext(setR, setT, LODF, arrayNorm)
        #Internal elements influenced by external contingencies
        setT1 = determineT1(setT, setIext, LODF, arrayNorm)
        #Internal contingencies
        setIint = determineIint(setT, setT1, LODF, arrayNorm)
        setI = setIext + setIint
        setI = excludeRadialI(setI)
        setT = excludeRadialI(setT)
        with open("setT.csv", "w") as file:
            file.write("Index;Name;Node From;Node To;Impedance;Ring" + '\n')
            for elt in setT:
                file.write(str(elt) + '\n')
        print("Sets determined in " + str(round(time.clock() - t0, 1)) + " seconds.")

        t0 = time.clock()
        
        results = computeIF(LODF, arrayNorm)
        dictResults[countryCode] = finalResult(branches, nodes, results)
        #Storing results in .csv file.
        storeResults(results)
        print("IF computed in " + str(round(time.clock() - t0,1)) + " seconds.")

        #Determining observability area for France
        if countryCode == 'F':
            print("Drawing Observability Areas")
            for i in range(3):
                setOATh = 0.03 + i * 0.01
                setOAnTh = 0.05 + i * 0.025
                modeOA = 4
                setOA = establishOA(results)
                nodesOAs = [node for node in nodes if len([branch for branch in node.branches if branch in setOA])>0]
                drawOA(nodesOAs)

        setR = [gen for gen in generators if gen.country != countryCode]
        genPower = numpy.array([elt.power for elt in setR])
        arrayNormg = buildNormGenerators(arrayPATL, genPower)
        LODFg, LODFgn = computeLODFg(setR, arrayISF)
        resultsSGU = computeIFSGU()
        storeResultsSGU(resultsSGU)
        print("IF determined for SGU in " + str(round(time.clock() - t0, 1)) + " seconds.")
        """
        for (i,j) in numpy.ndindex(1,1):
            setOATh = 0.03
            setOAnTh = 0.05 + j * 0.02
            nodesOAs = {}
            for modeOA in range(5):
                setOA = establishOA(results)
                nodesOAs[modeOA] = [node for node in nodes if len([branch for branch in node.branches if branch in setOA])>0]
            nodesOAs[0] = [node for node in nodes if node.ring == 0]
            drawOA(nodesOAs)


        setOA = establishOA(results)
        nodesOA = [node for node in nodes if len([branch for branch in node.branches if branch in setOA])>0]
        print("OA determined with " + str(len(setOA)) + " elements, " + str(len(nodesOA)) + " nodes.")
        extendedOA = [branch for branch in branches if branch.nodeFrom in nodesOA or branch.nodeTo in nodesOA]
        extendedNodesOA = [node for node in nodes if len([branch for branch in node.branches if branch in extendedOA])>0]
        print("OA extended to " + str(len(extendedOA)) + " elements, " + str(len(extendedNodesOA)) + " nodes in " + str(round(time.clock() - t0,1)) + " seconds." )
        
        #Determining SGU to assess
        t0 = time.clock()
        setR = [gen for gen in generators if gen.node in extendedNodesOA and gen.country != countryCode]
        print(str(len(setR)) + " generators to assess.")
        genPower = numpy.array([elt.power for elt in setR])
        arrayNormg = buildNormGenerators(arrayPATL, genPower)
        LODFg, LODFgn = computeLODFg(setR, arrayISF)
        resultsSGU = computeIFSGU()
        storeResultsSGU(resultsSGU)
        print("IF determined for SGU in " + str(round(time.clock() - t0, 1)) + " seconds.")
        """
        fileLog.close()
        print("Whole process performed in " + str(round(time.clock() - tt, 0)) + " seconds.")
    """
    for countryCode in countries:
        t0 = time.clock()
        print("Drawing Observability Areas")
        for (i,j) in numpy.ndindex(3,3):
            setOATh = round(0.05 + i * 0.025,2)
            setOAnTh = round(0.05 + j * 0.025,2)
            nodes = dictResults[countryCode].getNodes()
            branches = dictResults[countryCode].getBranches()
            results = dictResults[countryCode].getResults()
            internalOAs = {}
            internalOAs[0] = [node for node in nodes if node.ring == 0]
            externalOAs = {}
            for modeOA in range(1,5):
                setOA = establishOA(results)
                externalOAs[modeOA] = [node for node in nodes if
                                       len([branch for branch in node.branches if branch in setOA])>0]
                internalOAs[modeOA] = []
                for otherCountry in [country for country in countries if country != countryCode]:
                    otherResults = dictResults[otherCountry].getResults()
                    setOA = establishOA(otherResults)
                    internalOAs[modeOA] += [node for node in nodes if
                                  len([branch for branch in node.branches if branch in setOA])>0
                                  and node.ring == 0 and not node in internalOAs[modeOA]]
            drawOA(internalOAs, externalOAs)
    """