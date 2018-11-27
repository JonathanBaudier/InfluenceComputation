# -*-coding:utf-8 -*
#   This Source Code Form is subject to the terms of the Apache License, v. 2.0.
#   If a copy of the licence was not distributed with this file, You can obtain one at http://www.apache.org/licenses/LICENSE-2.0
import numpy
import sys
import time
import math
from numba import jit

#   Parameters
countries = ['A','B','C','D2','D4','D7','D8','E','F','G','H','I','J','L','M','N','O','P',
             'Q','R','S','T', 'U','V','W','Y','Z','0'] # The list of control area on which the assessment is performed.
epsilon = 0.001              #As we are working on numeric value, values below epsilon are rounded to 0 and values between 1 - epsilon and 1 + epsilon are rounded to 1
iatlTH = 5000               #Threshold above which IATL are regarded as infinite.
ringChars = 7               #Significant characters used to determined rings : 8, one ring per node; 7, one ring per voltage level; 6, one ring per substation (all voltage)
fileUCT = 'example.uct'

#pre-processing parameters
pMergeCouplers = True
pMergeXNodes = True
pMergeEquivalents = False

#output parameters
colSep = ';'        #Column separator for .csv files ("," international standard, ";" for France)
decSep = ','        #Decimal separator for .csv files ("." international standard, "," for France)

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
            print(f'Error while merging branch : {self.nameBranch}, '
                  f'node:{node1} was not found among {self.nameFrom} and {self.nameTo}')

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
        return f'Index{colSep}Name{colSep}Node From{colSep}Node To{colSep}' \
               f'Impedance{colSep}PATL{colSep}Ring{colSep}Tie-Line'
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
        return f'name{colSep}PATL{colSep}Filtering IF{colSep}i{colSep}' \
               f't{colSep}Identification IF{colSep}i{colSep}t\n'
    header = staticmethod(header)

    def __str__(self):
        resultat = f'{self.eltR.nameBranch}{colSep}{self.eltR.PATL}{colSep}' \
                   f'{str(round(self.IFN2,4)).replace(".", decSep)}{colSep}' \
                   f'{self.eltI.nameBranch}{colSep}{self.eltT.nameBranch}{colSep}' \
                   f'{str(round(self.nIFN2,4)).replace(".", decSep)}{colSep}' \
                   f'{self.eltIn.nameBranch}{colSep}{self.eltTn.nameBranch}\n'
        return resultat     

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
        return f'Index{colSep}Name{colSep}Ring{colSep}Connected{colSep}Branches'
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
        print('No line ##L was found in the UCT file')
        sys.exit()
    print('Lines read')
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
        print('No line ##T was found in the UCT file')
        sys.exit()
    print('Transformers read')

def readGenerators(fileRead):
    fileLog.write('Reading generators' + '\n')
    generators = []
    #looking for  ##N line.
    i = 0
    while i < len(fileRead) and fileRead[i] != '##N':
        i += 1
    if i<len(fileRead):
        i+=1
        while i < len(fileRead) and (fileRead[i][0:2] != "##" or fileRead[i][0:3] == "##Z"):
            if len(fileRead[i]) > 80:
                nodeName = fileRead[i][0:8]
                try:
                    generatorPower = float(fileRead[i][73:80])
                    if generatorPower >= 0.0:
                        fileLog.write(f'     Generator {nodeName} has negative or '
                                      f'zero maximum generation power\n')
                    else:
                        generators.append(GenerationUnit(nodeName, -generatorPower))
                except:
                    fileLog.write(f'     Generator {nodeName} maximum permissible '
                                  f'generation could not be read.\n')
            i+=1
    else:
        print('No line ##N was found in the UCT file')
        sys.exit()
    print('Generators read')
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
    fileLog.write('Attaching generators' + '\n')
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
    print(f'Connectivity established in {connectionSteps} steps.')
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
        print(f'Node {elt} is connected but has no ring')
    for elt in [node.name for node in setOfNodes if node.ring < 99 and not node.connected]:
        print(f'Node {elt} is in a ring but is not connected')
    print(f'Rings determined. Maximum ring is #{currentRing - 1}.')
    
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
            print(f'Node {node.name} should not be in main component')
    for branch in connexBranches:
        if not branch.nodeFrom.connected:
            print(f'Branch {branch.nameBranch} should not be in main component')
        if not branch.nodeTo.connected:
            print(f'Branch {branch.nameBranch} should not be in main component')
    #Rebuilding index
    for i in range(len(connexNodes)):
        connexNodes[i].index = i
    for i in range(len(connexBranches)):
        connexBranches[i].index = i
    print(f'System restricted to main connected component with {len(connexNodes)} nodes '
          f'and {len(connexBranches)} elements.')
    return connexNodes, connexBranches

def computeISF(setOfNodes, setOfElements):
    t1 = time.clock()
    #selecting slack node
    maxConnection = max([len(node.branches) for node in setOfNodes if node.name[0:len(countryCode)]== countryCode and node.name[6:7] == "1"])
    slackNode = [node for node in setOfNodes if len(node.branches) == maxConnection and node.name[0:len(countryCode)] == countryCode][0]
    print(f'Slack node for the system is {slackNode.name}')
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
    print(f'Susceptance matrix B built in {round(time.clock() - t1, 2)} seconds.')
    #Susceptance matrix inversion
    t1 = time.clock()
    inverseB = numpy.linalg.inv(matrixB)
    print(f'Susceptance matrix B inverted in {round(time.clock() - t1, 2)} seconds.')
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
    print(f'ISF matrix computed in {round(time.clock() - t1,1)} seconds.')
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
    fileLog.write('computing IF for SGU')
    LODF = []
    for elt in setOfElements:
        balancingGenerators = [eltGenerator for eltGenerator in generators if eltGenerator.country == elt.country and 
                               eltGenerator != elt]
        if len(balancingGenerators) == 0:
            fileLog.write(f'No generators found to balance the contingency of {elt.name}')
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
            print(f'{elt.nameBranch} has negative self-PTDF : {elt.PTDF}')
        if elt.PTDF>1+epsilon:
            print(f'{elt.nameBranch} has self-PTDF higher than 1 : {elt.PTDF}')
    print(f'PTDF computed in {round(time.clock() - t1, 1)} seconds.')
    return arrayPTDF

def determineIext(setOfElements, setT, inputLODF, normalisationMatrix):
    setIext = []
    currentRing = 1
    setJ = [elt for elt in setOfElements if elt.ring == currentRing]
    while len(setJ) > 0:
        LODF = numpy.absolute(matrixReduction(setJ, setT, inputLODF))
        LODFn = LODF * matrixReduction(setJ, setT, normalisationMatrix)
        for i in range(len(setJ)):
            eltI = setJ[i]
            #Include any element from the ring
            setIext.append(eltI)
            #fileI.write(eltI.nameBranch + "," + str(eltI.ring) + "," + str(eltI.PTDF) + "," + str(numpy.amax(LODF[:,i])) + "," + str(numpy.amax(LODFn[:,i])) + '\n')
        currentRing += 1
        setJ = [elt for elt in setOfElements if elt.ring == currentRing]
    print(f'External contingencies determinated : {len(setIext)} elements selected within '
          f'maximum ring {currentRing}')
    return setIext

def determineT1(setT, setIext, inputLODF, normalisationMatrix):
    setT1 = []
    setT1 = setT
    print(f'Influenced Internal elements determinated : {len(setT1)} elements selected.')
    return setT1

def determineIint(setT1, setT, inputLODF, normalisationMatrix):
    setIint = list(setT)
    print(f'Internal contingencies determinated : {len(setIint)} elements selected.')
    return setIint

def excludeRadialI(setJ):
    result = []
    for eltI in setJ:
        if eltI.PTDF > 1 - epsilon:
            pass
        else:
            result.append(eltI)
    print(f'Radial elements which do not lead to disconnection of a generetor are excluded : '
          f'{len(result)}/{len(setJ)} kept.')
    return result

def computeIF(inputLODF, normalizationMatrix):
    results = []

    sizeI = len(setI)
    sizeT = len(setT)
    print(f'Internal elements monitored : {sizeT}')
    print(f'Contingencies : {sizeI}')
    vectorKii = [elt.PTDF for elt in setI]
    matrixKit = matrixReduction(setI, setT, arrayPTDF)
    excludeTI = excludeAB(setT, setI)

    currentRing = 1
    setR = excludeRadialI([elt for elt in branches if elt.ring == currentRing])
    #setR = excludeRadialI([elt for elt in branches if elt.ring >0])
    while len(setR)>0:
        sizeR = len(setR)
        setsSize = numpy.array([sizeR,sizeI,sizeT], dtype=numpy.int32)
        print(f'Assessing IF for ring #{currentRing} with {sizeR} elements.')
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
        currentRing += 1
        setR = [elt for elt in branches if elt.ring == currentRing]
    return results

def storeResults(results):
    with open(f'resultsElements-{countryCode}.csv', 'w') as fileOut:
        #Writing results
        fileOut.write(resultIF.header())
        for elt in results:
            fileOut.write(str(elt))         

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
            vectorLODFn = rLODFgn[:,r] + rLODFn[:,i]*LODFgn[setI[i].index,r]
            IFn = numpy.max(numpy.abs(vectorLODFn))
            if IFn > results[r][5]:
                results[r][5] = IFn
                results[r][6] = [setI[i].nameBranch]
                results[r][7] = [setT[k].nameBranch for k in range(len(setT)) if abs(vectorLODFn[k]) == IFn]
    return results

def storeResultsSGU(results):
    with open(f'resultsSGU-{countryCode}.csv', 'w') as fileOut:
        fileOut.write(f'SGU{colSep}Power{colSep}Filtering IF{colSep}i{colSep}t{colSep}'
                      f'Identification IF{colSep}i{colSep}t\n')
        for elt in resultsSGU:
            fileOut.write(f'{elt[0]}{colSep}{str(elt[1]).replace(".",decSep)}{colSep}'
                          f'{str(round(elt[2],4)).replace(".",decSep)}{colSep}'
                          f'{str(elt[3]).replace("[","").replace("]","")}{colSep}'
                          f'{str(elt[4]).replace("[","").replace("]","")}{colSep}'
                          f'{str(round(elt[5],4)).replace(".", decSep)}{colSep}'
                          f'{str(elt[6]).replace("[","").replace("]","")}{colSep}'
                          f'{str(elt[7]).replace("[","").replace("]","")}\n')

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
    dictResults = {}
    for countryCode in countries:
        t0 = time.clock()
        tt = time.clock()
        fileLog = open(f'logs-{countryCode}.txt', 'w')
        print(f'Required functions compiled ! Processing {fileUCT}')
            #Opening .uct file.
        with open(fileUCT, 'r') as file:
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

        print(f'{len([branch for branch in branches if branch.impedance < 0])} branches '
              f'have negative impedance.')
        print(f'System read from {fileUCT} in {round(time.clock() - t0, 1)} seconds')
    
        t0 = time.clock()
        initializeRingAndConnection(nodes, countryCode)
    
        determineRings(nodes)
    
        #Restriction to main connex component
        nodes, branches = mainComponentRestriction(nodes, branches)

        attachGenerators(nodes, generators)
    
        #storeTopology(branches, nodes)
        print(f'Topology determined in {round(time.clock() - t0, 1)} seconds')

        t0 = time.clock()
        #Determination of set T
        setT = [branch for branch in branches if branch.ring == 0]
        sizeT = len(setT)
        # Reverse ring determination of set T
        # determineInverseRings(nodes)
        print(f'Control area contains {sizeT} elements')
        # print("Control area inner rings determined in " + str(round(time.clock() - t0)) + " seconds")
    
        # Normalization matrix
        t0 = time.clock()
        arrayPATL = numpy.array([elt.PATL for elt in branches])
        arrayNorm = buildNormMatrix(arrayPATL)
        print(f'Normalization matrix built in {round(time.clock() - t0,1)} seconds.')

        # PTDF matrix computation
        t0 = time.clock()
        arrayISF = computeISF(nodes, branches)
        arrayPTDF = computePTDF(branches, arrayISF)
        print('ISF and PTDF computed.')

        #N-1 IF computation
        t0 = time.clock()
        setR = [branch for branch in branches if branch.ring >0]
        LODF = computeLODF(branches, arrayPTDF)
        print(f'N-1 IF computed in {round(time.clock() - t0, 1)} seconds.')
    
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
        print(f'Sets determined in {round(time.clock() - t0, 1)} seconds.')

        t0 = time.clock()
        
        results = computeIF(LODF, arrayNorm)
        #Storing results in .csv file.
        storeResults(results)
        print(f'IF computed in {round(time.clock() - t0,1)} seconds.')
        t0 = time.clock()
        t1 = time.clock()
        setR = [gen for gen in generators if gen.country != countryCode]
        genPower = numpy.array([elt.power for elt in setR])
        arrayNormg = buildNormGenerators(arrayPATL, genPower)
        LODFg, LODFgn = computeLODFg(setR, arrayISF)
        print(f'IF computation for SGU initialised in {round(time.clock() - t1, 1)} seconds.')
        resultsSGU = computeIFSGU()
        storeResultsSGU(resultsSGU)
        print(f'IF determined for SGU in {round(time.clock() - t0, 1)} seconds.')
        fileLog.close()
        print(f'Whole process performed in {round(time.clock() - tt, 0)} seconds.')