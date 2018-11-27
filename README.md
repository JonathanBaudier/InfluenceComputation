# InfluenceComputation
A script compliant with EU's SOGL to assess influence of external elements on a UCTE DEF CGM

# Introduction
European TSOs are required by European Commission Regulation (EU) 2017/1485 of 2 August 2017 establishing a guideline on electricity transmission system operation to develop a methodology to assess the influence on their system of elements located in another TSO's control area. This script was developed as a proposal for an implementation of a methodology that should be compliant with SOGL's requirements and provides a flexible implementation.

# Requirements
This Python script requires the following dependencies:
* sys, time and math (standard Python packages)
* numpy, a package for scientific computing that provides matrix implementation
* numba, an optimizing compiler compatible with numba that compile functions to accelerate their execution
This script should be compatible with any grid model compliant with UCTE data exchange format (UCTE DEF) version 2. However, for large grid size (around 8 000 nodes, 13 000 branches for Continental Europe's Synchronous Area), a large memory space is required. Thus, for this kind of grid 8 GB of RAM are required (16 GB recommended).

# How to run it

1. Place this script in any folder
2. Provide a UCTE DEF grid model and set the variable **fileUCT** to the adequate path
    fileUCT = example.uct
3. List the control areas on which the assessment shall be performed in the variable **countries**
    countries = ['A','B','C','D2','D4','D7','D8','E','F','G','H','I','J','L','M','N','O','P','Q','R','S','T',                 'U','V','W','Y','Z','0']
4. Run the script !

Using default parameters, this script shall provides for each control area 2 .csv files. One with the influence factor of external grid elements and another with the influence factor of external generators.

**Results are provided in .csv file compliant with French formats i.e. semi-colon (;) as column separators and comma (,) as decimal separators**. This can be modified in the changing the parameters __colSep__ and __decSep__

# How influence is defined

For each grid element located outside of the investigated control area, the influence is defined as the maximum Line Outage Distribution Factor on any element located in the investigated control area in any N-i situation in which an i element is disconnected.
For each grid element located outside of the investigated control area, the influence is defined as the maximum Line Outage Distribution Factor on any element located in the investigated control area in any N-i situation in which an i element is disconnected multiplied by the ratio of MVA thermal limits of the investigated element and the influenced element.

# Handling of UCTE files

This script has been developed as part of tests performed to assess the feasability of SOGL methodology of identification of relevant assets. As such, it has been developed taking into account possible flaws that may be encountered in .uct files and using defaults parameters:
* any element with a current limit above 5,000 Amps is considered as physically irrelevant and its current limit is set to 0 Amps (thus resulting in an Idenfication Factor equal to 0), this may be changed by setting the value of __iatlTH__
* all nodes linked by a coupler i.e. by a line with status = 2 (according to UCTE DEF specifications) are merged into a unique node, this may be changed by setting the value of __pMergeCouplers__ to False
* the two tie-lines which are connected to a fictitious X-node are merged into a single line whose reactance is equal to the sum of the two reactances of the tie-lines and whose current limit is equal to the minimum current limit among the two tie-lines. If both tie-lines don't have the same order code, a fictitious code X is used. This may be changed by setting the value of __pMergeXNodes__ to False
