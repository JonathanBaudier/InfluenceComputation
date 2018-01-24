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
3. List the control areas on which the assessment shall be performed in the variable **countries** 
