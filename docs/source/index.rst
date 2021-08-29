.. msp documentation master file, created by
   sphinx-quickstart on Sun Aug 29 00:55:53 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MSP's documentation!
===============================

Welcome to the Machine Scheduling Problem (MSP) python package guide!

MSP python package offers a deep reinforcement learning model based on
low-level API of tensorflow. It addresses the MSP problem especially 
for the Fabric Industry. However, the presented method is not only 
limited to it.

**NOTE: This package has been built during the research internship at 
LIMOS in Clermont-Ferrand, France**

In this research, we design Deep Learning model to sovle MSP problem 
(NP-hard combinatorial optimization problem) where the goal is to find a
schedule from combinatorial search space such that it minimizes makespan
(i.e., an objective) under certain constraints.

To define the MSP problem, we employ graph structure to model the various 
interaction between machines and jobs. We cast the MSP as a Graph-to-Sequence 
machine learning problem that follows the standard encoder-decoder architecture 
consisting of graph encoder, and a recurrent decoder. We train the model using a
policy-gradient reinforcement learning algorithm, namely REINFORCE algorithm. 
Furthermore, we also present different novel masking strategies to filter out 
the invalid action at any timestep during the decoding.

**To access the full research document, 
send the Request Email to mohammedbilalansari.official@gmail.com**

If you want to use this package follow the instructions to 
:ref:`install <settingup>` it.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   instructions

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
