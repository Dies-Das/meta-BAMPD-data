# meta-BAMDP

## Project Description and Motivation

In this project we formulate and solve a meta version of a BAMDP problem for two-armed Bernoulli bandit (TABB) task. The file convexity_1comp.py obtains the solution of the meta-BAMDP using the myopic assumption (i.e. k=2 from the manuscript). The file meta_tree.py shows a more general implementation and allows to solve the meta-BAMDP for more relaxed assumptions. All the other files either have helper functions, or are used for making plots. 


## Prerequisites

To reproduce the data presented in the paper, you need Python 3.11 or above installed and need to install the dependencies specified in requirements.txt

## Installation and Usage

### Installation
Install the dependencies with
```
pip install -r requirements.txt
```

### Usage

run 
```
python main.py
```
to generate all plots found in the paper. This file also shows normalized value obtained as a function of computational cost for the two kinds of approximation schemes mentioned in the manuscript.
