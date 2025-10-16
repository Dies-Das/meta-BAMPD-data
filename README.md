# meta-BAMDP
**Note for readers coming from the arXiv paper:**  
The version of the code used in the paper is available under the [v1.0 release](https://github.com/Dies-Das/meta-BAMPD-data/releases/tag/v1.0).  
For the most up-to-date code, please refer to the main branch.
## Project Description and Motivation

In this project we formulate and solve a meta version of a BAMDP problem for two-armed Bernoulli bandit (TABB) task. The file convexity_1comp.py obtains the solution of the meta-BAMDP using the myopic assumption (i.e. k=2 from the manuscript). The file meta_tree.py shows a more general implementation and allows to solve the meta-BAMDP for more relaxed assumptions. All the other files either have helper functions, or are used for making plots. 

## C++
### Prerequisites
This project depends on boost and unordered_dense. The latter is fetched and build by the project itself, the former you need to install on your system.

On Debian-based systems, this is done via
´´´
sudo apt install libboost-all-dev
´´´
clone this repository with
´´´
$ git clone https://github.com/Dies-Das/meta-BAMPD-data
´´´
then build and run the code
´´´
$ cd meta-BAMPD-data
$ mkdir build
$ mkdir bin
$ cd build
$ cmake ..
$ make
$ ../bin/meta-BAMDP
´´´
So far, the only output (relevant for you) are the belief states corresponding to the metanodes of the metagraph. 
### TODOs
- Approximations, i.e. maximum belief size, maximum computations, maximum depth while search
## Python
### Prerequisites

To reproduce the data presented in the paper, you need Python 3.11 or above installed and need to install the dependencies specified in requirements.txt

### Installation and Usage

#### Installation
Install the dependencies with
```
pip install -r requirements.txt
```

#### Usage

run 
```
python main.py
```
to generate all plots found in the paper. This file also shows normalized value obtained as a function of computational cost for the two kinds of approximation schemes mentioned in the manuscript.
