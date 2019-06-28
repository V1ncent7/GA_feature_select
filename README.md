
# Genetic Algorithm For Feature Selection
> Search the best feature subset for DNN model

## Description
Feature selection is the process of finding the most relevant variables for a predictive model. These techniques can be used to identify and remove unneeded, irrelevant and redundant features that do not contribute or decrease the accuracy of the predictive model.

In nature, the genes of organisms tend to evolve over successive generations to better adapt to the environment. The Genetic Algorithm is an heuristic optimization method inspired by that procedures of natural evolution.

In feature selection, the function to optimize is the generalization performance of a predictive model. More specifically, we want to minimize the error of the model on an independent data set not used to create the model.

## Dependencies
[tensorflow](https://tensorflow.google.cn/)

[Pandas](https://pandas.pydata.org/)

[Numpy](http://www.numpy.org/)

[scikit-learn](http://scikit-learn.org/stable/)

[Deap](https://deap.readthedocs.io/en/master/)


## Usage
1. Go to the repository folder
1. Run:
```
python GA.py path n_population n_generation
```
Obs:
  - `path` should be the path to the folder features extracted.
  - `n_population` and `n_generation` must be integers
  - You can go to the code and change the classifier so that the search is optimized for your classifier.

## Usage Example
```
python GA.py D:\test_025\GA_feature_selection\json\ 20 6
```
Returns:
```
Accuracy with all features: 	(0.9037199443989865,)

gen	nevals	avg     	min     	max     
0  	30    	0.899774	0.867973	0.929663
1  	13    	0.911329	0.887807	0.929663
2  	19    	0.919716	0.896043	0.93313 
3  	19    	0.923191	0.888438	0.935418
4  	19    	0.927005	0.899314	0.935418
5  	25    	0.926911	0.890472	0.936996
Best Accuracy: 	(0.9354180427260271,)
Number of Features in Subset: 	7
Individual: 		[1, 0, 0, 1, 1, 1, 1, 1, 1]
Feature Subset	: ['consts', 'numAs', 'numCalls', 'numIns', 'numLIs', 'numTIs', 'betw']


creating a new classifier with the result
Accuracy with Feature Subset: 	0.9274146511354659

```
## Fonts
1. This repository was heavily based on [GeneticAlgorithmFeatureSelection](https://github.com/scoliann/GeneticAlgorithmFeatureSelection)
1. For the description was used part of the introduction of  [Genetic algorithms for feature selection in Data Analytics](https://www.neuraldesigner.com/blog/genetic_algorithms_for_feature_selection). Great text.

#### Author: [V1ncent7](https://github.com/V1ncent7)
