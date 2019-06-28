import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from deap import creator, base, tools, algorithms
import file_deal
import train
import sys
import warnings

def avg(l):
    """
    Returns the average between list elements
    """
    return (sum(l)/float(len(l)))


def getFitness(individual, Path, fea_dim):
    """
    Feature subset fitness function
    """

    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]

        dim = fea_dim - len(cols)
        file_deal.deal_file(Path, cols)
        ans = train.dnn(dim)
        '''
        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)
        
        # apply classification algorithm
        clf = LogisticRegression(solver='liblinear')
        '''
        return (ans,) #(avg(cross_val_score(clf, X_subset, y, cv=5)),)
    else:
        return(0,)


def geneticAlgorithm(Path, n_population, n_generation, fea_dim):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, fea_dim)#len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate", getFitness, Path=Path, fea_dim=fea_dim)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # initialize parameters
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                   ngen=n_generation, stats=stats, halloffame=hof,
                                   verbose=True)

    # return hall of fame
    return hof


def bestIndividual(hof):
    """
    Get the best individual
    """
    maxAccurcy = 0.0
    for individual in hof:
        if(float(individual.fitness.values[0]) > maxAccurcy):
            maxAccurcy = individual.fitness.values[0]
            _individual = individual

    header = ['consts', 'strings', 'offs', 'numAs', 'numCalls',
              'numIns', 'numLIs', 'numTIs', 'betw']
    _individualHeader = [header[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader


def getArguments():
    """
    Get argumments from command-line
    If pass only dataframe path, pop and gen will be default
    """
    dfPath = sys.argv[1]
    if(len(sys.argv) == 5):
        pop = int(sys.argv[2])
        gen = int(sys.argv[3])
        fea_dim = int(sys.argv[4])
    else:
        pop = 10
        gen = 2
        fea_dim = 9
    return dfPath, pop, gen, fea_dim


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # get dataframe path, population number and generation number from command-line argument
    dataframePath, n_pop, n_gen, fea_dim = getArguments()
    # read dataframe from csv
    '''
    df = pd.read_csv(dataframePath, sep=',')

    # encode labels column to numbers
    le = LabelEncoder()
    le.fit(df.iloc[:, -1])
    y = le.transform(df.iloc[:, -1])
    X = df.iloc[:, :-1]
    '''

    # get accuracy with all features
    individual = [1 for i in range(fea_dim)]
    print("Accuracy with all features: \t" +
          str(getFitness(individual, dataframePath, fea_dim)) + "\n")

    # apply genetic algorithm
    hof = geneticAlgorithm(dataframePath, n_pop, n_gen, fea_dim)

    # select the best individual
    accuracy, individual, header = bestIndividual(hof)
    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))
    print('Feature Subset\t: ' + str(header))

    print('\n\ncreating a new classifier with the result')

    # read dataframe from csv one more time
    '''
    df = pd.read_csv(dataframePath, sep=',')

    # with feature subset
    X = df[header]

    clf = LogisticRegression(solver='liblinear')

    scores = cross_val_score(clf, X, y, cv=5)
    '''
    scores = getFitness(individual, dataframePath, fea_dim)
    print("Accuracy with Feature Subset: \t" + str(scores[0]) + "\n")
