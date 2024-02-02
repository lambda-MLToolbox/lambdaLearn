import numpy as np

def POSS_MSE(X,y,k):
    """
    ATTN
    ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou (
    
    ATTN2
    ATTN2: This package was developed by Mr. Chao Qian (
    For any problem concerning the code, please feel free to contact Mr. Qian.
    The given criterion to be minimized is the mean squared error
    Some varables used in the code
    input:
    NOTE THAT we use a Boolean string "1001....1100" to represent a subset of variables, where "1" means that the corresponding variable is selected, and "0" means ignoring the variable.
    X: a matrix (m*n); one column corresponds to one observation variable.
    y: a vector (m*1); it corresponds to the predictor variable.
    k: the constraint on the number of selected variables.
    NOTE THAT all variables are normalized to have expectation 0 and variance 1.
    ouptut:
    selectedVariables: a Boolean string of length n representing the selected variables, the number of which is not larger than k.
    """
    m,n=X.shape
    # initialize the candidate solution set (called "population"): generate a Boolean string with all 0s (called "solution").
    population=np.zeros((1,n))
    # popSize: record the number of solutions in the population.
    popSize=1
    # fitness: record the two objective values of a solution.
    fitness=np.zeros((1,2))
    # the first objective is f; for the special solution 00...00 (i.e., it does not select any variable) and the solutions with the number of selected variables not smaller than 2*k, set its first objective value as inf.
    fitness[0,0]=np.inf
    # the second objective is the number of selected variables, i.e., the sum of all the bits.
    fitness[0,1]=np.sum(population)
    # repeat to improve the population; the number of iterations is set as 2*e*k^2*n suggested by our theoretical analysis.
    T=int(np.round(n*k*k*2*np.exp(1)))

    for i in range(T):
        # randomly select a solution from the population and mutate it to generate a new solution.
        offspring=np.abs(population[np.random.randint(0,popSize,1),:]-np.random.choice([1,0],size=n,p=[1/n,1-1/n]))
        # compute the fitness of the new solution.
        offspringFit=np.zeros((1,2))
        offspringFit[0,1]=np.sum(offspring)
        if offspringFit[0,1]==0 or offspringFit[0,1]>=2*k:
            offspringFit[0,0]=np.inf
        else:
            pos=np.where(offspring==1)[0]
            coef=np.linalg.lstsq(X[:,pos],y,rcond=None)[0]
            err=X[:,pos].dot(coef)-y
            offspringFit[0,0]=err.T.dot(err)/m
        # use the new solution to update the current population.
        if np.sum((fitness[0:popSize,0]<offspringFit[0,0])*(fitness[0:popSize,1]<=offspringFit[0,1]))+np.sum((fitness[0:popSize,0]<=offspringFit[0,0])*(fitness[0:popSize,1]<offspringFit[0,1]))>0:
            continue
        else:
            deleteIndex=((fitness[0:popSize,0]>=offspringFit[0,0])*(fitness[0:popSize,1]>=offspringFit[0,1])).T
        # ndelete: record the index of the solutions to be kept.
        ndelete=np.where(deleteIndex==0)[0]
        population=np.concatenate((population[ndelete,:],offspring),axis=0)
        fitness=np.concatenate((fitness[ndelete,:],offspringFit),axis=0)
        popSize=len(ndelete)+1
    # select the final solution according to the constraint k on the number of selected variables.
    temp=np.where(fitness[:,1]<=k)[0]
    j=np.max(fitness[temp,1])
    seq=np.where(fitness[:,1]==j)[0]
    selectedVariables=population[seq,:]
    return selectedVariables

