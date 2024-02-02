import numpy as np


def POSS_RSS(X,y,k,lambda_):
    """
    ATTN
    ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou (

    ATTN2
    ATTN2: This package was developed by Mr. Chao Qian (
    """
    [m,n]=X.shape
    C=np.dot(X.T,X)
    b=np.dot(X.T,y)
    #initialize the candidate solution set (called "population"): generate a Boolean string with all 0s (called "solution").
    population=np.zeros((1,n))
    #popSize: record the number of solutions in the population.
    popSize=1
    #fitness: record the two objective values of a solution.
    fitness=np.zeros((1,2))
    #the first objective is f; for the special solution 00...00 (i.e., it does not select any variable) and the solutions with the number of selected variables not smaller than 2*k, set its first objective value as inf.
    fitness[0,0]=np.inf
    #the second objective is the number of selected variables, i.e., the sum of all the bits.
    fitness[0,1]= np.sum(population)
    #repeat to improve the population; the number of iterations is set as 2*e*k^2*n suggested by our theoretical analysis.
    T=int(np.round(n*k*k*2*np.exp(1)))

    for i in range(1,T):
        #randomly select a solution from the population and mutate it to generate a new solution.
        offspring=np.abs(population[np.random.randint(0,popSize,1),:]-np.random.choice([1,0],n,p=[1/n,1-1/n]))
        #compute the fitness of the new solution.
        offspringFit=np.zeros((1,2))
        offspringFit[0,1]= np.sum(offspring)
        if offspringFit[0,1]==0 or offspringFit[0,1]>=2*k:
            offspringFit[0,0]=np.inf
        else:
            pos=np.where(offspring==1)[0]
            # coef=np.linalg.inv(C[pos,pos]+lambda_*m*np.eye(len(pos)))*b[pos]
            coef=np.linalg.inv(C[pos,pos]+lambda_*m*np.eye(len(pos)))@b[pos]
            res=y-np.dot(X[:,pos],coef)
            offspringFit[0,0]=np.dot(res.T,res)/m+lambda_*np.dot(coef.T,coef)

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
