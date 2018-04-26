import numpy as np
from numpy.random import multinomial as multn
import matplotlib.pyplot as plt

# Generate n samples from a tree source defined by the dictionary params where
##### IF NOT USING SIDE INFO
# params[key] = [p0,...,pL] with sum(pi) 1 is a
# list of parameters for the probabilities of x in [1...L]
# for context given by key = str([x0,...,xD])
##### IF USING SIDE INFO
# params[key] = ([p0,...,pL],[q0,...,qM]) with sum(pi) = sum(qi) = 1 is a
# tuple of parameters for the probabilities of x in [1...L] and y in [1...M]
# for context given by key = str(([x0,...,xD],[y0,...,yD]))
def gendata(N=1000,params=None,plot=True,plot_samples=-1):
    # if no params specified do this simple tree
    if params is None:
        params = {'[0]':[0.5,0.5],
                  '[1]':[0.8,0.2]}
    # check to see if using side info (are params a list of ints or tuples?)
    if list(params.keys())[0][1] == '[':
        # find number of symbols in x and y
        L = len(params[list(params.keys())[0]][0])
        M = len(params[list(params.keys())[0]][1])
        # find tree depth
        D = len(list(params.keys())[0].split(','))//2
    else:
        # find number of symbols in x
        L = len(params[list(params.keys())[0]])
        M = 1
        # find tree depth
        D = len(list(params.keys())[0].split(','))
    # keep track of the probabilities of each x at each time
    pxs = np.zeros((L,N))
    # initialize the first context to all zeros
    x = [0]*D
    if M > 1:
        y = [0]*D
    # generate the rest of the samples
    for n in range(N):
        if M > 1:
            context = [list(x[-D:]),list(y[-D:])]
            (px,py) = params[str(context).replace(' ','')]
            x = np.append(x,int(np.argmax(multn(1,px))))
            y = np.append(y,int(np.argmax(multn(1,py))))
        else:
            context = x[-D:]
            px = params[str(context).replace(' ','')]
            x = np.append(x,int(np.argmax(multn(1,px))))
        # store the true distribution of this sample
        pxs[:,n] = px
    # plot the data
    if plot:
        fig,[ax1,ax2] = plt.subplots(2,1,figsize=(15,6))
        if M > 1:
            ax1.plot(x[D:],'ro',label='Sequence')
            ax1.plot(y[D:],'bo',markersize=1,label='Side Sequence')
            ax1.set_title('Generated Sequences')
        else:
            ax1.plot(x[D:],'ro',label='Sequence')
            ax1.set_title('Generated Sequence')
        ax1.legend()
        if plot_samples > 0:
            ax1.set_xlim([N-plot_samples,N])
        plotprobs(pxs,ax2,plot_samples=plot_samples)
    return x,y

def plotprobs(probs,ax=None,plot_samples=-1,estimate=False):
    L,N = probs.shape
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(15,3))
    for n in range(N):
        for l in range(L):
                ax.plot([n],[l],'ro',alpha=probs[l,n],
                    markersize=15*probs[l,n])
                # only display end probabilities
                if plot_samples > 0:
                    ax.set_xlim([N-plot_samples,N])
                if estimate:
                    ax.set_title('Estimated Sequence Probabilities')
                else:
                    ax.set_title('Sequence Probabilities')