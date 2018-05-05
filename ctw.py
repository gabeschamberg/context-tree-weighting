import numpy as np

class CTW:
    def __init__(self,depth,symbols=2,sidesymbols=1,staleness=0):
        # tree depth
        self.D = depth
        # number of predicted symbols (for keeping counts)
        self.M = symbols
        # number of recent sideinfo samples to ignore
        self.K = staleness
        # number of symbols with side info (for contexts)
        self.Mtot = symbols*sidesymbols
        # create list of "restricted" contexts (no side info)
        self.rcontexts = range(symbols)
        # create list of "complete" contexts (w/ side info)
        self.ccontexts = []
        for x in range(symbols):
            for y in range(sidesymbols):
                self.ccontexts.append((x,y))
        # keep track of leaf nodes
        self.leaves = {}
        # create root (which in turn creates tree)
        self.root = Node(parent=None,context=[],ctw=self)
        # initialize distribution
        self.distribution = np.ones((symbols,))/symbols

    def set_distribution(self,distribution):
        self.distribution = distribution

    def get_distribution(self):
        return self.distribution

    def add_leaf(self,node):
        self.leaves[str(node.context)] = node

    def update(self,symbol,context):
        leaf = self.leaves[str(context)]
        leaf.update(symbol)

    def predict_sequence(self,seq,sideseq=None):
        # create a matrix of distributions (columns->samples,rows->symbols)
        distributions = np.zeros((self.M,len(seq)-self.D))
        # use first d symbols as context
        if sideseq is None:
            context = [seq[d] for d in reversed(range(self.D))]
        else:
            ccontext = [(seq[d],sideseq[d]) for d in reversed(range(self.D))]
            context = ccontext.copy()
            if self.K > 0:
                for k in range(self.K):
                    context[k] = ccontext[k][0]
        # loop though samples
        for n,x in enumerate(seq[self.D:]):
            # update the appropriate nodes for current context
            self.update(x,context)
            # get the distribution for the next symbol
            distributions[:,n] = self.get_distribution()
            # update the context
            if sideseq is None:
                context.insert(0,x)
                context = context[:self.D]
            else:
                ccontext.insert(0,(x,sideseq[n+self.D]))
                ccontext = ccontext[:self.D]
                context = ccontext.copy()
                if self.K > 0:
                    for k in range(self.K):
                        context[k] = ccontext[k][0]
        return distributions

class Node:
    def __init__(self,ctw,context,parent):
        self.ctw = ctw
        self.counts = np.zeros((self.ctw.M,))
        self.context = context
        self.parent = parent
        self.beta = 1
        if self.context == []:
            self.root = True
        else:
            self.root = False
        if len(self.context) == self.ctw.D:
            self.leaf = True
            self.ctw.add_leaf(self)
        else:
            self.leaf = False
        if not self.leaf:
            # if using side info, compare depth to staleness
            if (self.ctw.Mtot > self.ctw.M) and \
                        (len(self.context) >= self.ctw.K):
                contexts = self.ctw.ccontexts
            # otherwise just create context based on symbols
            else:
                contexts = self.ctw.rcontexts
            # create all the children
            self.children = []
            for c in contexts:
                self.children.append(Node(
                        ctw=ctw,
                        context=context+[c],
                        parent=self))

    def update(self,symbol,etain=None):
        # if we are at a leaf node with no incoming eta
        if etain is None:
            etain = (self.counts[:-1]+0.5)/(self.counts[-1]+0.5)
        # get the number of symbols
        M = self.counts.size
        # find the weighted probabilities using incoming eta
        pw = np.append(etain,1)
        pw = pw/sum(pw)
        # find the kt estimates
        pe = (self.counts + 0.5)/(sum(self.counts)+0.5*M)
        # compute outgoing eta
        etaout = (self.beta*pe[:-1]+pw[:-1])/(self.beta*pe[-1]+pw[-1])
        # update beta
        self.beta *= pe[symbol]/pw[symbol]
        # update counts
        self.counts[symbol] += 1
        # if not root, pass outgoing eta up
        if not self.root:
            self.parent.update(symbol,etaout)
        # if we are at the root, set the ctw distribution
        else:
            etasum = sum(etaout) + 1
            self.ctw.set_distribution(np.append(etaout,1)/etasum)