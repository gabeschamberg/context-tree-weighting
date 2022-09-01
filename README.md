# context-tree-weighting
Python implementation of the context tree weighting (CTW) method for sequential probability assignment of M-ary sequences with the option of including L-ary (with L and M not necessary being equal) side information. See `ctw-usage.ipynb` for example use cases of the provided `CTW` class.

For more on context tree weighting see this [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.352&rep=rep1&type=pdf)

For a description of the efficient updating implementation see Chapter 5 of this [follow up paper](https://www.sps.tue.nl/wp-content/uploads/2015/09/WillemsTjalkens1997eidma.pdf)

For more information about CTW with stale side information see this [paper](https://arxiv.org/pdf/1810.05250.pdf)

For a Matlab implementation that provided much guidance see [here](https://github.com/EEthinker/Universal_directed_information)