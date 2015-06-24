# RecurrentNN.jl

RecurrentNN.jl is a Julia language package originally based on Andrej Karpathy's excellent [RecurrentJS](http://cs.stanford.edu/people/karpathy/recurrentjs) library in javascript.
It implements:

- Deep **Recurrent Neural Networks** (RNN)
- **Long Short-Term Memory networks** (LSTM)
- **Gated Recurrent Neural Networks** (GRU)
- **Gated Feedback Recurrent Neural Networks** (GF-RNN)
- **Gated Feedback Long Short-Term Memory networks** (GF-LSTM)
- In fact, the library is more general because it has functionality to construct arbitrary **expression graphs** over which the library can perform **automatic differentiation** similar to what you may find in Theano for Python, or in Torch etc. Currently, the code uses this very general functionality to implement RNN/LSTM/GRU, but one can build arbitrary Neural Networks and do automatic backprop.
- For information an the **Gated Feedback** variants see [Gated Feedback Recurrent Neural Networks](http://arxiv.org/abs/1502.02367) 


## Online demo of original library in javascript

An online demo that memorizes character sequences can be found below. Sentences are input data and the networks are trained to predict the next character in a sentence. Thus, they learn English from scratch character by character and eventually after some training generate entirely new sentences that sometimes make some sense :)

[Character Sequence Memorization Demo](http://cs.stanford.edu/people/karpathy/recurrentjs)

*The same demo as above implemented in Julia can be found in* [example/example.jl](https://github.com/Andy-P/RecurrentNN.jl/blob/master/example/example.jl)


## Example code

To construct and train an LSTM for example, you would proceed as follows:

```julia
using RecurrentNN

# takes as input Mat of 10x1, contains 2 hidden layers of
# 20 neurons each, and outputs a Mat of size 2x1
hiddensizes = [20, 20]
outputsize = 2
cost = 0.
lstm = LSTM(10, hiddensizes, outputsize)
x1 = randNNMat(10, 1) # example input #1
x2 = randNNMat(10, 1) # example input #2
x3 = randNNMat(10, 1) # example input #3

# pass 3 examples through the LSTM
G = Graph()
# build container to hold output after each time step
prevhd   = Array(NNMatrix,0) # holds final hidden layer of the recurrent model
prevcell = Array(NNMatrix,0) #  holds final cell output of the LSTM model
out  = NNMatrix(outputsize,1) # output of the recurrent model
prev = (prevhd, prevcell, out)

out1 = forwardprop(G, lstm, x1, prev)
out2 = forwardprop(G, lstm, x2, out1);
out3 = forwardprop(G, lstm, x3, out2);

# the last part of the tuple contains the outputs:
outMat =  prev[end]

# for example lets assume we have binary classification problem
# so the output of the LSTM are the log probabilities of the
# two classes. Lets first get the probabilities:
probs = softmax(outMat)
ix_target = 1 # suppose first input has target class

# cross-entropy loss for softmax is simply the probabilities:
outMat.dw = probs.w
# but the correct class gets an extra -1:
outMat.dw[ix_target] -= 1;

# in real application you'd probably have a desired class
# for every input, so you'd iteratively se the .dw loss on each
# one. In the example provided demo we are, for example,
# predicting the index of the next letter in an input sentence.

# update the LSTM parameters
backprop(G)
s = Solver() # RMSProp optimizer

# perform RMSprop update with
# step size of 0.01
# L2 regularization of 0.00001
# and clipping the gradients at 5.0 elementwise
step(s, lstm, 0.01, 0.00001, 5.0);
```

A much more detailed example can be found in the example folder.

##Credits
This library draws on the work of [Andrej Karpathy](https://github.com/karpathy). Speed enhancements were added by [Iain Dunning](https://github.com/IainNZ). The Gated Recurrent Neural Network implementation and Gated Feedback variants were added by [Paul Heideman](https://github.com/paulheideman).

## License
MIT
