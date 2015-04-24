# RecurrentNN.jl

RecurrentNN.jl is a Julia language package based on Andrej Karpathy's excellent [RecurrentJS] library in javascript
(http://cs.stanford.edu/people/karpathy/recurrentjs) that implements:

- Deep **Recurrent Neural Networks** (RNN)
- **Long Short-Term Memory networks** (LSTM)
- In fact, the library is more general because it has functionality to construct arbitrary **expression graphs** over which the library can perform **automatic differentiation** similar to what you may find in Theano for Python, or in Torch etc. Currently, the code uses this very general functionality to implement RNN/LSTM, but one can build arbitrary Neural Networks and do automatic backprop.

## Online demo of original library in javascript

An online demo that memorizes character sequences can be found below. Sentences are input data and the networks are trained to predict the next character in a sentence. Thus, they learn English from scratch character by character and eventually after some training generate entirely new sentences that sometimes make some sense :)

[Character Sequence Memorization Demo](http://cs.stanford.edu/people/karpathy/recurrentjs)

The same demo as above implemented in Julia can be found in example/example.jl

## Example code

The core of the library is a **Graph** structure which maintains the links between matrices and how they are related th Here is how you would implement a simple Neural Network layer:

```julia
# T.B.D.
Will added soon ... I hope :)
```

To construct and train an LSTM for example, you would proceed as follows:

```julia
# T.B.D.
Will added soon ...
```

## Warning: Beta
Tests using the example from Karpathy's library show similar results but the code needs to be more thoroughly tested.

## License
MIT
