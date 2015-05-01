module RecurrentNN
import Base.tanh

export Model, RNN, LSTM
export NNMatrix, randNNMat, forwardprop, softmax, Solver, step
export Graph, rowpluck, resetposition!, dobackprop!, nobackprop!

include("recurrent.jl")
include("graph.jl")
include("solver.jl")
include("rnn.jl")
include("lstm.jl")


end # module
