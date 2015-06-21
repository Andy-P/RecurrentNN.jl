module RecurrentNN
import Base.tanh

export Model, RNN, LSTM, GRU, GFLSTM, GFGRU
export NNMatrix, randNNMat, forwardprop, softmax, Solver, step
export Graph, backprop, rowpluck

include("recurrent.jl")
include("graph.jl")
include("solver.jl")
include("rnn.jl")
include("lstm.jl")
include("gru.jl")
include("gflstm.jl")
include("gfgru.jl")


end # module
