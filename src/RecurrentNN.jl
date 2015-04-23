module RecurrentNN
import Base.tanh

include("recurrent.jl")
include("graph.jl")
include("solver.jl")
include("rnn.jl")
include("lstm.jl")


# export NNMatrix
# package code goes here

end # module
