type RNNLayer # a single layer of a multi-layer RNN
    wxh::NNMatrix # input (X) to hidden weights & gradients
    whh::NNMatrix # hidden to hidden weights & gradients
    bhh::NNMatrix # bias of hidden to hidden
    function RNNLayer(prevsize::Int, hiddensize::Int, std::FloatingPoint)
        wxh = randNNMat(prevsize, hiddensize, std)
        whh = randNNMat(prevsize, hiddensize, std)
        wbh = NNMatrix(hiddensize, 1, zeros(hiddensize), zeros(hiddensize))
        new(wxh, whh, wbh)
    end
end

type RNN <: Model
    hdlayers::Array{RNNLayer,1} # holds variable number of hidden layers
    whd::NNMatrix # hidden to decoder weights & gradients
    bd::NNMatrix  # bias of hidden to decoder layer
    function RNN(inputsize::Int, hiddensizes::Array{Int,1}, outputsize::Int, std::FloatingPoint=0.08)
        hdlayers = Array(RNNLayer, length(hiddensizes))
        for d in 1:length(hiddensizes)
            prevsize = d == 1? inputsize : hiddensizes[d-1]
            layer = RNNLayer(prevsize, hiddensizes[d], std)
            hdlayers[d] = layer
        end
        whd = randNNMat(outputsize, hiddensizes[end], std)
        bd = NNMatrix(outputsize, 1, zeros(outputsize), zeros(outputsize))
        new(hdlayers, whd, bd)
    end
end
