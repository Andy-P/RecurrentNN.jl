type RNNLayer # a single layer of a multi-layer RNN
    wxh::NNMatrix # input (X or prev layer) to hidden weights & gradients
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
    hiddensizes::{Int,1}
    function RNN(inputsize::Int, hiddensizes::Array{Int,1}, outputsize::Int, std::FloatingPoint=0.08)
        hdlayers = Array(RNNLayer, length(hiddensizes))
        for d in 1:length(hiddensizes)
            prevsize = d == 1? inputsize : hiddensizes[d-1]
            layer = RNNLayer(prevsize, hiddensizes[d], std)
            hdlayers[d] = layer
        end
        whd = randNNMat(outputsize, hiddensizes[end], std)
        bd = NNMatrix(outputsize, 1, zeros(outputsize), zeros(outputsize))
        new(hdlayers, whd, bd, hiddensizes)
    end
end

function forwardprop(g::Graph, model::RNN, x, prevhd, prevout)

    # forward prop for a single tick of RNN
    # G is graph to append ops to
    # model contains RNN parameters
    # x is 1D column vector with observation
    # prev is a struct containing hidden activations from last step

    hiddenprevs = Array(NNMatrix,0)
    if size(prevs,1) == 0
        for hdsize in model.hiddensizes
            push!(hiddenprevs, NNMatrix(hdsize,1))
        end
    else
      hiddenprevs = prevhd
    end

    hidden = Array(NNMatrix,0)
    for d in 0:length(hiddensizes) # for each hidden layer

        input = d == 0 ? x : hidden[d]
        hdprev = hiddenprevs[d]
        wxh = model.hdlayers[d].wxh
        whh = model.hdlayers[d].whh
        bhh = model.hdlayers[d].bhh

        h0 = mul(g, wxh, input)
        h1 = mul(g, whh, hdprev)
        hidden_d = relu(add(g, add(g, h0, h1), bhh))

        push!(hidden,hidden_d)
    end

    # one decoder to outputs at end
    output = add(g, mul(g, model.whd, hidden[end]),model.bd)

    # return cell memory, hidden representation and output
    return hidden, output
end

