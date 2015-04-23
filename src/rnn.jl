type RNNLayer # a single layer of a multi-layer RNN
    wxh::NNMatrix # input (X or prev layer) to hidden weights & gradients
    whh::NNMatrix # hidden to hidden weights & gradients
    bhh::NNMatrix # bias of hidden to hidden
    function RNNLayer(prevsize::Int, hiddensize::Int, std::FloatingPoint)
        wxh = randNNMat(hiddensize, prevsize, std)
        whh = randNNMat(hiddensize, hiddensize, std)
        wbh = NNMatrix(hiddensize, 1, zeros(hiddensize), zeros(hiddensize))
        new(wxh, whh, wbh)
    end
end

type RNN <: Model
    hdlayers::Array{RNNLayer,1} # holds variable number of hidden layers
    whd::NNMatrix # hidden to decoder weights & gradients
    bd::NNMatrix  # bias of hidden to decoder layer
    hiddensizes::Array{Int,1}
    function RNN(inputsize::Int, hiddensizes::Array{Int,1}, outputsize::Int, std::FloatingPoint=0.08)
        hdlayers = Array(RNNLayer, length(hiddensizes))
        for d in 1:length(hiddensizes)
            prevsize = d == 1? inputsize : hiddensizes[d-1]
            layer = RNNLayer(prevsize, hiddensizes[d], std)
            hdlayers[d] = layer
        end
        whd = randNNMat(outputsize, hiddensizes[end], std)
        bd = NNMatrix(outputsize, 1, zeros(outputsize,1), zeros(outputsize,1))
#         println("bh.w size($(size(bd.w)))")
        new(hdlayers, whd, bd, hiddensizes)
    end
end

# this function does not exist in Andrej Karpathy's version
# due to difference between JavaScript. it has been added
# to allow the solver to collect and adjust all of the
# matrices' weights without knowing the structure of any given model
function collectNNMat(model::RNN)
    modelNNMats = Array(NNMatrix,0)
    for d in 1:length(model.hiddensizes)
        layer = model.hdlayers[d]
        push!(modelNNMats, layer.wxh) # input (X or prev layer) to hidden weights & gradients
        push!(modelNNMats, layer.whh) # hidden to hidden weights & gradients
        push!(modelNNMats, layer.bhh) # bias of hidden to hidden
    end
    push!(modelNNMats, model.whd) # bias of hidden to hidden
    push!(modelNNMats, model.bd) # bias of hidden to hidden
    return modelNNMats
end

function forwardprop(g::Graph, model::RNN, x, prevhd, prevout)

    # forward prop for a single tick of RNN
    # G is graph to append ops to
    # model contains RNN parameters
    # x is 1D column vector with observation
    # prev is a struct containing hidden activations from last step

    hiddenprevs = Array(NNMatrix,0)
    if length(prevhd) == 0
        for hdsize in model.hiddensizes
            push!(hiddenprevs, NNMatrix(hdsize,1))
        end
    else
      hiddenprevs = prevhd
    end

    hidden = Array(NNMatrix,0)
    for d in 1:length(model.hiddensizes) # for each hidden layer

        input = d == 1 ? x : hidden[d-1]
        hdprev = hiddenprevs[d]
        wxh = model.hdlayers[d].wxh
        whh = model.hdlayers[d].whh
        bhh = model.hdlayers[d].bhh

#         println((d,size(wxh.w),(size(input.w))))

        h0 = mul(g, wxh, input)
        h1 = mul(g, whh, hdprev)
#         println((typeof(h0),typeof(h1),typeof(bhh)))
        hidden_d = relu(g, add(g, add(g, h0, h1), bhh))

        push!(hidden,hidden_d)
    end

    # one decoder to outputs at end
#     println((size(model.whd.w), size(hidden[end].w), size(model.bd.w)))
    output = add(g, mul(g, model.whd, hidden[end]),model.bd)

    # return cell memory, hidden representation and output
    return hidden, output
end
