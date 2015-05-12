type RNNLayer # a single layer of a multi-layer RNN
    wxh::NNMatrix # input (X or prev layer) to hidden weights & gradients
    whh::NNMatrix # hidden to hidden weights & gradients
    bhh::NNMatrix # bias of hidden to hidden
    function RNNLayer(prevsize::Int, hiddensize::Int, std::Float64)
        wxh = randNNMat(hiddensize, prevsize, std)
        whh = randNNMat(hiddensize, hiddensize, std)
        wbh = NNMatrix(hiddensize, 1, zeros(hiddensize,1), zeros(hiddensize,1))
        new(wxh, whh, wbh)
    end
end

type RNN <: Model
    hdlayers::Array{RNNLayer,1} # holds variable number of hidden layers
    whd::NNMatrix # hidden to decoder weights & gradients
    bd::NNMatrix  # bias of hidden to decoder layer
    matrices::Array{NNMatrix,1} # used by solver - holds references to each of matrices in model
    hiddensizes::Array{Int,1}
    function RNN(inputsize::Int, hiddensizes::Array{Int,1}, outputsize::Int, std::Float64=0.08)
        hdlayers = Array(RNNLayer, length(hiddensizes))
        matrices = Array(NNMatrix, 0)
        for d in 1:length(hiddensizes)
            prevsize = d == 1? inputsize : hiddensizes[d-1]
            layer = RNNLayer(prevsize, hiddensizes[d], std)
            hdlayers[d] = layer
            # store a reference to all the matrices in each
            push!(matrices, layer.wxh) # input (X or prev layer) to hidden weights & gradients
            push!(matrices, layer.whh) # hidden to hidden weights & gradients
            push!(matrices, layer.bhh) # bias of hidden to hidden
        end
        whd = randNNMat(outputsize, hiddensizes[end], std)
        bd = NNMatrix(outputsize, 1, zeros(outputsize,1), zeros(outputsize,1))

        # store a reference to these matrices
        push!(matrices, whd)
        push!(matrices, bd)
#         println("bh.w size($(size(bd.w)))")
        new(hdlayers, whd, bd, matrices, hiddensizes)
    end
end

function forwardprop(g::Graph, model::RNN, x, prev)

    # forward prop for a single tick of RNN
    # G is graph to append ops to
    # model contains RNN parameters
    # x is 1D column vector with observation
    # prev is a struct containing hidden activations and output from last step

    prevhd, _ = prev # extract previous hidden from the tuple
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

    # return hidden representation and output
    return hidden, output
end
