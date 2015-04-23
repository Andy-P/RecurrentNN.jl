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

function forwardprop(g::Graph, model::RNN, hiddensizes::Array{Int,1}, x, prev)

    # forward prop for a single tick of RNN
    # G is graph to append ops to
    # model contains RNN parameters
    # x is 1D column vector with observation
    # prev is a struct containing hidden activations from last step

    hiddenprevs = Array(NNMatrix,0)
    if size(prevs,1) == 0
        for d = 1:size(hiddensizes,1)
            push!(hiddenprevs, NNMatrix(hiddensize[d],1))
        end
    else
      hidden_prevs = prev.h
    end

#     var hidden = [];
#     for(var d=0;d<hidden_sizes.length;d++) {

#       var input_vector = d === 0 ? x : hidden[d-1];
#       var hidden_prev = hidden_prevs[d];

#       var h0 = G.mul(model['Wxh'+d], input_vector);
#       var h1 = G.mul(model['Whh'+d], hidden_prev);
#       var hidden_d = G.relu(G.add(G.add(h0, h1), model['bhh'+d]));

#       hidden.push(hidden_d);
#     }

#     // one decoder to outputs at end
#     var output = G.add(G.mul(model['Whd'], hidden[hidden.length - 1]),model['bd']);

#     // return cell memory, hidden representation and output
#     return {'h':hidden, 'o' : output};
end

