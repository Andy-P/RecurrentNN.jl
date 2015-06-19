# gated recurrent units

type GRULayer # a single layer of a multi-layer RNN
        # update gate params
        wux::NNMatrix
        wuh::NNMatrix
        bu::NNMatrix
        # reset gate params
        wrx::NNMatrix
        wrh::NNMatrix
        br::NNMatrix
        # candidate gate params
        wcx::NNMatrix
        wch::NNMatrix
        bc::NNMatrix
    function GRULayer(prevsize::Int, hiddensize::Int, std::Float64)
        ###  gate parameters ###
        # cell's input gate params
        wux = randNNMat(hiddensize, prevsize, std)
        wuh = randNNMat(hiddensize, hiddensize, std)
        bu = NNMatrix(hiddensize, 1, zeros(hiddensize,1), zeros(hiddensize,1))

        # cell's forget gate parameters
        wrx = randNNMat(hiddensize, prevsize, std)
        wrh = randNNMat(hiddensize, hiddensize, std)
        br = NNMatrix(hiddensize, 1, zeros(hiddensize,1), zeros(hiddensize,1))

        # cell's out gate parameters
        wcx = randNNMat(hiddensize, prevsize, std)
        wch = randNNMat(hiddensize, hiddensize, std)
        bc = NNMatrix(hiddensize, 1, zeros(hiddensize,1), zeros(hiddensize,1))

        new(wux,wuh,bu,　wrx,wrh,br,　wcx,wch,bc)
    end
end

type GRU <: Model
    hdlayers::Array{GRULayer,1} # holds variable number of hidden layers
    whd::NNMatrix # hidden to decoder weights & gradients
    bd::NNMatrix  # bias of hidden to decoder layer
    matrices::Array{NNMatrix,1} # used by solver - holds references to each of matrices in model
    hiddensizes::Array{Int,1}
    function GRU(inputsize::Int, hiddensizes::Array{Int,1}, outputsize::Int, std::Float64=0.08)
        hdlayers = Array(GRULayer, length(hiddensizes))
        matrices = Array(NNMatrix, 0)
        for d in 1:length(hiddensizes)
            prevsize = d == 1? inputsize : hiddensizes[d-1]
            layer = GRULayer(prevsize, hiddensizes[d], std)
            hdlayers[d] = layer
            # update gate
            push!(matrices, layer.wux)
            push!(matrices, layer.wuh)
            push!(matrices, layer.bu)
            # reset gate
            push!(matrices, layer.wrx)
            push!(matrices, layer.wrh)
            push!(matrices, layer.br)
            # candidate gate
            push!(matrices, layer.wcx)
            push!(matrices, layer.wch)
            push!(matrices, layer.bc)
        end
        whd = randNNMat(outputsize, hiddensizes[end], std)
        bd = NNMatrix(outputsize, 1, zeros(outputsize,1), zeros(outputsize,1))
        push!(matrices, whd) # bias of hidden to hidden
        push!(matrices, bd) # bias of hidden to hidden
#         println("bh.w size($(size(bd.w)))")
        new(hdlayers, whd, bd, matrices, hiddensizes)
    end
end

function forwardprop(g::Graph, model::GRU, x, prev)

    # forward prop for a single tick of GRU
    # g is graph to append ops to
    # model contains GRU parameters
    # x is 1D column vector with observation
    # prev is a struct containing hidden and cell from previous iteration

    prevhd, _ = prev # extract previous hidden and cell states from the tuple
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
        # update gate parameters
        wux = model.hdlayers[d].wux
        wuh = model.hdlayers[d].wuh
        bu = model.hdlayers[d].bu
        # reset gate parameters
        wrx = model.hdlayers[d].wrx
        wrh = model.hdlayers[d].wrh
        br = model.hdlayers[d].br
        # candidate parameters
        wcx = model.hdlayers[d].wcx
        wch = model.hdlayers[d].wch
        bc = model.hdlayers[d].bc

        # update gate
        h0 = mul(g, wux, input)
        h1 = mul(g, wuh, hdprev)
        updategate = sigmoid(g, add(g, add(g, h0,h1), bu))

        # reset gate
        h2 = mul(g, wrx, input)
        h3 = mul(g, wrh, hdprev)
        resetgate = sigmoid(g, add(g, add(g, h2, h3), br))

        # candidate
        p1 = mul(g, wcx, input)
        p2 = mul(g, wch, eltmul(g, resetgate, hdprev))
        candidate = tanh(g, add(g, add(g, p1, p2), bc))

        # compute hidden state
        ones = onesNNMat(updategate.n, updategate.d)
        oneminusupdate = add(g, mul(g, updategate, -1.0), 1.0)
        hidden_d = add(g, eltmul(g, oneminusupdate, hdprev),
                          eltmul(g, updategate, candidate))

        push!(hidden,hidden_d)
    end

    # one decoder to outputs at end
    output = add(g, mul(g, model.whd, hidden[end]),model.bd)

    # return cell memory, hidden representation and output
    return hidden, output
end

