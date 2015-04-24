type LSTMLayer # a single layer of a multi-layer RNN
        # cell's input gate params
        wix::NNMatrix
        wih::NNMatrix
        bi::NNMatrix
        # cell's forget gate parameters
        wfx::NNMatrix
        wfh::NNMatrix
        bf::NNMatrix
        # cell's out gate parameters
        wox::NNMatrix
        woh::NNMatrix
        bo::NNMatrix
        # cell's write parameters
        wcx::NNMatrix
        wch::NNMatrix
        bc::NNMatrix
    function LSTMLayer(prevsize::Int, hiddensize::Int, std::FloatingPoint)
        ###  gate parameters ###
        # cell's input gate params
        wix = randNNMat(hiddensize, prevsize, std)
        wih = randNNMat(hiddensize, hiddensize, std)
        bi = NNMatrix(hiddensize, 1, zeros(hiddensize,1), zeros(hiddensize,1))

        # cell's forget gate parameters
        wfx = randNNMat(hiddensize, prevsize, std)
        wfh = randNNMat(hiddensize, hiddensize, std)
        bf = NNMatrix(hiddensize, 1, zeros(hiddensize,1), zeros(hiddensize,1))

        # cell's out gate parameters
        wox = randNNMat(hiddensize, prevsize, std)
        woh = randNNMat(hiddensize, hiddensize, std)
        bo = NNMatrix(hiddensize, 1, zeros(hiddensize,1), zeros(hiddensize,1))

        ### cell's write parameters ###
        wcx = randNNMat(hiddensize, prevsize, std)
        wch = randNNMat(hiddensize, hiddensize, std)
        bc = NNMatrix(hiddensize, 1, zeros(hiddensize,1), zeros(hiddensize,1))

        new(wix,wih,bi,　wfx,wfh,bf,　wox,woh,bo,　wcx,wch,bc)
    end
end

type LSTM <: Model
    hdlayers::Array{LSTMLayer,1} # holds variable number of hidden layers
    whd::NNMatrix # hidden to decoder weights & gradients
    bd::NNMatrix  # bias of hidden to decoder layer
    hiddensizes::Array{Int,1}
    function LSTM(inputsize::Int, hiddensizes::Array{Int,1}, outputsize::Int, std::FloatingPoint=0.08)
        hdlayers = Array(LSTMLayer, length(hiddensizes))
        for d in 1:length(hiddensizes)
            prevsize = d == 1? inputsize : hiddensizes[d-1]
            layer = LSTMLayer(prevsize, hiddensizes[d], std)
            hdlayers[d] = layer
        end
        whd = randNNMat(outputsize, hiddensizes[end], std)
        bd = NNMatrix(outputsize, 1, zeros(outputsize,1), zeros(outputsize,1))
#         println("bh.w size($(size(bd.w)))")
        new(hdlayers, whd, bd, hiddensizes)
    end
end

# this function does not exist in Andrej Karpathy's version.
#  It has been added to allow the solver to collect and adjust all of the
# matrices' weights without knowing the structure of any given model.
function collectNNMat(model::LSTM)
    modelNNMats = Array(NNMatrix,0)
    for d in 1:length(model.hiddensizes)
        layer = model.hdlayers[d]
        # input gate
        push!(modelNNMats, layer.wix)
        push!(modelNNMats, layer.wih)
        push!(modelNNMats, layer.bi)
        # forget gate
        push!(modelNNMats, layer.wfx)
        push!(modelNNMats, layer.wfh)
        push!(modelNNMats, layer.bf)
        # output gate
        push!(modelNNMats, layer.wox)
        push!(modelNNMats, layer.woh)
        push!(modelNNMats, layer.bo)
        # cell params
        push!(modelNNMats, layer.wcx)
        push!(modelNNMats, layer.wch)
        push!(modelNNMats, layer.bc)
    end
    push!(modelNNMats, model.whd) # bias of hidden to hidden
    push!(modelNNMats, model.bd) # bias of hidden to hidden
    return modelNNMats
end

function forwardprop(g::Graph, model::LSTM, x, prev)

    # forward prop for a single tick of LSTM
    # g is graph to append ops to
    # model contains LSTM parameters
    # x is 1D column vector with observation
    # prev is a struct containing hidden and cell from previous iteration

    prevhd, prevcell, _ = prev # extract previous hidden and cell states from the tuple
    hiddenprevs = Array(NNMatrix,0)
    cellprevs = Array(NNMatrix,0)
    if length(prevhd) == 0
        for hdsize in model.hiddensizes
            push!(hiddenprevs, NNMatrix(hdsize,1))
            push!(cellprevs, NNMatrix(hdsize,1))
        end
    else
        hiddenprevs = prevhd
        cellprevs = prevcell
    end

    hidden = Array(NNMatrix,0)
    cell = Array(NNMatrix,0)
    for d in 1:length(model.hiddensizes) # for each hidden layer

        input = d == 1 ? x : hidden[d-1]
        hdprev = hiddenprevs[d]
        cellprev = cellprevs[d]
        # cell's input gate params
        wix = model.hdlayers[d].wix
        wih = model.hdlayers[d].wih
        bi = model.hdlayers[d].bi
        # cell's forget gate parameters
        wfx = model.hdlayers[d].wfx
        wfh = model.hdlayers[d].wfh
        bf = model.hdlayers[d].bf
        # cell's out gate parameters
        wox = model.hdlayers[d].wox
        woh = model.hdlayers[d].woh
        bo = model.hdlayers[d].bo
        # cell's write parameters
        wcx = model.hdlayers[d].wcx
        wch = model.hdlayers[d].wch
        bc = model.hdlayers[d].bc

        # input gate
        h0 = mul(g, wix, input)
        h1 = mul(g, wih, hdprev)
        inputgate = sigmoid(g, add(g, add(g, h0,h1), bi))

        # forget gate
        h2 = mul(g, wfx, input)
        h3 = mul(g, wfh, hdprev)
        forgetgate = sigmoid(g, add(g, add(g, h2, h3), bf))

        # output gate
        h4 =mul(g, wox, input)
        h5 = mul(g, woh, hdprev)
        outputgate = sigmoid(g, add(g, add(g, h4, h5), bo))

        # write operation on cells
        h6 = mul(g, wcx, input)
        h7 = mul(g, wch, hdprev)
        cellwrite = tanh(g, add(g, add(g, h6, h7),bc))

        # compute new cell activation
        retaincell = eltmul(g, forgetgate, cellprev) # what do we keep from cell
        writecell = eltmul(g, inputgate, cellwrite)  # what do we write to cell
        cell_d = add(g, retaincell, writecell)        # new cell contents

        # compute hidden state as gated, saturated cell activations
        hidden_d = eltmul(g, outputgate, tanh(g, cell_d))

        push!(hidden,hidden_d)
        push!(cell, cell_d)
    end

    # one decoder to outputs at end
    output = add(g, mul(g, model.whd, hidden[end]),model.bd)

    # return cell memory, hidden representation and output
    return hidden, cell, output
end

