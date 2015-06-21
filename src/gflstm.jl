# Gated-Feedback LSTM

type GFLSTMLayer # a single layer of a multi-layer RNN
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
        wgh::Array{NNMatrix, 1}
        wch::Array{NNMatrix, 1}
        wgx::Array{NNMatrix, 1}
        bg::Array{NNMatrix, 1}
        bc::NNMatrix
    function GFLSTMLayer(prevsize::Int, hiddensizes::Array{Int, 1}, layer::Int, std::Float64)
        ###  gate parameters ###
        # cell's input gate params
        hiddensize = hiddensizes[layer]
        totalsize = sum(hiddensizes)
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
        layers = length(hiddensizes)
        wgx = Array(NNMatrix, layers)
        wgh = Array(NNMatrix, layers)
        wch = Array(NNMatrix, layers)
        bg = Array(NNMatrix, layers)
        for d in 1:layers
            wgh[d] = randNNMat(hiddensize, totalsize, std)
            wch[d] = randNNMat(hiddensize, hiddensize, std)
            wgx[d] = randNNMat(hiddensize, prevsize, std)
            bg[d] = NNMatrix(hiddensize, 1, zeros(hiddensize,1), zeros(hiddensize,1))
        end
        bc = NNMatrix(hiddensize, 1, zeros(hiddensize,1), zeros(hiddensize,1))

        new(wix,wih,bi,　wfx,wfh,bf,　wox,woh,bo,　wcx,wgh,wch,wgx,bg,bc)
    end
end

type GFLSTM <: Model
    hdlayers::Array{GFLSTMLayer,1} # holds variable number of hidden layers
    whd::NNMatrix # hidden to decoder weights & gradients
    bd::NNMatrix  # bias of hidden to decoder layer
    matrices::Array{NNMatrix,1} # used by solver - holds references to each of matrices in model
    hiddensizes::Array{Int, 1}
    function GFLSTM(inputsize::Int, hiddensizes::Array{Int, 1}, outputsize::Int, std::Float64=0.08)
        hdlayers = Array(GFLSTMLayer, length(hiddensizes))
        matrices = Array(NNMatrix, 0)
        for d in 1:length(hiddensizes)
            prevsize = d == 1? inputsize : hiddensizes[d-1]
            layer = GFLSTMLayer(prevsize, hiddensizes, d, std)
            hdlayers[d] = layer
            # input gate
            push!(matrices, layer.wix)
            push!(matrices, layer.wih)
            push!(matrices, layer.bi)
            # forget gate
            push!(matrices, layer.wfx)
            push!(matrices, layer.wfh)
            push!(matrices, layer.bf)
            # output gate
            push!(matrices, layer.wox)
            push!(matrices, layer.woh)
            push!(matrices, layer.bo)
            # cell params
            push!(matrices, layer.wcx)
            for d2 in 1:length(hiddensizes)
                push!(matrices, layer.wgh[d2])
                push!(matrices, layer.wch[d2])
                push!(matrices, layer.wgx[d2])
                push!(matrices, layer.bg[d2])
            end
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

function forwardprop(g::Graph, model::GFLSTM, x, prev)

    # forward prop for a single tick of Gated Feedback LSTM
    # g is graph to append ops to
    # model contains GFLSTM parameters
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
    hstar = concat(g, hiddenprevs...)
    layers = length(model.hiddensizes)
    for d in 1:layers
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
        wgh = model.hdlayers[d].wgh
        wgx = model.hdlayers[d].wgx
        wch = model.hdlayers[d].wch
        bg = model.hdlayers[d].bg
        bc = model.hdlayers[d].bc

        # input gate
        h0 = mul(g, wix, input)
        h1 = mul(g, wih, hdprev)
        inputgate = sigmoid(g, add(g, h0, h1, bi))

        # forget gate
        h2 = mul(g, wfx, input)
        h3 = mul(g, wfh, hdprev)
        forgetgate = sigmoid(g, add(g, h2, h3, bf))

        # output gate
        h4 =mul(g, wox, input)
        h5 = mul(g, woh, hdprev)
        outputgate = sigmoid(g, add(g, h4, h5, bo))

        # global reset gates
        gr = Array(NNMatrix, layers)
        @inbounds for gd in 1:layers
            hg = mul(g, wgx[gd], input)
            hu = mul(g, wgh[gd], hstar)
            gr[gd] = sigmoid(g, add(g, hg, hu, bg[gd]))
        end

        # write operation on cells
        h6 = mul(g, wcx, input)
        h = Array(NNMatrix, layers)
        @inbounds for hd in 1:layers
            hi = mul(g, wch[hd], hiddenprevs[hd])
            h[hd] = eltmul(g, gr[hd], hi)
        end
        cellwrite = tanh(g, add(g, h6, bc, h...))

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

