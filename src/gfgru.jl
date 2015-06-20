# gated recurrent units

type GFGRULayer # a single layer of a multi-layer RNN
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
        wgh::Array{NNMatrix, 1}
        wch::Array{NNMatrix, 1}
        wgx::Array{NNMatrix, 1}
        bg::Array{NNMatrix, 1}
        bc::NNMatrix
    function GFGRULayer(prevsize::Int, hiddensizes::Array{Int, 1}, layer::Int, std::Float64)
        hiddensize = hiddensizes[layer]
        totalsize = sum(hiddensizes)
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

        new(wux,wuh,bu,　wrx,wrh,br,　wcx,wgh,wch,wgx,bg,bc)
    end
end

type GFGRU <: Model
    hdlayers::Array{GFGRULayer,1} # holds variable number of hidden layers
    whd::NNMatrix # hidden to decoder weights & gradients
    bd::NNMatrix  # bias of hidden to decoder layer
    matrices::Array{NNMatrix,1} # used by solver - holds references to each of matrices in model
    hiddensizes::Array{Int,1}
    function GFGRU(inputsize::Int, hiddensizes::Array{Int,1}, outputsize::Int, std::Float64=0.08)
        hdlayers = Array(GFGRULayer, length(hiddensizes))
        matrices = Array(NNMatrix, 0)
        for d in 1:length(hiddensizes)
            prevsize = d == 1? inputsize : hiddensizes[d-1]
            layer = GFGRULayer(prevsize, hiddensizes, d, std)
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

function forwardprop(g::Graph, model::GFGRU, x, prev)

    # forward prop for a single tick of GFGRU
    # g is graph to append ops to
    # model contains GFGRU parameters
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
    hstar = concat(g, hiddenprevs...)
    layers = length(model.hiddensizes)
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
        wgh = model.hdlayers[d].wgh
        wgx = model.hdlayers[d].wgx
        wch = model.hdlayers[d].wch
        bg = model.hdlayers[d].bg
        bc = model.hdlayers[d].bc

        # update gate
        h0 = mul(g, wux, input)
        h1 = mul(g, wuh, hdprev)
        updategate = sigmoid(g, add(g, h0, h1, bu))

        # reset gate
        h2 = mul(g, wrx, input)
        h3 = mul(g, wrh, hdprev)
        resetgate = sigmoid(g, add(g, h2, h3, br))

        # global reset gates
        gr = Array(NNMatrix, layers)
        @inbounds for gd in 1:layers
            hg = mul(g, wgx[gd], input)
            hu = mul(g, wgh[gd], hstar)
            gr[gd] = sigmoid(g, add(g, hg, hu, bg[gd]))
        end

        # candidate
        p1 = mul(g, wcx, input)
        h = Array(NNMatrix, layers)
        @inbounds for hd in 1:layers
            hi = mul(g, wch[hd], hiddenprevs[hd])
            h[hd] = eltmul(g, gr[hd], hi)
        end
        p2 = eltmul(g, resetgate, add(g, h...))
        candidate = tanh(g, add(g, p1, p2, bc))

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

