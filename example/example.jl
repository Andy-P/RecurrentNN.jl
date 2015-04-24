using RecurrentNN
reload("RecurrentNN.jl")
# # global settings
# const generator = "rnn" # can be 'rnn' or 'lstm'
const generator = "lstm" # can be 'rnn' or 'lstm'
const hiddensizes = [20,20] # list of sizes of hidden layers
const lettersize = 5 # size of letter embeddings

# optimization
const regc = 0.000001 # L2 regularization strength
const learning_rate = 0.001 # learning rate for rnn
const clipval = 5.0 # clip gradients at this value

function initVocab(inpath::String)

    f = open(inpath,"r")
    sents = [string(l[1:end-2]) for l in readlines(f)] #split(str,"\r\n") # array of sentences
    str = ""
    for s in sents str = "$str $(s[1:end-1])" end
    vocab = sort(setdiff(unique(str),['\r','\n'])) # unique characters in data
#     sents = [string(l[1:end-2]) for l in readlines(f)] #split(str,"\r\n") # array of sentences
    inputsize = length(vocab) + 1 # 1 additional token (zero) in used for beginning and end tokens
    outputsize = length(vocab) + 1
    epochsize = length(sents) # nmber of sentence in sample

    # build char <-> index lookups
    letterToIndex =  Dict{Char,Int}()
    indexToLetter =  Dict{Int,Char}()
    [letterToIndex[vocab[i]] = i for i in 1:length(vocab)]
    [indexToLetter[i] = vocab[i] for i in 1:length(vocab)]

    return sents, vocab, letterToIndex, indexToLetter, inputsize, outputsize, epochsize
end

function initModel(inputsize::Int, lettersize::Int, hiddensizes::Array{Int,1},outputsize::Int)
    wil = RecurrentNN.randNNMat(inputsize,lettersize,.008)
    nn = generator == "rnn"? RecurrentNN.RNN(lettersize,hiddensizes,outputsize):
            RecurrentNN.LSTM(lettersize,hiddensizes,outputsize)
#     println((typeof(wil),typeof(nn)))
    return wil, nn
end

#########################################
#          initialize the model         #
#########################################

solver = RecurrentNN.Solver() # RMSProp optimizer

# init the text source
sents, vocab, letterToIndex, indexToLetter, inputsize, outputsize, epochsize =
    initVocab(joinpath(dirname(@__FILE__),"samples.txt"))

# init the rnn/lstm
wil, model = initModel(inputsize, lettersize, hiddensizes, outputsize)

tickiter = 0

pplcurve = Array(FloatingPoint,0) # track perplexity
pplgraph = Dict{Int,FloatingPoint}() # track perplexity

#########################################
#             run the model             #
#########################################

function predictsentence(model::RecurrentNN.Model, sent::String)


end

function costfunc(model::RecurrentNN.Model, wil::RecurrentNN.NNMatrix, sent::String)

    # takes a model and a sentence and
    # calculates the loss. Also returns the Graph
    # object which can be used to do backprop
    n = length(sent)
    g = RecurrentNN.Graph()
    log2ppl = 0.0
    cost = 0.0
    prevhd   = Array(RecurrentNN.NNMatrix,0) # final hidden layer of the recurrent model after each forward step
    prevcell = Array(RecurrentNN.NNMatrix,0) # final cell output of the LSTM model after each forward step
    prevout  = RecurrentNN.NNMatrix(outputsize,1) # output of the recurrent model after each forward step
    prev = (prevhd, prevcell, prevout)
    for i= 0:length(sent)

#         println((i,n))
        # start and end tokens are zeros
        ix_source = i == 0 ? 0 : letterToIndex[sent[i]] # first step: start with START token
        ix_target = i == n ? 0 : letterToIndex[sent[i+1]] # last step: end with END token

        x = RecurrentNN.rowpluck(g, wil, ix_source+1)

        # returns a 2-tuple (RNN) or 3-tuples (LSTM). Last part is always output NNMatrix
        prev = RecurrentNN.forwardprop(g, model, x, prev)

        # set gradients into logprobabilities
        logprobs =  prev[end] # interpret output (last position in tuple) as logprobs
        probs = RecurrentNN.softmax(logprobs) # compute the softmax probabilities

#         println((2, i,ix_source,ix_target))
        log2ppl += -log2(probs.w[ix_target+1]) # accumulate base 2 log prob and do smoothing
        cost += -log(probs.w[ix_target+1])

        # write gradients into log probabilities
        logprobs.dw = probs.w;
        logprobs.dw[ix_target+1] -= 1
    end
    ppl = (log2ppl/(n))^2
    return g, ppl, cost
end

function tick(model::RecurrentNN.Model, wil::RecurrentNN.NNMatrix, sents::Array, solver::RecurrentNN.Solver, tickiter::Int, pplcurve::Array{FloatingPoint,1})

    # sample sentence fromd data
#     sent = sents[rand(1:length(sents))]
    sent = sents[rand(21:21)]
#     sent = sents[22]
#     println((i,sent))

    t1 = time_ns() # log start timestamp

    # evaluate cost function on a sentence
    g, ppl, cost = costfunc(model,wil, sent)

    # use built up graph of backprop functions to compute backprop (set .dw fields in matirices)
    for i = length(g.backprop):-1:1  g.backprop[i]() end

    # perform param update ( learning_rate, regc, clipval are global constants)
    solverstats = RecurrentNN.step(solver, model, learning_rate, regc, clipval)

    tm = (time_ns()-t1)/1e9

    push!(pplcurve, ppl) #keep track of perplexity

    tickiter += 1;
    if tickiter % 50 == 0
    # draw samples
#     for(var q=0;q<5;q++) {
#       var pred = predictSentence(model, true, sample_softmax_temperature);
#       var pred_div = '<div class="apred">'+pred+'</div>'
#       $('#samples').append(pred_div);
#     }
    end

    if tickiter % 10 == 0
    #     // draw argmax prediction
    #     $('#argmax').html('');
    #     var pred = predictSentence(model, false);
    #     var pred_div = '<div class="apred">'+pred+'</div>'
    #     $('#argmax').append(pred_div);

        # keep track of perplexity
    #     $('#epoch').text('epoch: ' + (tick_iter/epoch_size).toFixed(2));
    #     $('#ppl').text('perplexity: ' + cost_struct.ppl.toFixed(2));
    #     $('#ticktime').text('forw/bwd time per example: ' + tick_time.toFixed(1) + 'ms');
        if tickiter % 50 == 0
            pplmedian = median(pplcurve)
            println("Perplexity = $(round(pplmedian,4)) @ $tickiter")
            pplgraph[tickiter] = pplmedian
            pplcurve = Array(FloatingPoint,0)
        end
    end
    return model, wil, solver, tickiter, pplcurve
end


tic()
interations = 60
for i = 1:interations
    model, wil, solver, tickiter, pplcurve  = tick(model, wil, sents, solver, tickiter, pplcurve)
end
toc()

# iter = sort(collect(keys(pplgraph)))
# plotdata = zeros(length(pplgraph),2)
# for i = 1:length(pplgraph)
#     println((float(i), round(pplgraph[iter[i]],3)))
# end

