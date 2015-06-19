using RecurrentNN
# reload("RecurrentNN.jl")

# # global settings
# const generator = "rnn" # can be 'rnn' or 'lstm'
srand(12345)
const generator = "gru" # can be 'rnn' or 'lstm' or 'gru'
const hiddensizes = [100,100]
const lettersize = 7 # size of letter embeddings

# optimization
const regc = 0.000001 # L2 regularization strength
const learning_rate = 0.001 # Initial learning rate for lstm
const clipval = 5.0 # clip gradients at this value

function initVocab(inpath::String)

    f = open(inpath,"r")
    sents = [string(l[1:end-1]) for l in readlines(f)] # split(str,"\r\n") # array of sentences
    str = ""
    for i in 1:length(sents)
        s = sents[i]
        if contains(s,"\r")
            idx = findfirst(s,'\r')
            s = s[1:idx-1]
            sents[i] = s
        end
        str = "$str $(s[1:end-1])"
    end
    vocab = sort(setdiff(unique(str),['\r','\n'])) # unique characters in data
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
    wil = randNNMat(inputsize,lettersize,0.08)
    nn = if generator == "rnn"
        RNN(lettersize,hiddensizes,outputsize)
    elseif generator == "gru"
        GRU(lettersize,hiddensizes,outputsize)
    else
        LSTM(lettersize,hiddensizes,outputsize)
    end
    # println((typeof(wil),typeof(nn)))
    return wil, nn
end

#########################################
#          initialize the model         #
#########################################

solver = Solver() # RMSProp optimizer

# init the text source
sents, vocab, letterToIndex, indexToLetter, inputsize, outputsize, epochsize =
    initVocab(joinpath(dirname(@__FILE__),"samples.txt"))

# init the rnn/lstm
wil, model = initModel(inputsize, lettersize, hiddensizes, outputsize)

tickiter = 0
pplmedian = Inf # perplexity will slowly move towards 0
pplcurve = Array(FloatingPoint,0) # track perplexity

#########################################
#             run the model             #
#########################################

function predictsentence(model::Model, wil::NNMatrix, samplei::Bool=false, temp::FloatingPoint=1.0)

    g = Graph(false) # backprop not needed
    s = ""
    max_chars_gen = 100
    prevhd   = Array(NNMatrix,0) # final hidden layer of the recurrent model after each forward step
    prevcell = Array(NNMatrix,0) # final cell output of the LSTM model after each forward step
    prevout  = NNMatrix(outputsize,1) # output of the recurrent model after each forward step
    prev = (prevhd, prevcell, prevout)
    cnt = 0
    while cnt < 100

        # RNN tick
        ix = length(s) == 0 ? 0 : letterToIndex[s[end]]
        x = rowpluck(g, wil, ix+1) # get letter's embedding (vector)

        # returns a 2-tuple (RNN) or 3-tuple (LSTM). Last part is always outputNNMatrix
        prev = forwardprop(g, model, x, prev)

        # sample predicted letter
        logprobs =  prev[end] # interpret output (last position in tuple) as logprobs
        if temp != 1.0 && samplei
            # scale log probabilities by temperature and renormalize
            # if temp is high, logprobs will go towards zero
            # and the softmax outputs will be more diffuse. if temp is
            # very low, the softmax outputs will be more peaky
            logprobs.w ./= temp
        end

        probs = softmax(logprobs)
        rndIX = rand()
        ix = samplei? findfirst(p-> p >= rndIX, cumsum(probs.w)) : indmax(probs.w);

        # break on out to the other side!
        if ix-1 == 0 break end # END token predicted, break out
        if length(s) > max_chars_gen break end # something is wrong

        letter = indexToLetter[ix-1]
        s = "$(s)$(letter)"
        cnt = cnt + 1
    end

    return s
end

function costfunc(model:: Model, wil::NNMatrix, sent::String)

    # takes a model and a sentence and
    # calculates the loss. Also returns the Graph
    # object which can be used to do backprop
    n = length(sent)
    g =  Graph()
    log2ppl = 0.0
    cost = 0.0
    prevhd   = Array(NNMatrix,0) # final hidden layer of the recurrent model after each forward step
    prevcell = Array(NNMatrix,0) # final cell output of the LSTM model after each forward step
    prevout  = NNMatrix(outputsize,1) # output of the recurrent model after each forward step
    prev = (prevhd, prevcell, prevout)
    for i= 0:length(sent)

        # start and end tokens are zeros
        ix_source = i == 0 ? 0 : letterToIndex[sent[i]] # first step: start with START token
        ix_target = i == n ? 0 : letterToIndex[sent[i+1]] # last step: end with END token

        # get the letter embbeding of the char
        x = rowpluck(g, wil, ix_source+1)

        # returns a 2-tuple (RNN) or 3-tuples (LSTM). Last part of tuple is always the outputNNMatrix
        prev = forwardprop(g, model, x, prev)

        # set gradients into logprobabilities
        logprobs =  prev[end] # interpret output (last position in tuple) as logprobs
        probs = softmax(logprobs) # compute the softmax probabilities

        log2ppl += -log2(probs.w[ix_target+1]) # accumulate base 2 log prob and do smoothing
        cost += -log(probs.w[ix_target+1])

        # write gradients into log probabilities
        logprobs.dw = probs.w
        logprobs.dw[ix_target+1] -= 1
    end
    ppl = (log2ppl/(n))^2
    return g, ppl, cost
end

function tick(model::Model, wil::NNMatrix, sents::Array, solver::Solver, tickiter::Int, pplcurve::Array{FloatingPoint,1}, pplmedian)

    # sample sentence fromd data
#     sent = sents[rand(1:21)] # use this if just kicking tires (faster)
    sent = sents[rand(1:length(sents))] # switch to this for a proper model (1-3hrs)

    t1 = time_ns() # log start timestamp

    # evaluate cost function on a sentence
    g, ppl, cost = costfunc(model,wil, sent)

    # use built up graph of backprop functions to compute backprop
    # i.e. set .dw fields in matrices
    backprop(g)

    # perform param update ( learning_rate, regc, clipval are global constants)
    solverstats = step(solver, model, learning_rate, regc, clipval)

    tm = (time_ns()-t1)/1e9

    push!(pplcurve, ppl) #keep track of perplexity

    tickiter += 1;

    if tickiter % 50 == 0 # output sample sentences every Xth iteration
        pplmedian = round(median(pplcurve),4)
        println("Perplexity = $(pplmedian) @ $tickiter")
        push!(pplcurve, pplmedian)
        # draw samples to see how we're doing
        for i =1:2
            pred = predictsentence(model, wil, true, 0.7)
            println("   $(i). $pred ")
        end
        # greedy argmax (i.e if we were to select the most likely letter at each point)
        println("   Argmax: $(predictsentence(model, wil, false))")
    end

    return model, wil, solver, tickiter, pplcurve, pplmedian
end

maxIter = 100 # make this about 100_000 to run full model
trgppl = 1.1 # stop if this perplexity score is reached
@time while tickiter < maxIter && pplmedian > trgppl
    model, wil, solver, tickiter, pplcurve, pplmedian = tick(model, wil, sents, solver, tickiter, pplcurve, pplmedian)
end

# Profile.print(format=:flat)
#using ProfileView
#ProfileView.view()
#readline()
