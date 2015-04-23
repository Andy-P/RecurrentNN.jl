using RecurrentNN
reload("RecurrentNN.jl")

# # global settings
const generator = "rnn" # can be 'rnn' or 'lstm'
const hiddensizes = [20,20] # list of sizes of hidden layers
const lettersize = 5 # size of letter embeddings

# optimization
const regc = 0.000001 # L2 regularization strength
const learning_rate = 0.01 # learning rate
const clipval = 5.20 # clip gradients at this value

type TextModel
    wil::RecurrentNN.NNMatrix  # character input to char-index layer
    model::RecurrentNN.Model
end

function initVocab(inpath::String)

    f = open(inpath,"r")
    str = readall(inpath)
    vocab = sort(setdiff(unique(str),['\n'])) # unique characters in data
    sents = split(str,'\n') # array of sentences
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
    nn = RecurrentNN.RNN(lettersize,hiddensizes,outputsize)
    println((typeof(wil),typeof(nn)))
    return TextModel(wil,nn)
end

#########################################
#          initialize the model         #
#########################################

solver = RecurrentNN.Solver() # RMSProp optimizer

# init the text source
sents, vocab, letterToIndex, indexToLetter, inputsize, outputsize, epochsize =
    initVocab(joinpath(dirname(@__FILE__),"samples.txt"))

# init the rnn/lstm
model = initModel(inputsize, lettersize, hiddensizes, outputsize)


#########################################
#             run the model             #
#########################################

function forwardIndex()

end

function predictsentence(model::Model, sent::String)


end


function costfunc(model::Model, sent::String)

    # takes a model and a sentence and
    # calculates the loss. Also returns the Graph
    # object which can be used to do backprop

    n = length(sent)
    g = Graph()
    log2ppl = 0.0
    cost = 0.0
    prevhd = Array(NNMatrix,0) # final hidden layer of the recurrent model after each forward step
    prevout = NNMatrix(outputsize,1) # output of the recurrent model after each forward step
    for i= 0:length(sent)
        c = i == 0? 0 : sent[i]
        println((i,c))

    # start and end tokens are zeros
    ix_source = i == 0 ? 0 : letterToIndex[sent[i]] # first step: start with START token
    ix_target = i == n ? 0 : letterToIndex[sent[i+1]] # last step: end with END token

    lh = forwardIndex(g, model, ix_source, prev);
    prevhd, prevout = lh

    # set gradients into logprobabilities
    logprobs = lh.o # interpret output as logprobs
    probs = R.softmax(logprobs) # compute the softmax probabilities

    log2ppl += -Math.log2(probs.w[ix_target]); // accumulate base 2 log prob and do smoothing
    cost += -Math.log(probs.w[ix_target]);

    # write gradients into log probabilities
    logprobs.dw = probs.w;
    logprobs.dw[ix_target] -= 1
    }
    var ppl = Math.pow(2, log2ppl / (n - 1));
    return {'G':G, 'ppl':ppl, 'cost':cost};

end

function costfunc(model::Model, sent::String)


end

sent = sents[20]
for i= 0:length(sent)
    c = i == 0? 0 : sent[i]
    println((i,c))
end
