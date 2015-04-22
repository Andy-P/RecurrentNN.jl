using RecurrentNN
reload("RecurrentNN.jl")
# # global settings
const generator = "rnn" # can be 'rnn' or 'lstm'
const hiddensizes = [20,20] # list of sizes of hidden layers
const lettersize = 5 # size of letter embeddings

# optimization
regc = 0.000001 # L2 regularization strength
learning_rate = 0.01 # learning rate
clipval = 5.20 # clip gradients at this value

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

function initModel(lettersize::Int, hiddensizes::Array{Int,1},outputsize::Int)
    wil = RecurrentNN.randNNMat(inputsize,lettersize,.008)
    rnn = RecurrentNN.RNN(lettersize,hiddensizes,outputsize)
    println((typeof(wil),typeof(rnn)))
    m = TextModel(wil,rnn)
    return m
end

# function reinit()
# end

sents, vocab, letterToIndex, indexToLetter, inputsize, outputsize, epochsize =
    initVocab(joinpath(dirname(@__FILE__),"samples.txt"))

model = initModel(lettersize, hiddensizes, outputsize)

# grph = RecurrentNN.Graph()
# nn = RecurrentNN.RNN(120,[10,10,10],30)
# nnmat = RecurrentNN.randNNMat(3,5,.008)
# out = RecurrentNN.rowpluck(grph,nnmat,3)

# out.w
# out.dw
# dw_before = copy(nnmat.dw)
# randn!(out.dw)
# grph.backprop[1]()

# nnmat.dw
# dw_before
