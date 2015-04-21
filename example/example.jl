using RecurrentNN
reload("RecurrentNN.jl")
# # global settings
const generator = "lstm" # can be 'rnn' or 'lstm'
const hiddensizes = [20,20] # list of sizes of hidden layers
const lettersize = 5 # size of letter embeddings

# optimization
regc = 0.000001 # L2 regularization strength
learning_rate = 0.01 # learning rate
clipval = 5.0 # clip gradients at this value

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

# function initModel()
# end

# function reinit()
# end

# sents, vocab, letterToIndex, indexToLetter, inputsize, outputsize, epochsize =
#     initVocab(joinpath(dirname(@__FILE__),"samples.txt"))

# nn = RecurrentNN.RNNLayer(20,10,.008)
# nn = RecurrentNN.RNN(120,[10,10,10],30)
# out = RecurrentNN.relu(grph,nnmat)

# out.w
# out.dw
# dw_before = copy(nnmat.dw)
# randn!(out.dw)
# grph.backprop[1]()

# nnmat.dw
# dw_before
grph = RecurrentNN.Graph()
nnmat = RecurrentNN.randNNMat(2,3,.008)
nnmat2 = RecurrentNN.randNNMat(3,2,.008)

out = RecurrentNN.mul(grph,nnmat, nnmat2)
randn!(out.dw)
out

nnmat
grph.backprop[1]()

nnmat
