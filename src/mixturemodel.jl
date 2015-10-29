# Normal probability distribution
const sqrt2PiInv = 1 / sqrt(2*π)
normalpdf(y, μ, σ) =  sqrt2PiInv * exp(-(y-μ)^2 / 2σ^2) /  σ

type MixtureDensityNetwork
    n::Int     # number of models
    π::Vector{AbstractFloat}  # model mix coeff
    μ::Vector{AbstractFloat}  # means
    σ::Vector{AbstractFloat} # variances
    function MixtureDensityNetwork(n::Int, std::AbstractFloat=0.08)
        new(n, zeros(n), zeros(n), zeros(n))
    end
end

function updateCoeff!(mdn::MixtureDensityNetwork, m::NNMatrix)

    # calculate softmax to get mixture weights
    n = mdn.n
    maxval = maximum(m.w[1:n,1])
    mdn.π[:] = exp(m.w[1:n,1] - maxval)
    mdn.π[:] /= sum(mdn.π[:])

    for i = 1:n
        mdn.μ[i] = m.w[i+n,1]
        mdn.σ[i] = exp(m.w[i+n*2,1])
    end
    return mdn
end

function forward!(mdn::MixtureDensityNetwork, m::NNMatrix)



end


function Base.show(io::IO, mdn::MixtureDensityNetwork)
    @printf(io, "Mixture Density Network\n")
    @printf(io, " * n:  %d\n", mdn.n)
    println(io, " * π:  $(mdn.π)")
    println(io, " * μ:  $(mdn.μ)")
    println(io, " * σ:  $(mdn.σ)")
    return
end

