# Normal probability density function
const sqrt2PiInv = 1 / sqrt(2*π)
normalpdf(y, μ, σ) =  sqrt2PiInv * exp(-(y-μ)^2 / 2σ^2) /  σ

type MixtureDensityNetwork
    n::Int                    # number of gaussians
    π::Vector{AbstractFloat}  # model mix coeff
    μ::Vector{AbstractFloat}  # means
    σ::Vector{AbstractFloat}  # variances
    function MixtureDensityNetwork(n::Int, std::AbstractFloat=0.08)
        new(n, zeros(n), zeros(n), zeros(n))
    end
end


function updateCoeff!(mdn::MixtureDensityNetwork, m::NNMatrix)

    # calculate softmax to get mixture weights
    n = mdn.n
    maxval = maximum(m.w[1:n,1])
    mdn.π[:] = exp(m.w[1:n,1] - maxval)
    mdn.π[:] /= sum(mdn.π)
    for i = 1:n
        mdn.μ[i] = m.w[i+n,1] # means
        mdn.σ[i] = exp(m.w[i+n*2,1]) # variances
    end
    return
end


function calcGamma(mdn::MixtureDensityNetwork, y::AbstractFloat)
    n = mdn.n
    γ = zeros(n);
    Σγ  = 0.
    for i in 1:n
        γ[i] = mdn.π[i] * normalpdf(y, mdn.μ[i], mdn.σ[i])
        Σγ += γ[i] # sum of γ (used for normalization and log error calc)
    end
    γ /= Σγ    # normalized gamma
    ε = -log(Σγ)  # log error
    return γ, ε
end


function calcGradients!(mdn::MixtureDensityNetwork, m::NNMatrix, y::AbstractFloat)
    γ, ε = calcGamma(mdn, y)
    # calc and assign gradients
    n = mdn.n
    for i = 1:n
        μ = mdn.μ[i]
        σ = mdn.σ[i]
        m.dw[i]     =  mdn.π[i] - γ[i]        # ∂ε/∂m.π
        m.dw[i+n]   =  γ[i]*((μ-y)/(σ^2))     # ∂ε/∂m.μ
        m.dw[i+n*2] = -γ[i]*((y-μ)^2/(σ^2)-1) # dε/∂m.σ
    end
#     println((m.dw'))
    return ε
end

mean(mdn::MixtureDensityNetwork) = dot(mdn.π, mdn.μ)

expectation(mdn::MixtureDensityNetwork) = mdn.μ[indmax(mdn.π)]

function sample(mdn::MixtureDensityNetwork)
    Σπ = 0.
    r = rand()
    nDist = 0
    for i = 1:mdn.n
        Σπ += mdn.π[i]
        if Σπ >= r
            nDist = i
            break
        end
    end
    x = randn() * mdn.σ[nDist] + mdn.μ[nDist]
    return x
end

function Base.show(io::IO, mdn::MixtureDensityNetwork)
    @printf(io, "Mixture Density Network\n")
    @printf(io, " * n:  %d\n", mdn.n)
    println(io, " * π:  $(mdn.π)")
    println(io, " * μ:  $(mdn.μ)")
    println(io, " * σ:  $(mdn.σ)")
    return
end

