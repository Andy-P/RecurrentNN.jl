type MixtureDensityNetwork
    n::Int
    α::Vector
    μ::Vector
    σ²::Vector
    function MixtureDensityNetwork(n::Int)
        new(n,zeros(n),zeros(n),zeros(n))
    end
end

function mixturemodel(m::NNMatrix)




end


function Base.show(io::IO, mdn::MixtureDensityNetwork)
    @printf(io, "Mixture Density Network\n")
    @printf(io, " * n:  %d\n", mdn.n)
    println(io, " * α:  $(mdn.α)")
    println(io, " * μ:  $(mdn.μ)")
    println(io, " * σ²: $(mdn.σ²)")
    return
end
