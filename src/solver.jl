type Solver
   decayrate::FloatingPoint
   smootheps::FloatingPoint
   stepcache::Array{NNMatrix,1}
   Solver() = new(0.999, 1e-8, Array(NNMatrix,0))
end

function step(solver::Solver, model::Model, stepsize::FloatingPoint, regc::FloatingPoint, clipval::FloatingPoint)

    solverstats = Array(FloatingPoint,0)
    numclipped = 0
    numtot = 0


end
