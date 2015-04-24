type Solver
   decayrate::FloatingPoint
   smootheps::FloatingPoint
   stepcache::Array{NNMatrix,1}
   Solver() = new(0.999, 1e-8, Array(NNMatrix,0))
end

function step(solver::Solver, model::Model, stepsize::FloatingPoint, regc::FloatingPoint, clipval::FloatingPoint)

    # perform parameter update
    solverstats = Array(FloatingPoint,0)
    numclipped = 0
    numtot = 0

    # New function not in orginal recurrentjs. Needed to gather all matrices into on collection
    modelMatices = collectNNMat(model)

    # init stepcache if needed
    if length(solver.stepcache) == 0
         for m in modelMatices
            push!(solver.stepcache, NNMatrix(m.n, m.d))
        end
    end

      for k = 1:length(modelMatices)
          m = modelMatices[k] # mat ref
          s = solver.stepcache[k]
          for i = 1:m.n
            for j = 1:m.d

                # rmsprop adaptive learning rate
                mdwi = m.dw[i,j]
                s.w[i,j] = s.w[i,j] * solver.decayrate + (1.0 - solver.decayrate) * mdwi * mdwi;

                # gradient clip
                if mdwi > clipval
                    mdwi = clipval
                    numclipped += 1
                end

                if mdwi < -clipval
                    mdwi = -clipval
                    numclipped += 1
                end
                numtot += 1

                # update (and regularize)
                m.w[i,j] += - stepsize * mdwi / sqrt(s.w[i,j] + solver.smootheps) - regc * m.w[i,j]
                m.dw[i,j] = 0. # reset gradients for next iteration
            end
        end
    end
    solverstats　=  numclipped * 1.0 / numtot
    return solverstats
end
