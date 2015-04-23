type Graph
   backprop::Array{Function,1}
   Graph() = new(Array(Function,0))
end

function rowpluck(g::Graph, m::NNMatrix, ix::Int)
    # pluck a row of m and return it as a column vector
    out = NNMatrix(m.d, 1)
    out.w[:,1] = m.w[ix,:]'
    # backprop function
    push!(g.backprop,
          function ()
             m.dw[ix,:] += out.dw[:,1]' #
          end )
    return out
end

function tanh(g::Graph, m::NNMatrix)
    out = NNMatrix(m.n, m.d)
    out.w = tanh(m.w)
    # backprop function
    push!(g.backprop,
          function ()
              for i = 1:m.n
                  for j = 1:m.d
                      m.dw[i,j] += (1. - out.w[i,j]^2) * out.dw[i,j]
                  end
              end
          end )
    return out
end

function sigmoid(g::Graph, m::NNMatrix)
    out = NNMatrix(m.n, m.d)
    out.w = ones(m.w) ./ (ones(m.w) .+ exp(-m.w)) # sigmoid function
    # backprop function
    push!(g.backprop,
          function ()
              for i = 1:m.n
                  for j = 1:m.d
                      m.dw[i,j] +=  out.w[i,j] * (1. - out.w[i,j]) *  out.dw[i,j]
                  end
              end
          end )
    return out
end

function relu(g::Graph, m::NNMatrix)
    out = NNMatrix(m.n, m.d)
    for i = 1:m.n
      for j = 1:m.d
          out.w[i,j] = m.w[i,j] < 0. ? 0. :  m.w[i,j]
      end
    end
    # backprop function
    push!(g.backprop,
      function ()
          for i = 1:m.n
              for j = 1:m.d
                  m.dw[i,j] +=  m.w[i,j] < 0. ? 0 : out.dw[i,j]
              end
          end
      end )
    return out
end

function mul(g::Graph, m1::NNMatrix, m2::NNMatrix)
    out = NNMatrix(m1.n, m2.d)
#     println((m1.n, m2.d))
    n = m1.n
    d = m2.d
    for i = 1:n
        for j = 1:d
            dot = 0.
            for k = 1:m1.d
                dot += m1.w[i,k] * m2.w[k,j]
            end
            out.w[i,j] = dot
        end
    end

    # backprop function
    push!(g.backprop,
        function ()
            for i = 1:m1.n # m1's num row
                for j = 1:m2.d # m2's num row
                    for k = 1:m1.d # m1's num col
                          b = out.dw[i,j]
                          m1.dw[i,k] += m2.w[k,j] * b
                          m2.dw[k,j] += m1.w[i,k] * b
                      end
                  end
              end
          end )
    return out
end

function add(g::Graph, m1::NNMatrix, m2::NNMatrix)
    out = NNMatrix(m1.n, m1.d)
    out.w = m1.w .+ m2.w
    push!(g.backprop,
        function ()
              m1.dw .+= out.dw
              m2.dw .+= out.dw
        end )
    return out
end

function eltmul(g::Graph, m1::NNMatrix, m2::NNMatrix) # element-wise multiplication
    out = NNMatrix(m1.n, m2.d)
    out.w = m1.w .* m2.w
    # backprop function
    push!(g.backprop,
        function ()
            m1.dw .+= m2.w .* out.dw
            m2.dw .+= m1.w .* out.dw
          end )
    return out
end
