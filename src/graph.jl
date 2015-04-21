type Graph
   backprop::Array{Function,1}
   Graph() = new(Array(Function,0))
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

# y = [1,2,3]
# z = [4,5,6]

# function prnt(x::Array, y::Array)
#     println(vcat(x,y))
# end
# prnt(y,z)
# f -> prnt(z,y)()
# g -> prnt(y,z)

# fs = Array(Function,0)
# push!(fs,()->prnt(z,y))

# y[1:3] = [3,2,1]
# fs[1]()
