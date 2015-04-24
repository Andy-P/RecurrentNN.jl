using RecurrentNN
reload("RecurrentNN.jl")
using Base.Test

# graph output test
m1 = RecurrentNN.NNMatrix(3,2)
m1.w[1,1] = 1.; m1.w[1,2] = 2.
m1.w[2,1] = 3.; m1.w[2,2] = 4.
m1.w[3,1] = 5.; m1.w[3,2] = 6.

m2 = RecurrentNN.NNMatrix(2,3)
m2.w[1,1] = 2.; m2.w[1,2] = 3.; m2.w[1,3] = 4.
m2.w[2,1] = 5.; m2.w[2,2] = 6.; m2.w[2,3] = 7.

# add test
g =  RecurrentNN.Graph()
m3 = RecurrentNN.add(g,m1,m1)
m3.dw[1,1] = .1; m3.dw[1,2] = .2
m3.dw[2,1] = .3; m3.dw[2,2] = .4
m3.dw[3,1] = .5; m3.dw[3,2] = .6

g.backprop[1]()
# set previous chained gradient
@test m1.dw[1,1] == 0.2
@test m1.dw[1,2] == 0.4
@test m1.dw[2,1] == 0.6
@test m1.dw[2,2] == 0.8
@test m1.dw[3,1] == 1.0
@test m1.dw[3,2] == 1.2

# mul test
m1.dw[:] = 0. # reset gradient matrices
m2.dw[:] = 0. # reset  gradient matrices
g =  RecurrentNN.Graph()
m3 = RecurrentNN.mul(g,m1,m2)
@test m3.w[1,1] == 12.
@test m3.w[1,2] == 15.
@test m3.w[1,3] == 18.
@test m3.w[2,1] == 26.
@test m3.w[2,2] == 33.
@test m3.w[2,3] == 40.
@test m3.w[3,1] == 40.
@test m3.w[3,2] == 51.
@test m3.w[3,3] == 62.

m3.dw[1,1] = .1; m3.dw[1,2] = .2; m3.dw[1,3] = .3
m3.dw[2,1] = .4; m3.dw[2,2] = .5; m3.dw[2,3] = .6
m3.dw[3,1] = .7; m3.dw[3,2] = .8; m3.dw[3,3] = .9
g.backprop[1]()

# m1 gradient tests
@test m1.dw[1,1] == 2.
@test m1.dw[1,2] == 3.8000000000000003
@test m1.dw[2,1] == 4.699999999999999
@test m1.dw[2,2] == 9.2
@test m1.dw[3,1] == 7.4
@test m1.dw[3,2] == 14.600000000000001

# m2 gradient tests
@test m2.dw[1,1] == 4.800000000000001
@test m2.dw[1,2] == 5.7
@test m2.dw[1,3] == 6.6
@test m2.dw[2,1] == 5.999999999999999
@test m2.dw[2,2] == 7.200000000000001
@test m2.dw[2,3] == 8.4


# reul() tests
m4 = RecurrentNN.NNMatrix(3,2)
m4.w[1,1] = 1.; m4.w[1,2] =-2.
m4.w[2,1] =-3.; m4.w[2,2] = 4.
m4.w[3,1] = 5.; m4.w[3,2] =-6.
g =  RecurrentNN.Graph()
m5 = RecurrentNN.relu(g,m4)
@test m5.w[1,1] == 1.
@test m5.w[1,2] == 0.
@test m5.w[2,1] == 0.
@test m5.w[2,2] == 4.
@test m5.w[3,1] == 5.
@test m5.w[3,2] == 0.

m5.dw[1,1] =-.1; m5.dw[1,2] = .2
m5.dw[2,1] = .3; m5.dw[2,2] = .4
m5.dw[3,1] = .5; m5.dw[3,2] = .6

g.backprop[1]()
@test m4.dw[1,1] == -0.1
@test m4.dw[1,2] == 0.
@test m4.dw[2,1] == 0.
@test m4.dw[2,2] == 0.4
@test m4.dw[3,1] == 0.5
@test m4.dw[3,2] == 0.


# rowpluck() tests
m4 = RecurrentNN.NNMatrix(3,2)
m4.w[1,1] = 1.; m4.w[1,2] =-2.
m4.w[2,1] =-3.; m4.w[2,2] = 4.
m4.w[3,1] = 5.; m4.w[3,2] =-6.

g =  RecurrentNN.Graph()
m5 = RecurrentNN.rowpluck(g,m4,2)
@test m5.w[1,1] == -3.
@test m5.w[2,1] == 4.

m5.dw[1,1] =-.1; m5.dw[2,1] = .2
g.backprop[1]()
m4.dw
@test m4.dw[1,1] == 0
@test m4.dw[1,2] == 0
@test m4.dw[2,1] ==-0.1
@test m4.dw[2,2] == 0.2
@test m4.dw[3,1] == 0.
@test m4.dw[3,2] == 0.


# softmax tests
m6 = RecurrentNN.NNMatrix(5,1)
m6.w[1,1] = 0.3; m6.w[2,1] =0.1; m6.w[3,1] =0.6; m6.w[4,1] = 0.002; m6.w[5,1] = 0.00001
sm = RecurrentNN.softmax(m6)
# @test_approx_eq
@test_approx_eq sm.w[1,1] 0.21494050089813527
@test_approx_eq sm.w[2,1] 0.17597839816728894
@test_approx_eq sm.w[3,1] 0.2901393282421457
@test_approx_eq sm.w[4,1] 0.1595506217827435
@test_approx_eq sm.w[5,1] 0.15939115090968653
