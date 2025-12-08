using Test
using RecurrentNetworkModels
using Reactant
using StableRNGs
using Lux
using Random
using StatsBase

@testset "Loss functions" begin
    n = [0,1,2,3]
    η = log.([0.1, 0.5,1.2, 1.5])
    w = [0.1, 0.2, 0.3, 0.5]
    ll = RecurrentNetworkModels.poisson_loss.(η,n)
    @test ll ≈ [0.10000000000000002, 1.1931471805599454, 1.528504066972036, 2.0753641449035616]

    llw = RecurrentNetworkModels.weighted_poisson_loss.(η,n,w)
    @test llw ≈ [0.010000000000000002, 0.23862943611198909, 0.45855122009161076, 1.0376820724517808]

    nn = repeat(n, 1, 5, 10)
    ηη = repeat(η, 1, 5, 10)
    ww = repeat(w, 1, 5, 10)
    func = RecurrentNetworkModels.WeightedPoissonLoss()
    llq = func(ηη, nn, ww)
    @test llq ≈ 0.43621568216384626
    @test llq ≈ mean(llw)
end

@testset "Basic" begin 
    dev = reactant_device()
    cdev = cpu_device()
    rng = Random.default_rng()
    Random.seed!(rng, 1234)
    ninputs = 16
    nhidden = 64
    noutputs = 2
    output_nonlinearity = sigmoid
    τ = 0.2f0
    η = 0.01f0
    model = RecurrentNetworkModels.LeakyRNNModel(ninputs, nhidden, noutputs;output_nonlinearity=output_nonlinearity,τ=τ,η=η)
    ps,st = dev.(Lux.setup(rng, model))
    # compile a basic model with random input
    x = randn(Float32, 16, 32, 128)
    xe  = dev(x)
    model_compiled = @compile model(xe, ps, Lux.testmode(st))
    (ye,he), st_new = model_compiled(xe, ps, st)
    y,h = cdev.((ye, he))
    @test size(y) == (2,32,128)
    @test size(h) == (64,32,128)

    # check training
    y = randn(Float32, 2, 32, 128)
    w = randn(Float32, 2, 32, 128)
    data_provider() = (x,y,w)
    ps_train, st_train = RecurrentNetworkModels.train_model(model, data_provider)

    # load with re-training
    ps_load, st_load = RecurrentNetworkModels.train_model(model, data_provider)

    @test ps_load == ps_train
    @test st_load == st_train
end