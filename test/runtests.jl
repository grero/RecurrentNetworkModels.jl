using Test
using RecurrentNetworkModels
using Reactant
using StableRNGs
using Lux
using Random


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