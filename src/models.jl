using Lux
using Printf
using StableRNGs

function LeakyRNNModel(in_dims, hidden_dims, out_dims)
    rnn_cell = LeakyRNNCell(in_dims => hidden_dims)
    classifier = Dense(hidden_dims => out_dims, sigmoid)
    return @compact(;rnn_cell, classifier) do x::AbstractArray{T,2} where {T}
        x = reshape(x, size(x)..., 1)
        x_init, x_rest = Lux.Iterators.peel(LuxOps.eachslice(x, Val(2)))
        y, carry = rnn_cell(x_init)
        output = [vec(classifier(y))]
        for x in x_rest
            y,carry = rnn_cell((x,carry))
            output = vcat(output, [vec(classifier(y))])
        end
       @return hcat(output...)
    end
end
const lossfn = MSELoss()

function compute_loss(model, ps, st, (x,y))
    ŷ, st_ = model(x, ps, st)
    loss = lossfn(ŷ, y)
    return loss, st_, (;y_pred=ŷ) 
end

function train_model(model, x::AbstractArray{Float32,3},y::AbstractArray{Float32,3})
    dev = reactant_device()
    cdev = cpu_device()
    rng = StableRNG(12345)
    ps,st = dev(Lux.setup(rng, model))
    train_state = Training.TrainState(model, ps, st, Adam(0.01f0))
    (xt,yt)  = dev.((x,y))
    @show typeof(xt)
    model_compiled = @compile model(xt[:,:,1], ps, Lux.testmode(st))
    for epoch in 1:25
        total_loss = 0.0f0
        total_samples = 0
        for (_x,_y) in zip(Lux.eachslice(xt,dims=3), Lux.eachslice(yt,dims=3))
            (_, loss, _, train_state) = Training.single_train_step!(
                AutoEnzyme(), lossfn, (_x, _y), train_state
            )
            total_loss += loss * length(_y)
            total_samples += length(_y)
        end
        @printf "Epoch [%3d]: Loss %4.5f\n" epoch (total_loss / total_samples)
    end

    return cdev((train_state.parameters, train_state.states))
end