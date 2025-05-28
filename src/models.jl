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

#this is just random; replace with something meaningful
accuracy(y_pred, y_true) = mean(sqrt.(sum(abs2,y_pred[:,end-10:end] .- y_true[:,end-10:end],dims=1)))

function train_model(model, x::AbstractArray{Float32,3},y::AbstractArray{Float32,3})
    train_model(model, ()->(x,y))
end

function train_model(model, data_provider::Function)
    dev = reactant_device()
    cdev = cpu_device()
    rng = StableRNG(12345)
    ps,st = dev(Lux.setup(rng, model))
    train_state = Training.TrainState(model, ps, st, Adam(0.01f0))
    #evaluation set
    (xe,ye)  = dev.(data_provider())
    model_compiled = @compile model(first(Lux.eachslice(xe,dims=3)), ps, Lux.testmode(st))
    for epoch in 1:25
        (xt,yt)  = dev.(data_provider())
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

        total_acc = 0.0f0
        total_loss = 0.0f0
        total_samples = 0

        st_ = Lux.testmode(train_state.states) # what does this do?
        for (_x,_y) in zip(Lux.eachslice(xe,dims=3), Lux.eachslice(ye,dims=3))
            ŷ,st_ = model_compiled(_x, train_state.parameters, st_)
            ŷ, y = (cdev(ŷ), cdev(_y))
            total_acc += accuracy(ŷ, y)*length(y)
            total_loss += lossfn(ŷ, y)*length(y)
            total_samples += length(y)
        end

        @printf "Validatio: \tLoss %4.5f\tAccuracy %4.5f\n" (total_loss/total_samples) (total_acc / total_samples)

    end

    return cdev((train_state.parameters, train_state.states))
end