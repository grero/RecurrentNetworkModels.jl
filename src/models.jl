using Lux
using Printf
using StableRNGs
using CRC32c

function LeakyRNNModel(in_dims, hidden_dims, out_dims)
    rnn_cell = LeakyRNNCell(in_dims => hidden_dims)
    classifier = Dense(hidden_dims => out_dims, sigmoid)
    return @compact(;rnn_cell, classifier) do x::AbstractArray{T,3} where {T}
        #x = reshape(x, size(x)..., 1)
        x_init, x_rest = Lux.Iterators.peel(LuxOps.eachslice(x, Val(2)))
        y, carry = rnn_cell(x_init)
        oo = classifier(y)
        so = size(oo)
        output = reshape(oo, so[1], 1, so[2])
        for x in x_rest
            y,carry = rnn_cell((x,carry))
            #oo = classifier(y)
            #@show typeof(oo) size(oo)
            output = hcat(output, reshape(classifier(y),so[1],1,so[2]))
        end
       @return output
    end
end

const lossfn = WeightedMSELoss()

function compute_loss(model, ps, st, (x,y,w))
    ŷ, st_ = model(x, ps, st)
    loss = lossfn(ŷ, y, w)
    return loss, st_, (;y_pred=ŷ)
end

function compute_loss(ŷ, y, w)
    los = lossfn(ŷ,y,w)
end

#this is just random; replace with something meaningful
accuracy(y_pred, y_true) = mean(sqrt.(sum(abs2,y_pred[:,end-10:end,:] .- y_true[:,end-10:end,:],dims=1)))

function train_model(model, x::AbstractArray{Float32,3},y::AbstractArray{Float32,3},z::AbstractArray{Float32,3};kwargs...)
    train_model(model, ()->(x,y,z), accuracy;kwargs...)
end

function train_model(model, data_provider, accuracy_func::Function=accuracy;nepochs=25, accuracy_threshold=0.9f0,save_file="model_state.jld2",redo=false, learning_rate=0.01f0, freeze_input=false, rseed=12345, h=zero(UInt64))
    # create signature
    h = crc32c(string(nepochs),h)
    h = crc32c(string(accuracy_threshold), h)
    h = crc32c(string(learning_rate), h)
    h = crc32c(string(freeze_input),h)
    h = crc32c(string(rseed), h)
    hs = string(h, base=16)
    fname = replace(save_file, ".jld2"=> "_$(hs).jld2")
    if isfile(fname) && !redo
        print(stdin, "File $(fname) already exists. Starting training from previous parameters. To restart from a random state, call with `redo=true`\n")
        _ps,_st = JLD2.load(fname, "params","state")
    else
        _ps,_st = Lux.setup(rng, model)
    end
    dev = reactant_device()
    cdev = cpu_device()
    rng=StableRNG(rseed)
    ps,st = dev((_ps, _st))
    train_state = Training.TrainState(model, ps, st, Adam(learning_rate))
    #evaluation set
    (xe,ye,we)  = dev.(data_provider())
    model_compiled = @compile model(xe, ps, Lux.testmode(st))
    prog = Progress(nepochs, "Training model...")
    total_loss0 = 0.0f0
    total_loss_p = typemax(Float32)
    for epoch in 1:nepochs
        (xt,yt,wt)  = dev.(data_provider())

        (_, loss, _, train_state) = Training.single_train_step!(
            AutoEnzyme(), compute_loss, (xt, yt, wt), train_state
        )
        total_loss = loss * length(yt)
        total_samples = length(yt)

        if epoch == 1
            total_loss0 = total_loss
        end
        loss = @sprintf "%4.5f" total_loss / total_samples
        total_loss_change = (total_loss0-total_loss)/total_loss0

        # save the state only if the loss decreased
        if total_loss < total_loss_p
            _ps,_st =  cdev((train_state.parameters, train_state.states))
            JLD2.save(fname, Dict("state"=>_st, "params"=>_ps))
        end
        total_loss_p = total_loss

        #set up validation
        total_acc = 0.0f0
        total_loss = 0.0f0
        total_samples = 0

        st_ = Lux.testmode(train_state.states) # what does this do?
        ŷ,st_ = model_compiled(xe, train_state.parameters, st_)
        ŷp, y,w = (cdev(ŷ), cdev(ye), cdev(we))
        total_acc = accuracy_func(ŷp, y)*length(y)
        total_loss = compute_loss(ŷp, y,w)*length(y)
        total_samples = length(y)
        total_acc /= total_samples
        if total_acc >= accuracy_threshold
            finish!(prog)
            print(stdout, "Accuracy threshold achieved at $total_acc. Returning...\n")
            break
        end

        vloss = @sprintf "%4.5f" total_loss/total_samples
        vacc = @sprintf "%4.5f" total_acc
        next!(prog, showvalues=[(:Loss, loss),(:ValidationLoss, vloss), (:ValidationAccuracy, vacc),
                                (:TotalLossChange, total_loss_change)])

        # cleanup; at some point we should use DeviceIterator here, but for now try and make use of the internals
        Lux.MLDataDevices.Internal.unsafe_free!(xt)
        Lux.MLDataDevices.Internal.unsafe_free!(yt)
        Lux.MLDataDevices.Internal.unsafe_free!(wt)
        Lux.MLDataDevices.Internal.unsafe_free!(ŷ)
        # run GC every 10th epoch
        if epoch % 10 == 0
            GC.gc()
        end
    end

    return cdev((train_state.parameters, train_state.states))
end
