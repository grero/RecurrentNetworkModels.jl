using Lux

@concrete struct LeakyRNNCell <: AbstractRecurrentCell
    train_state <: StaticBool
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_weight
    init_recurrent_weight
    init_state
    τ
    use_bias <: StaticBool
end

function LeakyRNNCell(
    (in_dims, out_dims)::Pair{<:IntegerType,<:IntegerType},
    activation=tanh;
    use_bias::BoolType=True(),
    train_state::BoolType=False(),
    init_bias=nothing,
    init_weight=nothing,
    init_recurrent_weight=init_weight,
    init_state=zeros32,
    τ=Float32(0.2)
)
    return LeakyRNNCell(
        static(train_state),
        activation,
        in_dims,
        out_dims,
        init_bias,
        init_weight,
        init_recurrent_weight,
        init_state,
        τ,
        static(use_bias),
    )
end

function Lux.initialparameters(rng::AbstractRNG, rnn::LeakyRNNCell)
    weight_ih = Lux.init_rnn_weight(
        rng, rnn.init_weight, rnn.out_dims, (rnn.out_dims, rnn.in_dims)
    )
    weight_hh = Lux.init_rnn_weight(
        rng, rnn.init_recurrent_weight, rnn.out_dims, (rnn.out_dims, rnn.out_dims)
    )
    ps = (; weight_ih, weight_hh)
    if Lux.has_bias(rnn)
        bias_ih = Lux.init_rnn_bias(rng, rnn.init_bias, rnn.out_dims, rnn.out_dims)
        bias_hh = Lux.init_rnn_bias(rng, rnn.init_bias, rnn.out_dims, rnn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    Lux.has_train_state(rnn) &&
        (ps = merge(ps, (hidden_state=rnn.init_state(rng, rnn.out_dims),)))
    return ps
end

Lux.initialstates(rng::AbstractRNG, ::LeakyRNNCell) = (rng=Lux.Utils.sample_replicate(rng),)

function (rnn::LeakyRNNCell{False})(x::AbstractMatrix, ps, st::NamedTuple)
    rng = Lux.replicate(st.rng)
    hidden_state = Lux.init_rnn_hidden_state(rng, rnn, x)
    return rnn((x, (hidden_state,)), ps, merge(st, (; rng)))
end

function (rnn::LeakyRNNCell{True})(x::AbstractMatrix, ps, st::NamedTuple)
    hidden_state = Lux.init_trainable_rnn_hidden_state(ps.hidden_state, x)
    return rnn((x, (hidden_state,)), ps, st)
end

function (rnn::LeakyRNNCell)(
    (x, (hidden_state,))::Tuple{<:AbstractMatrix,Tuple{<:AbstractMatrix}},
    ps,
    st::NamedTuple,
)
    y, hidden_stateₙ = Lux.match_eltype(rnn, ps, st, x, hidden_state)

    bias_hh = Lux.safe_getproperty(ps, Val(:bias_hh))
    z₁ = Lux.fused_dense_bias_activation(identity, ps.weight_hh, hidden_stateₙ, bias_hh)
    bias_ih = Lux.safe_getproperty(ps, Val(:bias_ih))
    z₂ = Lux.fused_dense_bias_activation(identity, ps.weight_ih, y, bias_ih)

    # TODO: This operation can be fused instead of doing add then activation
    hₙ = Lux.fast_activation!!(rnn.activation, z₁ .+ z₂)
    hₙ = (1-rnn.τ)*hidden_stateₙ .+ rnn.τ*hₙ
    return (hₙ, (hₙ,)), st
end

