import Lux.LossFunctionImpl:fused_agg, fallback_fused_agg, check_sizes
import Lux.unsafe_apply_loss
using Lux: @thunk, CRC, GenericLossFunction, AbstractLossFunction

@concrete struct GenericWeightedLossFunction <: AbstractLossFunction
    loss_fn
    agg
end

GenericWeightedLossFunction(loss_fn; agg=mean) = GenericWeightedLossFunction(loss_fn, agg)

function unsafe_apply_loss(loss::GenericWeightedLossFunction, ŷ, y, w)
    return fused_agg(loss.agg, loss.loss_fn, ŷ, y , w)
end

# Match the sizes of the inputs to the loss function
function check_sizes(ŷ::AbstractArray, y::AbstractArray, w::AbstractArray)
    check_sizes(ŷ,y)
    if size(w,1) > 1 && size(w,1) != size(y,1)
        throw(DimensionMismatch("loss function expects size(w) to be 1 or to match \
                           size(y) = $(size(y,1))"))
    else
        for d in 2:max(ndims(w), ndims(y))
            if size(w, d) != size(y, d)
                throw(
                    DimensionMismatch("loss function expects size(w) = $(size(w)) to match \
                            size(y) = $(size(y))")
                )
            end
        end
    end
    return nothing
end

function (loss::AbstractLossFunction)(ŷ, y, w)
    check_sizes(ŷ, y, w)
    return unsafe_apply_loss(loss, ŷ, y,w)
end

weighted_l2_distance_loss(x::T1, y::T2, w::T3) where {T1,T2,T3} = w*abs2(x - y)
has_custom_derivative(::typeof(weighted_l2_distance_loss)) = false 
#function derivative(::typeof(weightd_l2_distance_loss), x::T1, y::T2, w::T3) where {T1,T2,T3}
#    d = x-y
#    return convert(T1, 2*d*w + abs2(d))
#end

WeightedMSELoss(; agg=mean) = GenericWeightedLossFunction(weighted_l2_distance_loss; agg)

fused_agg(::typeof(sum), op::OP, x::Number, y::Number, w::Number) where {OP} = op(x, y, w)
function fused_agg(::typeof(sum), op::OP, x::AbstractArray, y::AbstractArray, w::AbstractArray) where {OP}
    if fast_scalar_indexing(x) && fast_scalar_indexing(y) && fast_scalar_indexing(w)
        res = Core.Compiler.return_type(op, Tuple{eltype(x),eltype(y),eltype(w)})(0)
        @simd ivdep for i in eachindex(x, y, w)
            @inbounds res += op(x[i], y[i], w[i])
        end
        return res
    end
    return fallback_fused_agg(sum, op, x, y, w)
end

function fused_agg(::typeof(mean), op::OP, x::AbstractArray, y::AbstractArray, w::AbstractArray) where {OP}
    return fused_agg(sum, op, x, y, w) / length(x)
end

@inline fallback_fused_agg(f::F, op::OP, x::AbstractArray, y::AbstractArray, w::AbstractArray) where {F,OP} = f(op.(x,y,w))


#TODO: Implement the RRule for these op's
#
#
function CRC.rrule(
    cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
    ::typeof(fused_agg),
    ::typeof(sum),
    op::OP,
    x,
    y,
    w,
) where {OP}
    if has_custom_derivative(op)
        res = fused_agg(sum, op, x, y, w)
        ∇fused_agg_custom_derivative =
            Δ -> begin
                ∂x = @thunk derivative.(Ref(op), x, y, w).* Δ
                return NoTangent(), NoTangent(), NoTangent(), ∂x, NoTangent()
            end
        return res, ∇fused_agg_custom_derivative
    end

    # Without custom derivatives use ForwardDiff for the looped implementation
    if fast_scalar_indexing(x) && fast_scalar_indexing(y) && fast_scalar_indexing(w)
        x_dual = Dual{Nothing,eltype(x),1}.(x, (Partials{1,eltype(x)}((one(eltype(x)),)),))
        x_partials = similar(x)
        T = eltype(x)
        res = Core.Compiler.return_type(op, Tuple{T,eltype(y), eltype(w)})(0)
        @inbounds @simd for i in eachindex(x_partials, x, y, w)
            x_dual = Dual{Nothing,T,1}(x[i], Partials{1,T}((one(T),)))
            tmp = op(x_dual, y[i], w[i])
            x_partials[i] = ForwardDiff.partials(tmp, 1)
            res += ForwardDiff.value(tmp)
        end
        ∇fused_agg_loop =
            Δ -> begin
                @simd ivdep for i in eachindex(x_partials)
                    @inbounds x_partials[i] *= Δ
                end
                return NoTangent(), NoTangent(), NoTangent(), x_partials, NoTangent()
            end
        return res, ∇fused_agg_loop
    end

    return CRC.rrule_via_ad(cfg, fallback_fused_agg, sum, op, x, y,w)
end