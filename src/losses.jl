using Lux: fused_agg

Lux.fused_agg(::typeof(sum), op::OP, x::Number, y::Number, w::Number) where {OP} = w*op(x, y)
function Lux.fused_agg(::typeof(sum), op::OP, x::AbstractArray, y::AbstractArray, w::AbstractArray) where {OP}
    if fast_scalar_indexing(x) && fast_scalar_indexing(y) && fast_scalar_indexing(w)
        res = Core.Compiler.return_type(op, Tuple{eltype(x),eltype(y),eltpype(w)})(0)
        @simd ivdep for i in eachindex(x, y, w)
            @inbounds res += w[i]*op(x[i], y[i])
        end
        return res
    end
    return fallback_fused_agg(sum, op, x, y, w)
end

@inline Lux.fallback_fused_agg(f::F, op::OP, x::AbstractArray, y::AbstractArray, w::AbstractArray) where {F,OP} = f(w.*op.(x,y))


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
                ∂x = @thunk derivative.(Ref(op), x, y) .* Δ
                return NoTangent(), NoTangent(), NoTangent(), ∂x, NoTangent()
            end
        return res, ∇fused_agg_custom_derivative
    end

    # Without custom derivatives use ForwardDiff for the looped implementation
    if fast_scalar_indexing(x) && fast_scalar_indexing(y) && fast_scalar_indexing(w)
        x_dual = Dual{Nothing,eltype(x),1}.(x, (Partials{1,eltype(x)}((one(eltype(x)),)),))
        x_partials = similar(x)
        T = eltype(x)
        res = Core.Compiler.return_type(op, Tuple{T,eltype(y)})(0)
        @inbounds @simd for i in eachindex(x_partials, x, y)
            x_dual = Dual{Nothing,T,1}(x[i], Partials{1,T}((one(T),)))
            tmp = op(x_dual, y[i])
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

    return CRC.rrule_via_ad(cfg, fallback_fused_agg, sum, op, x, y)
end


