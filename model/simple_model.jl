using FieldMetadata
using Flatten
import Base: @kwdef
flatten = Flatten.flatten; reconstruct = Flatten.reconstruct

include("random_restarts.jl")

@metadata bounds (-Inf, -Inf, Inf, Inf) Tuple

abstract type AbstractWeighter end
abstract type SimpleWeighter <: AbstractWeighter end
abstract type DependentWeighter <: AbstractWeighter end

score(model::SimpleWeighter, u, control) = score(model, u)
score(::AbstractWeighter, u) = error("Not Implemented")


struct NullWeighter <: SimpleWeighter end

function score(model::NullWeighter, u)
    ones(length(u))
end


@bounds @kwdef struct Softmax{T,U} <: SimpleWeighter
    β::T = missing | (0, 0.1, 1, Inf)
    C::U = missing | (0, 0.1, 1, Inf)
end

function score(model::Softmax, u)
    (;β, C) = model
    @. C + exp(u * β) 
end


# @bounds @kwdef struct UWS{T,U} <: SimpleWeighter
#     ev::T = 0. | (-Inf, -1, 1, Inf)
#     β_abs::U = missing | (0, .1, 1, Inf)
# end

# function score(model::UWS, u)
#     (;ev, β_abs) = model
#     @. abs(u - ev) ^ β_abs
# end

@bounds @kwdef struct UWS{T,U} <: SimpleWeighter
    ev::T = 0. | (-Inf, -1, 1, Inf)
    C::U = missing | (0, .1, 1, Inf)
end

function score(model::UWS, u)
    (;ev, C) = model
    @. abs(u - ev) + C
end

@bounds @kwdef struct SwitchWeighter{W1,W2} <: DependentWeighter
    low::W1
    high::W2
end

function score(model::SwitchWeighter, u, control)
    if control == "high"
        score(model.high, u)
    else @assert control == "low"
        score(model.low, u)
    end
end

@bounds @kwdef struct Sequenced{T,U,V} <: DependentWeighter
    β_exp::T = missing | (0, 0.1, 1, Inf)
    β_abs::U = missing | (0, 0.1, 1, Inf)
    ev::V = 0. | (-Inf, -1, 1, Inf)
end

function score(model::Sequenced, u, control)
    (;β_exp, β_abs, ev) = model
    if control == "high"
        @. exp(u * β_exp)
    else @assert control == "low"
        @. abs(u - ev) ^ β_abs
    end
end

@bounds @kwdef struct SoftmaxUWS{T,U,V} <: SimpleWeighter
    β::T = missing | (0, 0.1, 1, Inf)
    β_abs::U = missing | (0, 0.1, 1, Inf)
    ev::V = 0. | (-Inf, -1, 1, Inf)
end

function score(model::SoftmaxUWS, u)
    (;β, β_abs, ev) = model
    @. exp(u * β) * abs(u - ev) ^ β_abs
end

@bounds @kwdef struct SwitchingSoftmaxUWS{T,U,V,W,X} <: DependentWeighter
    β_low::T = missing | (0, 0.1, 1, Inf)
    β_high::U = missing | (0, 0.1, 1, Inf)
    C_abs::V = missing | (0, 0.1, 1, Inf)
    C_exp::W = missing | (0, 0.1, 1, Inf)
    ev::X = 0. | (-Inf, -1, 1, Inf)
end

function score(model::SwitchingSoftmaxUWS, u, control)
    (;β_low, β_high, C_abs, C_exp, ev) = model
    β = control == "high" ? β_high : β_low
    @. (exp(u * β) + C_exp) * (abs(u - ev) + C_abs)
end

struct ProductWeighter{W1, W2} <: SimpleWeighter
    weighter1::W1
    weighter2::W2
end

function score(model::ProductWeighter, u)
    score(model.weighter1, u) .* score(model.weighter2, u)
end

# struct MixtureWeighter{W1,W2,T,U} <: DependentWeighter
#     weighter1::W1 = W1()
#     weighter2::W2 = W2()
#     w_low::T = missing | (0, .1, .9, 1)
#     w_high::U = missing | (0, .1, .9, 1)
# end

# function score(model::MixtureWeighter, u, control)
#     (;β_low, β_high, β_abs, ev) = model
#     β = control == "high" ? β_high : β_low
#     @. exp(u * β) * (abs(u - ev) + C)
# end

@bounds @kwdef struct AbsExp{T,U} <: SimpleWeighter
    w::T = missing | (0, 0, 1, 1)
    β::U = missing | (0, 0.1, 1, Inf)
end

function score(model::AbsExp, u)
    (;w, β) = model
    @. ((1 - w) * abs(u) + w * exp(u)) ^ β
end

# %% ==================== Model ====================

@bounds @kwdef struct Model{W<:AbstractWeighter,T,U}
    weighter::W = W()
    ε::T = 0 | (0, 0, 1, 1)
    β_acc::U = missing | (0, 0, 1, 1000)  # accessibility weight    
end

Model(weighter::AbstractWeighter; kws...) = Model(;weighter, kws...)
Model{W}(;ε=missing, β_acc=missing, kws...) where W = Model(W(;kws...); ε, β_acc)

function likelihood(model::Model, trial::NamedTuple)
    (;weighter, ε, β_acc) = model
    p = score(weighter, trial.evaluation, trial.control)
    n = length(p)
    p ./= sum(p)
    p .*= (1 .- ε)
    p .+= ε .* (1/n)
    p .*= trial.accessibility .^ β_acc
    p ./= sum(p)
    # end
    p
end

function logp(model::Model, trial::NamedTuple)
    p = likelihood(model, trial)
    sum(log.(p[trial.considered]))
end

function logp(model::Model, trials::AbstractVector{<:NamedTuple})
    mapreduce(+, trials) do trial
        logp(model, trial)
    end
end

function Base.show(io::IO, model::Model{W}) where W
    print(io, "Model{", W.name.name, "}")
    T = Union{Real,Missing}
    for (k, v) in zip(fieldnameflatten(model, T), flatten(model,T))
        print(io, "\n", lpad(k, 7), ": ", v)
    end
end

function get_bounds(base_model::Model)
    bb = map(invert(metaflatten(base_model, bounds, Missing))) do x
        collect(float.(x))
    end
    @assert all(isfinite.(bb[2])) && all(isfinite.(bb[3]))
    bb
end

function find_mle(base_model::Model, trials; n_restart = 10)
    loss(x) = -logp(reconstruct(base_model, x, Missing), trials)
    res = random_restarts(loss, get_bounds(base_model)..., n_restart)
    reconstruct(base_model, res.minimizer, Missing)
end

function update(model::Model; kws...)
    x = collect(flatten(model))
    for (i, fn) in enumerate(fieldnameflatten(model))
        if fn in keys(kws)
            x[i] = kws[fn]
        end
    end
    reconstruct(model, x)
end
