using FieldMetadata
using Flatten
using Distributions
import Base: @kwdef
flatten = Flatten.flatten; reconstruct = Flatten.reconstruct

include("random_restarts.jl")

@metadata bounds (-Inf, -Inf, Inf, Inf) Tuple
@metadata prior2 Normal(0,1) Distribution

# import FieldMetadata: @prior, prior

abstract type AbstractWeighter end
abstract type SimpleWeighter <: AbstractWeighter end
abstract type DependentWeighter <: AbstractWeighter end

score(model::SimpleWeighter, u, control) = score(model, u)
score(::AbstractWeighter, u) = error("Not Implemented")


struct NullWeighter <: SimpleWeighter end

function score(model::NullWeighter, u)
    ones(length(u))
end

@prior2 @kwdef struct Softmax{T,U} <: SimpleWeighter
    β::T = missing | Normal(0, 3)
    C::U = missing | Exponential(500)
end

function score(model::Softmax, u)
    (;β, C) = model
    @. C + exp(u * β) 
end

@prior2 @kwdef struct Softmax4{T,U} <: SimpleWeighter
    β::T = missing | Normal(0, 3)
    ε_soft::U = missing | Beta(1,1)
end

function score(model::Softmax4, u)
    (;β, ε_soft) = model
    p = exp.(u .* β) 
    p ./= sum(p)
    p .*= (1 .- ε_soft)
    p .+= ε_soft .* (1/length(p))
    p
end

@prior2 @kwdef struct UWS{T,U} <: SimpleWeighter
    ev::T = 0. | Normal(0, 5)
    C::U = missing | Exponential(10)
end

function score(model::UWS, u)
    (;ev, C) = model
    @. abs(u - ev) + C
end

@prior2 @kwdef struct SwitchingSoftmaxUWS{T,U,V,W,X} <: DependentWeighter
    β_low::T = missing | Normal(0, 3)
    β_high::U = missing | Normal(0, 3)
    C_abs::V = missing | Exponential(10)
    C_exp::W = missing | Exponential(500)
    ev::X = 0. | Normal(0, 5)
end

function score(model::SwitchingSoftmaxUWS, u, control)
    (;β_low, β_high, C_abs, C_exp, ev) = model
    β = control == "high" ? β_high : β_low
    @. (exp(u * β) + C_exp) * (abs(u - ev) + C_abs)
end

@kwdef struct SwitchWeighter{W1,W2} <: DependentWeighter
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


struct ProductWeighter{W1, W2} <: SimpleWeighter
    weighter1::W1
    weighter2::W2
end

function score(model::ProductWeighter, u)
    score(model.weighter1, u) .* score(model.weighter2, u)
end

@prior2 @kwdef struct MixtureWeighter{W1,W2,T,U} <: DependentWeighter
    weighter1::W1 = W1()
    weighter2::W2 = W2()
    w_low::T = missing | Beta(1,1)
    w_high::U = missing | Beta(1,1)
end
MixtureWeighter(weighter1, weighter2) = MixtureWeighter(;weighter1, weighter2)

function score(model::MixtureWeighter, u, control)
    (;weighter1, weighter2, w_low, w_high) = model
    w = control == "high" ? w_high : w_low
    p1 = score(weighter1, u); p1 ./= sum(p1)
    p2 = score(weighter2, u); p2 ./= sum(p2)
    (1-w) * p1 + w * p2
end

# @bounds @kwdef struct Sequenced{T,U,V} <: DependentWeighter
#     β_exp::T = missing | (0, 0.1, 1, Inf)
#     β_abs::U = missing | (0, 0.1, 1, Inf)
#     ev::V = 0. | (-Inf, -1, 1, Inf)
# end

# function score(model::Sequenced, u, control)
#     (;β_exp, β_abs, ev) = model
#     if control == "high"
#         @. exp(u * β_exp)
#     else @assert control == "low"
#         @. abs(u - ev) ^ β_abs
#     end
# end

# @bounds @kwdef struct SoftmaxUWS{T,U,V} <: SimpleWeighter
#     β::T = missing | (0, 0.1, 1, Inf)
#     β_abs::U = missing | (0, 0.1, 1, Inf)
#     ev::V = 0. | (-Inf, -1, 1, Inf)
# end

# function score(model::SoftmaxUWS, u)
#     (;β, β_abs, ev) = model
#     @. exp(u * β) * abs(u - ev) ^ β_abs
# end

# @bounds @kwdef struct AbsExp{T,U} <: SimpleWeighter
#     w::T = missing | (0, 0, 1, 1)
#     β::U = missing | (0, 0.1, 1, Inf)
# end

# function score(model::AbsExp, u)
#     (;w, β) = model
#     @. ((1 - w) * abs(u) + w * exp(u)) ^ β
# end

# %% ==================== Model ====================

@prior2 @kwdef struct Model{W<:AbstractWeighter,T,U}
    weighter::W = W()
    ε::T = 0 | Beta(1.1, 2)
    β_acc::U = missing | Exponential(3)  # accessibility weight    
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
    bb = map(collect(metaflatten(base_model, prior2, Missing))) do d
        quantile(d, [0, 0.1, 0.9, 1])
    end |> invert

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
