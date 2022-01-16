using FieldMetadata
using Flatten
flatten = Flatten.flatten; reconstruct = Flatten.reconstruct
using Parameters

include("random_restarts.jl")

@metadata bounds (-Inf, -Inf, Inf, Inf) Tuple

abstract type AbstractWeighter end
score(::AbstractWeighter, u) = error("Not Implemented")


struct NullWeighter <: AbstractWeighter end

function score(model::NullWeighter, u)
    ones(length(u))
end


@bounds @with_kw struct Softmax{T<:Real} <: AbstractWeighter
    β::T | (0, 0, 1, Inf)
end

function score(model::Softmax, u)
    @. exp(u * model.β) 
end


@bounds @with_kw struct AbsExp{T<:Real} <: AbstractWeighter
    w::T | (0, 0, 1, 1)
    β::T | (0, 0, 1, Inf)
end

function score(model::AbsExp, u)
    @unpack w, β = model
    @. ((1 - w) * abs(u) + w * exp(u)) ^ β
end


@bounds @with_kw struct UWS{T<:Real} <: AbstractWeighter
    ev::T = 0. | (-Inf, -10, 10, Inf)
    β_abs::T | (-Inf, -10, 10, Inf)
end

function score(model::UWS, u)
    @unpack ev, β_abs = model
    @. abs(u - ev) ^ β_abs
end

@bounds @with_kw struct SoftmaxUWS{T<:Real} <: AbstractWeighter
    β::T | (0, 0, 1, Inf)
    β_abs | (-Inf, -10, 10, Inf)
    ev::T = 0. | (-Inf, -10, 10, Inf)
end

function score(model::SoftmaxUWS, u)
    @unpack β, β_abs, ev = model
    @. exp(u * β) * abs(u - ev) ^ β_abs
end


@bounds struct Model{W<:AbstractWeighter,T<:Real}
    weighter::W
    ε::T | (0, 0, 1, 1)
    β_acc::T | (0, 0, 1, Inf)  # accessibility weight    
end

function Model{W}() where W
    weighter = W(fill(NaN, length(fieldnames(W)))...)
    Model(weighter, NaN, NaN)
end


function likelihood(model::Model, trial::NamedTuple)
    @unpack weighter, ε, β_acc = model
    p = score(weighter, trial.evaluation)
    n = length(p)
    @. begin
        p /= $sum(p)
        p += β_acc * trial.accessibility
        p /= $sum(p)
        p *= (1 - ε)
        p += ε * (1/n)
    end
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
    println(io, W)
    for (k, v) in zip(fieldnameflatten(model), flatten(model))
        println(io, lpad(k, 7), ": ", v)
    end
end

function find_mle(M::Type{<:Model}, trials; n_restart = 10)
    base_model = M()

    hard_lower, soft_lower, soft_upper, hard_upper = map(invert(metaflatten(base_model, bounds))) do x
        collect(float.(x))
    end
    @assert all(isfinite.(soft_lower)) && all(isfinite.(soft_upper))
    loss(x) = -logp(reconstruct(base_model, x), trials)

    res = random_restarts(loss, hard_lower, soft_lower, soft_upper, hard_upper, n_restart)
    reconstruct(base_model, res.minimizer)
end
