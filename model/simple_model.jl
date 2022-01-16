using FieldMetadata
using Flatten
import Base: @kwdef
flatten = Flatten.flatten; reconstruct = Flatten.reconstruct

include("random_restarts.jl")

@metadata bounds (-Inf, -Inf, Inf, Inf) Tuple

abstract type AbstractWeighter end
score(::AbstractWeighter, u) = error("Not Implemented")


struct NullWeighter <: AbstractWeighter end

function score(model::NullWeighter, u)
    ones(length(u))
end

@bounds @kwdef struct Softmax{T} <: AbstractWeighter
    β::T = missing | (0, 0, 1, Inf)
end

function score(model::Softmax, u)
    @. exp(u * model.β) 
end

@bounds @kwdef struct AbsExp{T,U} <: AbstractWeighter
    w::T = missing | (0, 0, 1, 1)
    β::U = missing | (0, 0, 1, Inf)
end

function score(model::AbsExp, u)
    (;w, β) = model
    @. ((1 - w) * abs(u) + w * exp(u)) ^ β
end

@bounds @kwdef struct UWS{T,U} <: AbstractWeighter
    ev::T = 0. | (-Inf, -10, 10, Inf)
    β_abs::U = missing | (-Inf, -10, 10, Inf)
end

function score(model::UWS, u)
    (;ev, β_abs) = model
    @. abs(u - ev) ^ β_abs
end

@bounds @kwdef struct SoftmaxUWS{T,U,V} <: AbstractWeighter
    β::T = mising | (0, 0, 1, Inf)
    β_abs::U = missing | (-Inf, -10, 10, Inf)
    ev::V = 0. | (-Inf, -10, 10, Inf)
end

function score(model::SoftmaxUWS, u)
    (;β, β_abs, ev) = model
    @. exp(u * β) * abs(u - ev) ^ β_abs
end

@bounds @kwdef struct Model{W<:AbstractWeighter,T,U}
    weighter::W = W()
    ε::T = missing | (0, 0, 1, 1)
    β_acc::U = missing | (0, 0, 1, Inf)  # accessibility weight    
end

Model(weighter::AbstractWeighter; kws...) = Model(;weighter, kws...)
Model{W}(;ε=missing, β_acc=missing, kws...) where W = Model(W(;kws...); ε, β_acc)

function likelihood(model::Model, trial::NamedTuple)
    (;weighter, ε, β_acc) = model
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

function foo(model::Model{W}) where W
    W
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
