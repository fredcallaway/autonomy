using Distributed
using StatsBase
using Distributions
using QuadGK
using Parameters
using Serialization
include("utils.jl")
# %% ==================== Environment ====================

@with_kw struct Env
    loc::Float64 = 0.
    signal::Float64 = 0
    noise::Float64 = 1.
    dispersion::Float64 = 1.
    k::Int = 2
    n::Int = 100
end

Problem = Tuple{Vector{Float64}, Vector{Float64}}

function sample_problem(env::Env)::Problem
    @unpack loc, signal, noise, dispersion, n = env
    u = loc + signal * randn() .+ noise .* randn(n)
    p = rand(Dirichlet(dispersion .* ones(n)))
    (u, p)
end

sample_objective_value(u, p, k) = maximum(sample(u, Weights(p), k; replace=false))

"Expected maximum of N samples from a Normal distribution"
function expected_maximum(N::Int, d::Normal)
    mcdf(x) = cdf(d, x)^N
    lo = d.μ - 10d.σ; hi = d.μ + 10d.σ
    - quadgk(mcdf, lo, 0, atol=1e-5)[1] + quadgk(x->1-mcdf(x), 0, hi, atol=1e-5)[1]
end

function expected_value(env::Env)
    @unpack loc, noise, k = env
    expected_maximum(k, Normal(loc, noise))
end


# %% ==================== Sampling strategies ====================

abstract type Sampler end

@with_kw struct Multiplicative <: Sampler
    β_p::Real
    β_u::Real
end

function score(sampler::Multiplicative, u, p)
    @. u ^ sampler.β_u * p ^ sampler.β_p
end

@with_kw struct Softmax <: Sampler
    β_p::Real
    β_u::Real
end

function score(sampler::Softmax, u, p)
    @. p ^ sampler.β_p * exp(u * sampler.β_u)
end

@with_kw struct AbsExp <: Sampler
    β_abs::Real
    β_exp::Real
end

function score(sampler::AbsExp, u, p)
    if sampler.β_abs == sampler.β_exp == 0
        return p
    else
        @. p * (sampler.β_abs * abs(u) + sampler.β_exp * exp(u))
    end
end

@with_kw struct AbsExpD <: Sampler
    β::Real
    d::Real
end

function score(sampler::AbsExpD, u, p)    
    @unpack β, d = sampler
    @. p * ((1 - β) * abs(u) + β * exp(u)) ^ d
end

@with_kw struct AbsExpDp <: Sampler
    β::Real
    d::Real
    dp::Real
end

function score(sampler::AbsExpDp, u, p)    
    @unpack β, d, dp = sampler
    @. p^dp * ((1 - β) * abs(u) + β * exp(u)) ^ d
end


# %% ==================== Evaluation ====================

abstract type Evaluator end

@with_kw struct MonteCarloEvaluator <: Evaluator
    s::Int
end

@everywhere function subjective_value(evaluator::MonteCarloEvaluator, env::Env, u, p)
    mean(sample_objective_value(u, p, env.k) for i in 1:evaluator.s)
end


@with_kw struct SampleEvaluator{S<:Sampler} <: Evaluator
    s::Int
    α::Real
    sampler::S
end

function subjective_value(evaluator::SampleEvaluator, env::Env, u, p)
    w = weight(evaluator.sampler, u, p)
    samples = if evaluator.α > 0  # reweighting
        idx = sample(eachindex(u), Weights(w), evaluator.s; replace=false)
        @. u[idx] * (p[idx] / w[idx]) ^ evaluator.α
    else
        sample(u, Weights(w), evaluator.s; replace=false)
    end
    mean(samples)
end

function weight(sampler::Sampler, u, p)
    x = score(sampler, u, p)
    x ./= sum(x)
    x
end

# %% ==================== Objective ====================

struct Objective
    env::Env
    problems::Vector{Problem}
    true_vals::Vector{Float64}
end

function Objective(env, n_problem=1000)
    problems = map(1:n_problem) do i
        sample_problem(env)
    end
    true_vals = pmap(problems) do (u, p)
        monte_carlo(10000) do
            sample_objective_value(u, p, env.k)
        end
    end
    Objective(env, problems, true_vals)
end

function (f::Objective)(evaluator::Evaluator, map=map)
    choice_probs = map(f.problems) do (u, p)
        monte_carlo(1000) do 
            subjective_value(evaluator, f.env, u, p) > 0
        end
    end
    mean(choice_probs .* f.true_vals)
end

function Base.show(io::IO, f::Objective) 
    print(io, "Objective")
end
