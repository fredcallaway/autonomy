using Distributed
using StatsBase
using Distributions
using QuadGK
using Parameters
using Serialization
using Memoize
using FieldMetadata
# include("utils.jl")
# %% ==================== Environment ====================

@with_kw struct Env
    μ_outcome::Float64 = 0.
    signal::Float64 = 0
    σ_outcome::Float64 = 1.
    σ_init::Float64 = 1.
    dispersion::Float64 = 1e10
    k::Int = 2
    n::Int = 100
end

struct State
    u_true::Vector{Float64}
    p::Vector{Float64}
    u_init::Vector{Float64}
end

sample_objective_value(u, p, k) = maximum(sample(u, Weights(p), k; replace=false))
sample_objective_value(s::State, k) = sample_objective_value(s.u_true, s.p, k)

function State(env::Env)
    @unpack μ_outcome, signal, σ_outcome, σ_init, dispersion, n = env
    u_true = μ_outcome + signal * randn() .+ σ_outcome .* randn(n)
    u_init = u_true + σ_init * randn(n)
    p = rand(Dirichlet(dispersion .* ones(n)))
    State(u_true, p, u_init)
end

"Expected maximum of N samples from a Normal distribution"
@memoize function expected_maximum(N::Real, d::Normal)
    mcdf(x) = cdf(d, x)^N
    lo = d.μ - 10d.σ; hi = d.μ + 10d.σ
    - quadgk(mcdf, lo, 0, atol=1e-5)[1] + quadgk(x->1-mcdf(x), 0, hi, atol=1e-5)[1]
end

function expected_value(env::Env)
    @unpack μ_outcome, σ_outcome, k = env
    expected_maximum(k, Normal(μ_outcome, σ_outcome))
end


# %% ==================== Sampling strategies ====================
@metadata bounds (-Inf, Inf) Tuple

abstract type Weighter end
score(weighter::Weighter, u, p) = error("Not Implemented")

function weight(weighter::Weighter, u, p)::Weights
    x = score(weighter, u, p)
    x ./= sum(x)
    Weights(x, 1.)
end

@bounds @with_kw struct Softmax <: Weighter
    β_u::Real
    β_p::Real = 1 | (0, Inf)   # default value and bounds
    C::Real = 0 | (0, Inf)
end

function score(weighter::Softmax, u, p)
    @unpack β_p, β_u, C = weighter
    @. p ^ β_p * exp(u * β_u) + C
end


@bounds @with_kw struct AbsExp <: Weighter
    β::Real | (0, 1)
    d::Real | (-Inf, Inf)
    dp::Real = 1 | (-Inf, Inf)
    C::Real = 0 | (0, Inf)
end

function score(weighter::AbsExp, u, p)    
    @unpack β, d, dp, C = weighter
    @. p^dp * ((1 - β) * abs(u) + β * exp(u)) ^ d + C
end


@bounds @with_kw struct Analytic <: Weighter
    k::Real | (0, Inf)
    μ::Real = 0 | (-Inf, Inf)
    σ::Real = 1 | (0, Inf)
    C::Real = 0 | (0, Inf)
end
Analytic(k::Real) = Analytic(;k)

function score(weighter::Analytic, u, p)
    @unpack k, μ, σ, C = weighter
    @. p * cdf(Normal(μ, σ), u)^(k-1) + C
end

@with_kw struct AnalyticUWS <: Weighter
    k::Real
    μ::Real = 0
    σ::Real = 1
    C::Real = 0
end
AnalyticUWS(k::Real) = AnalyticUWS(;k)

function score(weighter::AnalyticUWS, u, p)
    @unpack k, μ, σ = weighter
    ev = expected_maximum(k, Normal(μ, σ))
    @. p * cdf(Normal(0, 1), u)^(k-1) * abs(u - ev)
end

# @bounds @with_kw struct Logistic <: Weighter
#     L::Float64 | (0, Inf)
#     k::Float64
#     x0::Float64
#     C::Real = 0 | (0, Inf)
# end

# @. logistic(x, (L, k, x0)) = L / (1 + exp(-k * (x - x0)))

# function score(weighter::Logistic, u, p)
#     @unpack L, k, x0, C = weighter
#     p .* logistic(u, (L, k, x0)) + C
# end

# using LsqFit
# function Distributions.fit(::Type{Logistic}, env::Env)
#     d = Normal(env.μ_outcome, env.σ_outcome)
#     x = -6:.001:6
#     cdf_max = @. cdf(d, x) ^ env.k
#     pdf_max = diff(cdf_max) .* 1000
#     pdf_sample = pdf.(d, x[2:end])
#     fit = curve_fit(logistic, x[2:end], pdf_max ./ pdf_sample, [env.k, 2.5, 1])
#     Logistic(fit.param...)
# end


# %% ==================== Evaluation ====================

abstract type Evaluator end

# function subjective_value(evaluator::Evaluator, env::Env, s::State; N=1)
#     subjective_value(evaluator, env, s.u_init, s.p; N)
# end

@with_kw struct MonteCarloEvaluator <: Evaluator
    n_sample::Int
end


function subjective_value(evaluator::MonteCarloEvaluator, env::Env, s::State; N=1)
    [mean(sample_objective_value(s, env.k) for i in 1:evaluator.n_sample) for i in 1:N]
end

@with_kw struct BiasedMonteCarloEvaluator{W<:Weighter} <: Evaluator
    n_sample::Int
    α::Real
    weighter::W
end

function subjective_value(evaluator::BiasedMonteCarloEvaluator, env::Env, s::State; N=1)
    w = weight(evaluator.weighter, s.u_init, s.p)
    map(1:N) do i
        samples = if evaluator.α > 0  # reweighting
            idx = sample(eachindex(s.u_true), w, (evaluator.n_sample, env.k); replace=true)
            u_max = maximum(s.u_true[idx]; dims=2)
            reweight = prod(p[idx] ./ w[idx]; dims=2)
            @. u_max * reweight ^ evaluator.α
        else
            maximum(sample(s.u_true, Weights(w), (evaluator.n_sample, env.k); replace=true); dims=2)
        end
        mean(samples)
    end
end

@with_kw struct SampleEvaluator{W<:Weighter} <: Evaluator
    n_sample::Int
    α::Real
    weighter::W
    replace::Bool = true
end

function subjective_value(evaluator::SampleEvaluator, env::Env, s::State; N=1)
    w = weight(evaluator.weighter, s.u_init, s.p)  # weights based on initial value estimates
    map(1:N) do i
        samples = if evaluator.α > 0  # reweighting
            idx = sample(eachindex(s.u_true), w, evaluator.n_sample; evaluator.replace)
            @. s.u_true[idx] * (s.p[idx] / w[idx]) ^ evaluator.α
        else
            sample(s.u_true, Weights(w), evaluator.n_sample; evaluator.replace)
        end
        mean(samples)
    end
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
        mean(subjective_value(evaluator, f.env, u, p; N=100) .> 0)
    end
    mean(choice_probs .* f.true_vals)
end

function Base.show(io::IO, f::Objective) 
    print(io, "Objective")
end
