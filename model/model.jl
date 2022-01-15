using Distributed
using StatsBase
using Distributions
using QuadGK
using Parameters
using Serialization
using Memoize
using FieldMetadata
using Combinatorics

include("utils.jl")

"Expected maximum of N samples from a Normal distribution"
function expected_maximum(k::Real, d::Normal)
    mcdf(x) = cdf(d, x)^k
    lo = d.μ - 10d.σ; hi = d.μ + 10d.σ
    - quadgk(mcdf, lo, 0, atol=1e-5)[1] + quadgk(x->1-mcdf(x), 0, hi, atol=1e-5)[1]
end

# https://stats.stackexchange.com/questions/303436/expected-value-of-srswor-sample-maximum
"Expected maximum of k samples (without replacement) from a set"
function expected_maximum_srswor(k::Real, u::Vector)
  N = length(u)
  u = sort(u)
  sum(u[i] * binomial(i-1, k-1) for i in k:N) / binomial(N, k)
end

# %% ==================== Environment ====================

@with_kw struct Env
    μ_outcome::Float64 = 0.
    signal::Float64 = 0
    σ_outcome::Float64 = 1.
    σ_init::Float64 = 0.
    k::Int = 2
    n::Int = 100
end

struct State
    u_true::Vector{Float64}
    u_init::Vector{Float64}
end

objective_value(s::State, k) = expected_maximum_srswor(k, s.u_true)

function State(env::Env)
    @unpack μ_outcome, signal, σ_outcome, σ_init, n = env
    u_true = μ_outcome + signal * randn() .+ σ_outcome .* randn(n)
    u_init = u_true + σ_init * randn(n)
    State(u_true, u_init)
end

function expected_value(env::Env)
    @unpack μ_outcome, σ_outcome, k = env
    expected_maximum(k, Normal(μ_outcome, σ_outcome))
end


# %% ==================== Sample weighting ====================

@metadata bounds (-Inf, Inf) Tuple

abstract type Weighter end
score(weighter::Weighter, u) = error("Not Implemented")

function weight(weighter::Weighter, u)::Weights
    x = score(weighter, u)
    x ./= sum(x)
    Weights(x, 1.)
end

struct NullWeighter <: Weighter end

function score(weighter::NullWeighter, u)
    ones(length(u))
end

@bounds @with_kw struct Softmax <: Weighter
    β_u::Real
    C::Real = 0 | (0, Inf)
end

function score(weighter::Softmax, u)
    @unpack β_u, C = weighter
    @. exp(u * β_u) + C
end


@bounds @with_kw struct AbsExp <: Weighter
    β::Real | (0, 1)
    d::Real | (-Inf, Inf)
    C::Real = 0 | (0, Inf)
end

function score(weighter::AbsExp, u)    
    @unpack β, d, C = weighter
    @. ((1 - β) * abs(u) + β * exp(u)) ^ d + C
end


@bounds @with_kw struct Analytic <: Weighter
    k::Real | (0, Inf)
    μ::Real = 0 | (-Inf, Inf)
    σ::Real = 1 | (0, Inf)
    C::Real = 0 | (0, Inf)
end
Analytic(k::Real) = Analytic(;k)

function score(weighter::Analytic, u)
    @unpack k, μ, σ, C = weighter
    @. cdf(Normal(μ, σ), u)^(k-1) + C
end

@with_kw struct AnalyticUWS <: Weighter
    k::Real
    μ::Real = 0
    σ::Real = 1
    C::Real = 0
    ev::Float64 = 0 # expected_maximum(k, Normal(μ, σ))
end
AnalyticUWS(k::Real) = AnalyticUWS(;k)

function score(weighter::AnalyticUWS, u)
    @unpack k, μ, σ, C, ev = weighter
    @. cdf(Normal(μ, σ), u)^(k-1) * abs(u - ev) + C
end

@with_kw struct SoftmaxUWS <: Weighter
    β::Real
    μ::Real = 0
    σ::Real = 1
    C::Real = 0
    ev::Float64 = 0 # expected_maximum(k, Normal(μ, σ))
end
SoftmaxUWS(β::Real) = SoftmaxUWS(;β)

function score(weighter::SoftmaxUWS, u)
    @unpack β, μ, σ, C, ev = weighter
    @. exp(u * β) * abs(u - ev) + C
end


# @bounds @with_kw struct Logistic <: Weighter
#     L::Float64 | (0, Inf)
#     k::Float64
#     x0::Float64
#     C::Real = 0 | (0, Inf)
# end

# @. logistic(x, (L, k, x0)) = L / (1 + exp(-k * (x - x0)))

# function score(weighter::Logistic, u)
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


# %% ==================== Estimation ====================

abstract type Estimator end

@with_kw struct MonteCarloEstimator <: Estimator
    n_sample::Int
end

function subjective_value(est::MonteCarloEstimator, env::Env, s::State)
    monte_carlo(est.n_sample) do
        maximum(sample(s.u_true, env.k; replace=false))
    end
end

@with_kw struct BiasedMonteCarloEstimator{W<:Weighter} <: Estimator
    n_sample::Int
    α::Real  # controls reweighting, in (0, 1)
    weighter::W
end

function subjective_value(est::BiasedMonteCarloEstimator, env::Env, s::State)
    w = weight(est.weighter, s.u_init)
    objective_prob = float(1/env.n) ^ env.k

    monte_carlo(est.n_sample) do
        if est.α == 0  # no reweighting
            maximum(sample(s.u_true, w, env.k; replace=false))
        else
            idx = sample(eachindex(s.u_true), w, env.k; replace=false)
            u_max = maximum(s.u_true[idx])
            reweight = objective_prob / prod(w.values[idx])
            @. u_max * reweight ^ est.α
        end
    end
end

@with_kw struct SampleMeanEstimator{W<:Weighter} <: Estimator
    n_sample::Int
    α::Real
    weighter::W
    replace::Bool = true
end

function subjective_value(est::SampleMeanEstimator, env::Env, s::State)
    w = weight(est.weighter, s.u_init)  # weights based on initial value estimates
    objective_prob = 1/env.n

    samples = if est.α == 0
        sample(s.u_true, Weights(w), est.n_sample; est.replace)
    else
        idx = sample(eachindex(s.u_true), w, est.n_sample; est.replace)
        @. s.u_true[idx] * (objective_prob  / w.values[idx]) ^ est.α
    end
    mean(samples)
end

# %% ==================== Evaluation ====================

struct Evaluator
    env::Env
    states::Vector{State}
    true_vals::Vector{Float64}
end

function Evaluator(env::Env, n_state=1000)
    states = map(1:n_state) do i
        State(env)
    end
    true_vals = map(s -> objective_value(s, env.k), states)
    Evaluator(env, states, true_vals)
end

choice_objective(v_subj, true_val) = (v_subj > 0) * true_val
error_objective(v_subj, true_val) = -(v_subj - true_val)^2

# function Evaluator(f::Symbol, args...)
#     objective = eval(Symbol(string(f, "_objective")))
#     Evaluator(objective, args...)
# end

function (ev::Evaluator)(objective::Function, est::Estimator)
    length(ev.states) \ mapreduce(+, ev.states, ev.true_vals) do s, true_val
        objective(subjective_value(est, ev.env, s), true_val)
    end
end

function mean_se(ev::Evaluator, objective::Function, est::Estimator)
    v = map(ev.states, ev.true_vals) do s, true_val
        objective(subjective_value(est, ev.env, s), true_val)
    end
    mean(v), sem(v)
end

function Base.show(io::IO, f::Evaluator) 
    print(io, "Evaluator")
end
