using StatsBase
using Distributions
using QuadGK
using Parameters

abstract type SampleWeighter end

function weight(sw::SampleWeighter, u, p)
    x = score(sw, u, p)
    x /= sum(x)
    x
end

subjective_value(sw::SampleWeighter, s, u, p) = mean(sample(u, Weights(weight(sw, u, p)), s; replace=false))
objective_value(u, p, k) = maximum(sample(u, Weights(p), k; replace=false))

choice_prob(sw::SampleWeighter, s, u, p) = N_MC \ mapreduce(+, 1:N_MC) do i
    subjective_value(sw, s, u, p) > 0
end

@with_kw struct Env
    loc::Float64 = 0.
    signal::Float64 = 0
    noise::Float64 = 1.
    dispersion::Float64 = 1.
    k::Int = 2
    n::Int = 100
    s::Int = 10
end

Problem = Tuple{Vector{Float64}, Vector{Float64}}

function sample_problem(env::Env)::Problem
    @unpack loc, signal, noise, dispersion, n = env
    u = loc + signal * randn() .+ noise .* randn(n)
    p = rand(Dirichlet(dispersion .* ones(n)))
    (u, p)
end

function expected_value(env::Env)
    @unpack loc, noise, k = env
    emax(k, Normal(loc, noise))
end

function monte_carlo(f, N=10000)
    N \ mapreduce(+, 1:N) do i
        f()
    end
end

"Expected maximum of N samples from a Normal distribution"
function emax(N::Int, d::Normal)
    mcdf(x) = cdf(d, x)^N
    lo = d.μ - 10d.σ; hi = d.μ + 10d.σ
    - quadgk(mcdf, lo, 0, atol=1e-5)[1] + quadgk(x->1-mcdf(x), 0, hi, atol=1e-5)[1]
end

# choice_value(u, p, k) = N_MC \ mapreduce(+, 1:N_MC) do i
#     objective_value(u, p, k)
# end

# function sample_value(n, k, s; αs=0:0.1:1)
#     u, p = sample_problem(n)
#     val = choice_value(u, p, k)
#     prob = [choice_prob(sw, s, u, p) for sw in αs]
#     prob .* val
# end

# %% ==================== Sampling strategies ====================


struct Multiplicative <: SampleWeighter
    β_p::Real
    β_u::Real
end

function score(sw::Multiplicative, u, p)
    @. u ^ sw.β_u * p ^ sw.β_p
end

struct Softmax <: SampleWeighter
    β_p::Real
    β_u::Real
end

function score(sw::Softmax, u, p)
    @. p ^ sw.β_p * exp(u * sw.β_u)
end

struct AbsExp <: SampleWeighter
    β_abs::Real
    β_exp::Real
end

function score(sw::AbsExp, u, p)
    @. p * (sw.β_abs * abs(u) + sw.β_exp * exp(u))
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
            objective_value(u, p, env.k)
        end
    end
    Objective(env, problems, true_vals)
end

function (f::Objective)(sw::SampleWeighter, map=map)
    choice_probs = map(f.problems) do (u, p)
        monte_carlo(1000) do 
            subjective_value(sw, f.env.s, u, p) > 0
        end
    end
    mean(choice_probs .* f.true_vals)
end

function Base.show(io::IO, f::Objective) 
    print(io, "Objective")
end
