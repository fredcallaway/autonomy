using Serialization
using ProgressMeter

@everywhere include("model.jl")
include("utils.jl")
mkpath("tmp")
include("figure.jl")

# %% --------

@everywhere function choice_prob(evaluator::Evaluator, env::Env, shift::Vector)
    q1, q2 = quantile(Normal(env.loc, env.noise), [0.25, 0.75])
    p_choose = monte_carlo(1000) do 
        u, p = sample_problem(env)
        u[u .< q1] .+= shift[1]
        u[q1 .< u .< q2] .+= shift[2]
        u[u .> q2 ] .+= shift[3]
        mean(subjective_value(evaluator, env, u, p; N=100) .> 0)
    end
end

@everywhere function shift_effect(evaluator, env)
    cp1 = map(1:3) do i
        shift = zeros(3)
        shift[i] += .2
        choice_prob(evaluator, env, shift)
    end
    cp1
end

@everywhere function balanced_env(;kw...)
    env = Env(;kw...)
    mutate(env, loc=-expected_value(env))
end

S = @showprogress pmap(grid(k=1:3, s=[1,5,25], d=1:0.3:4, β=0:0.1:1)) do (k, s, d, β)
    env = balanced_env(;k)
    evaluator = SampleEvaluator(s, 0., AbsExpD(β, d))
    shift_effect(evaluator, env)
end

baseline = @showprogress pmap(grid(k=1:3, s=[1,5,25])) do (k, s)
    env = balanced_env(;k)
    evaluator = MonteCarloEvaluator(s)
    shift_effect(evaluator, env)
end

serialize("tmp/behavior", (S, baseline))
