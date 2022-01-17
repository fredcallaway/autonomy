using Revise
includet("model.jl")
includet("utils.jl")
# includet("optimization.jl")
# includet("gp_min.jl")

using Profile, ProfileView
using BenchmarkTools

# %% --------
include("model.jl")
include("utils.jl")

# %% --------

env = Env(;n=100, k=2)
env = mutate(env, μ_outcome=-expected_value(env))
ev = Evaluator(env, 1000)

# %% --------
weighter = AbsExp(;β=0.5, d=1, dp=1)
est = BiasedMonteCarloEstimator(;n_sample=10, α=0.5, weighter)
ev(choice_objective, est)
@profview ev(choice_objective, est)
@btime ev(choice_objective, est)
@btime subjective_value(est, env, s; N=30);
@btime subjective_value(est, env, s; N=1);
@btime weight(est.weighter, s.u_init, s.p)

# %% --------
sv = subjective_value(est, env, s; N=1000);
std(sv)
sv2 = map(ev.states) do s
    mean(subjective_value(est, env, s; N=1000))
end
std(sv2)


# %% --------
# Profile.init(10000000, 1e-4)
f() = ev(choice_objective, est)
f()
@profview f()

# %% --------
s = State(env)
@btime subjective_value(est, env, s; N=100);
@btime subjective_value(est, env, s; N=1);
@btime subjective_value(est, env, s; N=2);
@btime subjective_value(est, env, s; N=10);
@profview 