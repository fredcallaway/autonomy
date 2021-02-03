using Serialization
using ProgressMeter
using AxisKeys

@everywhere include("utils.jl")
@everywhere include("model.jl")
@everywhere include("gp_min.jl")
@everywhere include("box.jl")
mkpath("tmp")

# %% ==================== boiler plate ====================

function make_evaluators(n_state=1000; grid_kws...)
    G = grid(;grid_kws...)
    @showprogress map(G) do kw
        env = Env(;kw...)
        env = mutate(env, μ_outcome=-expected_value(env))
        Evaluator(env, n_state)
    end
end

function evaluate(name::String,
                  evaluators::KeyedArray{<:Evaluator}, 
                  objectives::KeyedArray{<:Function},
                  estimators::KeyedArray{<:Estimator},
                  N::Int = 100)
    X = @showprogress pmap(Iterators.product(evaluators, objectives, estimators)) do (ev, obj, est)
        mean_se(obj, est, N)
    end
    serialize("tmp/$name", X)
    table(X) |> CSV.write("results/$name.csv")
    X
end
bmap(f, x...) = pmap(f, x...; batch_size=10)

# %% ==================== Noisy weighting ====================

evaluators = make_evaluators(10_000, k=[1,2,4,8], σ_init=[0, 0.5, 1, 2])

objectives = map(keyed(:objective, [:choice])) do f
    eval(Symbol(string(f, "_objective")))
end

# %% --------
mc_basic = map(keyed(:n_sample, [1,5,25])) do s
    MonteCarloEstimator(n_sample)
end
evaluate("mc_basic", evaluators, objectives, mc_basic; N=200)

# %% --------
mc_absexp = map(grid(n_sample=[1,5,25], α=[0, 0.5, 1], β=0:.1:1, d=[0.5, 1,2,4,8])) do (n_sample, α, β, d)
    weighter = AbsExp(;β, d)
    BiasedMonteCarloEstimator(;n_sample, α, weighter)
end
evaluate("mc_absexp", evaluators, objectives, mc_absexp)

# %% --------
abs_exp = map(grid(n_sample=[1,5,25], α=[0, 0.5, 1], β=0:.1:1, d=[0.5, 1,2,4,8])) do (n_sample, α, β, d)
    weighter = AbsExp(;β, d)
    SampleEstimator(;n_sample, α, weighter)
end

evaluate("abs_exp", evaluators, objectives, abs_exp)

# %% ==================== Optimizing ====================

evaluators = make_evaluators(10_000, k=[1,2,4,8], σ_init=[0, 0.5, 1, 2])

objectives = map(keyed(:objective, [:choice, :error])) do f
    eval(Symbol(string(f, "_objective")))
end

n_samples = keyed(:n_sample, [1,5,25])

jobs = product(evaluators, objectives, n_samples)
# (ev, objective, n_sample) = first(jobs)
gp_results = @showprogress pmap(jobs) do (ev, objective, n_sample)
    box = Box(β=(0,1), d=(0, 10), α=(0,1))
    gp_minimize(length(box), verbose=false, iterations=300) do x
        @unpack β, d, α = box(x)
        weighter = AbsExp(;β, d)
        est = BiasedMonteCarloEstimator(;n_sample, α, weighter)
        -100ev(objective, est)
    end
end

# %% --------

# ================================= #
# ========= NOT FIXED YET ========= #
# ================================= #


for name in ["mc_basic", "mc_absexp", "abs_exp"]
    X = deserialize("tmp/$name")
    table(X) |> CSV.write("results/$name.csv")
end



# %% ==================== Analytic ====================
objectives = make_objectives(10000, k=1:3)

analytic = @showprogress pmap(Iterators.product(objectives, grid(s=[1,5,25]))) do (f, (s,))
    f(SampleEvaluator(5, 0, Analytic(f.env.k)))
end

serialize("tmp/analytic", analytic)

analytic_uws = @showprogress pmap(Iterators.product(objectives, grid(s=[1,5,25]))) do (f, (s,))
    f(SampleEvaluator(5, 0, AnalyticUWS(f.env.k)))
end

serialize("tmp/analytic_uws", analytic_uws)



# %% ==================== logistic vs softmax ====================


f = make_objectives(1000, k=[2]) |> only

f(SampleEvaluator(5, 0., Softmax(1., 1.2)), bmap)
f(SampleEvaluator(5, 0., fit(Logistic, f.env)), bmap)
f(SampleEvaluator(5, 0., AbsExpDp(0.3, 3, 3)), bmap)


X = getfield.(deserialize("tmp/bmc_vs_simple"), :simple)
keymax(X(;k=2,s=5,α=0))

# %% ==================== monte carlo ====================
objectives = make_objectives(1000, k=1:20)
evaluators = map(keyed(:s, [1,5,25])) do s
    MonteCarloEvaluator(s)
end
evaluate("monte_carlo", evaluators, objectives)

# %% ==================== biased monte carlo ====================
objectives = make_objectives(1000, k=1:20)
evaluators = map(grid(s=[1,5,25], α=[0], β=0:.1:1, d=1:4)) do (s, α, β, d)
    BiasedMonteCarloEvaluator(s, α, AbsExpDp(β, d, d))
end
evaluate("biased_monte_carlo", evaluators, objectives)

# %% ==================== comparison ====================

objectives = make_objectives(1000, k=[1,2,4,8])


bmc_vs_simple = @showprogress pmap(Iterators.product(objectives, grid(s=[1,5,25], α=[0], β=0:.1:1, d=1:4))) do (f, (s, α, β, d))
    bmc = f(BiasedMonteCarloEvaluator(s, α, AbsExpDp(β, d, d)))
    simple = f(SampleEvaluator(s*f.env.k, α, AbsExpDp(β, d, d)))
    (;bmc, simple)
end

serialize("tmp/bmc_vs_simple", bmc_vs_simple)
# %% --------

objectives = make_objectives(1000, k=[1,2,4,8])

bmc_vs_simple = @showprogress pmap(Iterators.product(objectives, grid(s=[1000], α=[0], β=0:.1:1, d=0:0.2:2))) do (f, (s, α, β, d))
    bmc = f(BiasedMonteCarloEvaluator(s, α, AbsExpDp(β, d, d)))
    simple = f(SampleEvaluator(s*f.env.k, α, AbsExpDp(β, d, d)))
    (;bmc, simple)
end

serialize("tmp/bmc_vs_simple_large_s", bmc_vs_simple)

# %% ==================== abs vs exp ====================

evaluators = map(grid(β=0:0.05:1, α=0:.1:1, d=1:.25:4, s=[1,5,25])) do (β, α, d)
        SampleEvaluator(s, α, AbsExp(;β, d, dp=d))
end
objectives = make_objectives(10000, k=[1,2,4,8])
length(evaluators) * length(objectives)
evaluate("abs_exp_full", evaluators, objectives)

# %% --------

# objectives = make_objectives(1000, k=1:20)
# evaluators = map(grid(β=0:0.05:1, α=[0.], d=[4.], s=[1,5,25])) do (β, α, d)
#     SampleEvaluator(s, α, AbsExp(;β, d, dp=d))
# end

# evaluate("abs_exp_β", evaluators, objectives)

# %% --------
# ALL BELOW IS BROKEN


# %% ==================== scratch ====================
X = deserialize("tmp/abs_exp_det");
x = X(k=2, s=1, α=0, dispersion=1e10)

f = objectives(k=2, s=1) |> only
@unpack β, d = keymax(x)

map(0:.2:1) do x
    f(Evaluator(AbsExpD(β, d+x), 0.), bmap)
end





# %% --------
f = make_objectives(1000; k=[2], s=[5], dispersion=[1e10])[1]

bmap(f, xs) = pmap(f, xs; batch_size=10)
f(Evaluator(AbsExpD(0.5, 1), 0.), bmap)

# %% --------

