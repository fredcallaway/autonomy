using Serialization
using ProgressMeter
using AxisKeys

@everywhere include("model.jl")
include("utils.jl")
mkpath("tmp")

# %% ==================== boiler plate ====================

function make_objectives(n_problem=1000; grid_kws...)
    G = grid(;grid_kws...)
    @showprogress map(G) do kw
        env = Env(;kw...)
        env = mutate(env, loc=-expected_value(env))
        Objective(env, n_problem)
    end
end

function evaluate(name::String, strategies::KeyedArray{<:Evaluator}, objectives::KeyedArray{<:Objective})
    X = @showprogress pmap(Iterators.product(strategies, objectives)) do (sw, f)
        f(sw)
    end
    serialize("tmp/$name", X)
    X
end
bmap(f, xs) = pmap(f, xs; batch_size=10)

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

# ALL BELOW IS BROKEN

evaluators = map(grid(β=0:0.1:1, α=0:.1:1, d=1:.3:4)) do (β, α, d)
    Evaluator(AbsExpDp(β, d, d), α)
end
objectives = make_objectives(1000, k=[1,2,3], s=[1,5,25], dispersion=[1e10])
evaluate("abs_exp_full", evaluators, objectives)

# %% --------

evaluators = map(grid(β=0:0.05:1, α=[0.], d=[4.])) do (β, α, d)
    Evaluator(AbsExpDp(β, d, d), α)
end

objectives = make_objectives(1000, k=1:20, s=[1,5,25], dispersion=[1e10])
evaluate("abs_exp_β_many", evaluators, objectives)

# %% --------
@everywhere using Optim
@everywhere function optimal_β(f; d, α)
    res = optimize(0, 1; abs_tol=.01) do β
        -f(Evaluator(AbsExpDp(β, d, d), α))
    end
    res.minimizer
end

objectives = make_objectives(1000, k=1:20, s=[1,5,25], dispersion=[1e10])
opt_βs = @showprogress pmap(objectives) do f
    optimal_β(f; d=4, α=0.)
end
serialize("tmp/opt_βs", opt_βs)

# %% --------

function optimal_β_local(f; d, α)
    res = optimize(0, 1; abs_tol=.01) do β
        y = f(Evaluator(AbsExpDp(β, d, d), α), bmap)
        println(β => y)
        -y
    end
    res.minimizer
end

f = objectives(s=25)[end]
optimal_β_local(f, d=4, α=0.)



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

