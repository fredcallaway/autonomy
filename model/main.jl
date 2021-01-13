using Serialization
using ProgressMeter
using AxisKeys

@everywhere include("model.jl")
include("utils.jl")
include("figure.jl")
mkpath("tmp")

# %% ==================== boiler plate ====================

function make_objectives(;grid_kws...)
    G = grid(;grid_kws...)
    @showprogress map(G) do kw
        env = Env(;kw...)
        env = mutate(env, loc=-expected_value(env))
        Objective(env, 10)
    end
end

function evaluate(name::String, strategies::KeyedArray{<:SampleWeighter}, objectives::KeyedArray{<:Objective})
    X = @showprogress pmap(Iterators.product(strategies, objectives)) do (sw, f)
        f(sw)
    end
    serialize("tmp/$name", X)
    X
end

# %% ==================== scratch ====================

env = Env(k=1, dispersion=1e10)
f = Objective(env)

f(AbsExp(1, 0), pmap)
f(AbsExp(0, 1), pmap)

f(Multiplicative(0, 0), pmap)

# %% ==================== abs exp ====================

strategies = map(grid(β_exp=0:0.1:1)) do (β_exp,)
    AbsExp(1 - β_exp, β_exp)
end
objectives = make_objectives(k=2 .^ (0:3))
evaluate("abs_exp", strategies, objectives)

# %% ==================== abs exp with s ====================

strategies = map(grid(β_exp=0:0.1:1)) do (β_exp,)
    AbsExp(1 - β_exp, β_exp)
end
objectives = make_objectives(s=5 .^ (0:2))
evaluate("tmp/abs_exp_s", strategies, objectives)

# %% ==================== abs exp full ====================

strategies = map(grid(β_abs=0:0.2:2, β_exp=0:0.2:2)) do kw
    AbsExp(kw...)
end
objectives = make_objectives(k=2 .^ (0:2))
evaluate("tmp/abs_exp_full", strategies, objectives)


# %% ==================== Optimization ====================

function wrap_objective(::Type{SW}, objective::Function) where SW <: SampleWeighter
    function wrapped(x)
        objective(SW(x...))
    end
end

f = wrap_objective(Softmax, objective)

