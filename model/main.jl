using Serialization
using ProgressMeter
using AxisKeys

@everywhere include("model.jl")
include("utils.jl")
mkpath("tmp")

# %% ==================== boiler plate ====================

function make_objectives(;grid_kws...)
    G = grid(;grid_kws...)
    @showprogress map(G) do kw
        env = Env(;kw...)
        env = mutate(env, loc=-expected_value(env))
        Objective(env)
    end
end

function evaluate(name::String, strategies::KeyedArray{<:Evaluator}, objectives::KeyedArray{<:Objective})
    X = @showprogress pmap(Iterators.product(strategies, objectives)) do (sw, f)
        f(sw)
    end
    serialize("tmp/$name", X)
    X
end

# %% ==================== abs vs exp with importance ====================

evaluators = map(grid(β_exp=0:0.1:1, importance=0:0.1:1)) do (β_exp, importance)
    Evaluator(AbsExp(1 - β_exp, β_exp), importance)
end
objectives = make_objectives(k=[1,2,3], s=[1,5,25])
evaluate("abs_exp_importance", evaluators, objectives)