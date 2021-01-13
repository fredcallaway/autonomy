using Serialization
using ProgressMeter
using AxisKeys

@everywhere include("model.jl")
include("utils.jl")
include("figure.jl")
mkpath("tmp")

# %% ==================== scratch ====================
env = Env(k=1, dispersion=1e10)
f = Objective(env)

f(AbsExp(1, 0), pmap)
f(AbsExp(0, 1), pmap)

f(Multiplicative(0, 0), pmap)


# %% ==================== abs exp ====================

ks = keyed(:k, 2 .^ (0:3))
objectives = @showprogress map(ks) do k
    env = Env(;k)
    env = mutate(env, loc=-expected_value(env))
    Objective(env, 10000)
end;
β_exps = keyed(:β_exp, 0:0.1:1)

X = @showprogress pmap(Iterators.product(objectives, β_exps)) do (f, β_exp)
    f(AbsExp(1 - β_exp, β_exp))
end

serialize("tmp/abs_exp", X)

# %% --------
X = deserialize("tmp/abs_exp")
figure() do
    plot(axiskeys(X, 2), X', 
        label=reshape(["k=$k" for k in axiskeys(X, 1)], 1, :),
        # palette=collect(cgrad(:blues, 6, rev=true, categorical=true)),
        xlabel="β_u", ylabel="Reward", legend=:topleft, 
    )
end

# %% ==================== abs exp with n ====================

objectives = @showprogress map(keyed(:s, 5 .^ (0:2))) do s
    env = Env(;s, k=2)
    env = mutate(env, loc=-expected_value(env))
    Objective(env, 10000)
end;
G = Iterators.product(objectives, keyed(:β_abs, 0:0.2:2))
X = @showprogress pmap(G) do (f, β_exp)
    f(AbsExp(1-β_exp, β_exp))
end

serialize("tmp/abs_exp_s", X)


# %% ==================== abs exp full ====================


ks = keyed(:k, 2 .^ (0:2))
objectives = @showprogress map(ks) do k
    env = Env(;k)
    env = mutate(env, loc=-expected_value(env))
    Objective(env, 10000)
end;
G = Iterators.product(objectives, keyed(:β_abs, 0:0.2:2), keyed(:β_exp, 0:0.2:2))
X = @showprogress pmap(G) do (f, β_abs, β_exp)
    f(AbsExp(β_abs, β_exp))
end

serialize("tmp/abs_exp_full", X)
# %% --------
X = deserialize("tmp/abs_exp_full")

figure("abs_exp_grids") do
    clim = (0, maximum(X))
    ps = map(axiskeys(X, :k)) do k
        heatmap(X(k=k); clim, cbar=false, title="k = $k")
    end
    plot(ps..., size=(900,300), layout=(1,3), bottom_margin=4mm)
end




# %% ==================== modulate k and β_u ====================

ks = keyed(:k, 2 .^ (0:4))
objectives = @showprogress map(ks) do k
    env = Env(;k)
    env = mutate(env, loc=-expected_value(env))
    Objective(env, 10000)
end;

# %% --------
β_us = keyed(:β_u, -1:0.05:10)
X = @showprogress pmap(Iterators.product(objectives, β_us)) do (objective, β_u)
    objective(Softmax(1, β_u))
end

serialize("tmp/k-β_u-0", X)
# figure("heat-k-β_u") do
#     heatmap(X)
# end

# %% --------
X = deserialize("tmp/k-β_u-0")
figure("k-β_u") do
    plot(axiskeys(X, 2), X', 
        label=reshape(["k=$k" for k in axiskeys(X, 1)], 1, :),
        palette=collect(cgrad(:blues, 6, rev=true, categorical=true)),
        xlabel="β_u", ylabel="Reward", legend=:topleft, 
    )
end

# %% ==================== k and β_u and β_p ====================
objectives = @showprogress map(keyed(:k, 2 .^ (0:2))) do k
    env = Env(;k)
    env = mutate(env, loc=-expected_value(env))
    Objective(env, 10000)
end;

G = Iterators.product(keyed(:β_u, 0:0.5:10), keyed(:β_p, 0:0.5:10), objectives)
X = @showprogress pmap(G) do (β_u, β_p, objective)
    objective(Softmax(β_p, β_u))
end
serialize("tmp/full_grid", X)

# %% --------
function Plots.heatmap(X::KeyedArray{<:Real,2}; kws...)
    ylabel, xlabel = dimnames(X)
    heatmap(reverse(axiskeys(X))..., X; xlabel, ylabel, kws...)
end

figure() do
    heatmap(X)
    # heatmap(axiskeys(x, 2), axiskeys(x, 1), x, xlabel=)
end


# %% --------
using Plots.Measures
X = deserialize("tmp/full_grid")
figure("full_grid") do
    ps = map(axiskeys(X, :k)) do k
        heatmap(X(k=k), clim=(0, .05), cbar=false, title="k = $k")
    end
    plot(ps..., size=(900,300), layout=(1,3), bottom_margin=4mm)
end

# %% --------
function foo(objective)
    pmap(-1:0.1:1) do β_u
        objective(Softmax(1, β_u))
    end
end
y = foo(objectives[1])


# %% ==================== Optimization ====================

function wrap_objective(::Type{SW}, objective::Function) where SW <: SampleWeighter
    function wrapped(x)
        objective(SW(x...))
    end
end

f = wrap_objective(Softmax, objective)

f([0., 0])
# %% --------



objective(Softmax(1, 3))












# %% --------
K = map(Iterators.product([1,2,3], [1, 2, 4, 8])) do (k, s)
    res = @showprogress pmap(1:5000; batch_size=10) do i
        sample_value(100, k, s);
    end
    sum(res) ./ 5000
end

serialize("tmp/K", K)

# %% --------
baseline = map([1,2,3]) do k
    res = @showprogress pmap(1:5000; batch_size=10) do i
        u, p = sample_problem(100)
        val = choice_value(u, p, k)
        prob =  N_MC \ mapreduce(+, 1:N_MC) do i
            objective_value(u, p, k) > 0
        end
        prob * val
    end
    sum(res) ./ 5000
end
serialize("tmp/baseline", baseline)
# %% --------
comp = map([1,2,3]) do k
    res = @showprogress pmap(1:5000; batch_size=10) do i
        u, p = sample_problem(100)
        val = choice_value(u, p, k)

        p1 = N_MC \ mapreduce(+, 1:N_MC) do i
            objective_value(u, p, k) > 0
        end
        p2 = choice_prob(1., k, u, p)
        p1 * val - p2 * val
    end
    sum(res) ./ 5000
end




# %% ==================== Plotting ====================
gr(label="")
K = deserialize("tmp/K")
baseline = deserialize("tmp/baseline")

figure() do
    pp = map(eachcol(K), [1, 2, 4, 8]) do kk, s
        plot(0:0.1:1, kk, 
            # title="$s samples", 
            xlabel="α", ylabel="Utility",
            label=["k=1" "k=2" "k=3"], legend=:bottomright)
    end
    plot(pp[1])
end
# plot!(pp[1], )

