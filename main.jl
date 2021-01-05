using Serialization
using ProgressMeter
using AxisKeys

@everywhere include("model.jl")
include("figure.jl")
mkpath("tmp")

# %% --------
@everywhere begin
    Problem = Tuple{Vector{Float64}, Vector{Float64}}
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
                subjective_value(sw, env.s, u, p) > 0
            end
        end
        mean(choice_probs .* f.true_vals)
    end

    function Base.show(io::IO, f::Objective) 
        print(io, "Objective")
    end
end

# %% ==================== scratch ====================
env = Env(k=1, n=10)
f = Objective(env)
f(Softmax(1,0), pmap)


# %% ==================== modulate β_p and β_u ====================

env = Env(loc= -0.56)
objective = Objective(env)
@everywhere objective = $objective

G = grid(β_p=0:0.5:20, β_u=0:0.5:20)

O = @showprogress pmap(G) do x
    objective(Softmax(x...))
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

