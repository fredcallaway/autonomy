using Serialization
using ProgressMeter
@everywhere include("model.jl")
mkpath("tmp")

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

