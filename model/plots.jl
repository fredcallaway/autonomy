include("figure.jl")
include("utils.jl")

# %% ==================== Behavior ====================

S, baseline = deserialize("tmp/behavior");
opt_βs = deserialize("tmp/opt_βs")(dispersion=1e10)

figure("accept_comparison") do
    ps = map(Iterators.product(axiskeys(S, :k), axiskeys(S, :s))) do (k, s)
        x = S(;k,s, d=4)
        plot(0:0.1:1, hcat(x...)', xlabel="β", ylabel="Probability Accept", label=["low" "medium" "high"],
            legend= (k, s) == (1, 1) ? :topleft : false)
        hline!(permutedims(baseline(;k, s)), color=permutedims(palette(:default)[1:3]), ls=:dash)
        vline!([opt_βs(;k, s)], color=:gray, alpha=0.5, lw=1)
    end
    plot(ps..., size=(900,900), layout=(3,3), bottom_margin=4mm)
end

# %% --------
full = deserialize("tmp/abs_exp_full")(dispersion=1e10)

figure("accept_comparison_simplified") do
    ps = map(Iterators.product(axiskeys(S, :k), axiskeys(S, :s))) do (k, s)
        plot(baseline(;k, s), label="Monte Carlo", line=(3, :black))

        X = mapreduce(hcat, axiskeys(S, :d)) do d
            β = keymax(full(;k, s, d, α=0))
            S(;k,s,β,d)
        end
        X = X[:, 1:2:end]

        plot!(X, palette=collect(cgrad(:Blues, size(X, 1)+1, categorical = true))[2:end])
        plot!(
            # legend= (k, s) == (1, 1) ? :bottomleft : false,
            legend=false,
            yaxis="Probability Accept",
            xaxis=("Shifted group", (1:3, ["Low", "Medium", "High"])),
            title="k=$k, s=$s",
        )
    end
    plot(ps..., size=(900,900), layout=(3,3), bottom_margin=4mm)
end

# %% ==================== Monte carlo value comparison ====================

mc = deserialize("tmp/monte_carlo")
full = deserialize("tmp/abs_exp_full")(dispersion=1e10)
X = maximum(full, dims=(:β, :α)) |> dropdims(:β, :α)


function plot_grid(f; kws...)
    rn, cn = keys(kws)
    rows, cols = values(kws)
    @show rows cols
    ps = map(Iterators.product(rows, cols)) do (r, c)
        title!(f(r, c), "$rn=$r, $cn=$c")
    end
    nr, nc = map(length, (rows, cols))
    plot(ps..., size=300 .* (nr, nc), layout=(nc,nr), bottom_margin=4mm)
end

figure() do
    ylim = (0, maximum(X))
    p = plot_grid(;k=1:3, s=[1, 5, 25]) do k, s
        plot(X(;k, s); ylim, ylabel="Reward", label="AbsExp")
        hline!([mc(;k, s)], line=(:gray, :dot), label="Monte Carlo", legend=false)
    end
    plot!(p[1], legend=:topleft)
end



# %% ==================== Abs vs exp ====================

function plot_ks_grid(f, X)
    ps = map(Iterators.product(axiskeys(X, :k), axiskeys(X, :s))) do (k, s)
        title!(f(X(;k, s)), "k=$k, s=$s")
    end
    plot(ps..., size=(900,900), layout=(3,3), bottom_margin=4mm)
end

function ks_heat(X)
    clim = (0, maximum(X))
    ps = map(Iterators.product(axiskeys(X, :k), axiskeys(X, :s))) do (k, s)
        x = X(;k,s)
        clim = (0, maximum(x))
        heatmap(x'; title="k=$k, s=$s", clim, cbar=false)
    end
    plot(ps..., size=(900,900), layout=(3,3), bottom_margin=4mm)
end

X = deserialize("tmp/abs_exp_full")(dispersion=1e10)

figure("abs_exp_optimized") do
    plot_ks_grid(X) do x
        plot(maximum(x; dims=(:d, :α)) |> dropdims((:d, :α))
    end
end

figure("abs_exp_β") do
    X = deserialize("tmp/abs_exp_β")(dispersion=1e10, α=0, d=4)
    plot_ks_grid(X) do x
        plot(x)
    end
end


figure("abs_exp_αd") do
    plot_ks_grid(X) do x
        clim = (0, maximum(x))
        heatmap(maximum(x, dims=:β) |> dropdims(:β); clim, cbar=false)
    end
end

figure("abs_exp_βd") do
    ks_heat(X(α=0))
end

figure("abs_exp_βα") do
    ks_heat(X(d=4))
end

# %% --------
X = deserialize("tmp/abs_exp_β_many")(dispersion=1e10, α=0, d=4)

# Y = map(Iterators.product(axiskeys(X, :k), axiskeys(X, :s))) do (k, s)
figure("opt_β_grid") do
    ps = map(axiskeys(X, :s)) do s
        x = X(; s)
        clim = (0, maximum(x))
        heatmap(x; clim, cbar=false)
    end
    plot(ps..., size=(900,300), layout=(1,3), bottom_margin=4mm)
end


# figure("optimize_β") do
#     plot(Y)
# end


# %% --------

opt_βs = deserialize("tmp/opt_βs")(dispersion=1e10)
opt_βs[.≈(opt_βs, .381966; atol=.001)] .= NaN
figure("abs_exp_opt_β") do
    plot(opt_βs, ylabel="Optimal β")
end


