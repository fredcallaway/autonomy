
function Plots.heatmap(X::KeyedArray{<:Real,2}; kws...)
    ylabel, xlabel = dimnames(X)
    heatmap(reverse(axiskeys(X))..., X; xlabel, ylabel, kws...)
end
using Plots.Measures


# %% --------
X = deserialize("tmp/abs_exp")
figure() do
    plot(axiskeys(X, 2), X', 
        label=reshape(["k=$k" for k in axiskeys(X, 1)], 1, :),
        # palette=collect(cgrad(:blues, 6, rev=true, categorical=true)),
        xlabel="β_u", ylabel="Reward", legend=:topleft, 
    )
end


# %% --------
X = deserialize("tmp/abs_exp_full")

figure("abs_exp_grids") do
    clim = (0, maximum(X))
    ps = map(axiskeys(X, :k)) do k
        heatmap(X(k=k); clim, cbar=false, title="k = $k")
    end
    plot(ps..., size=(900,300), layout=(1,3), bottom_margin=4mm)
end


# %% --------
X = deserialize("tmp/full_grid")
figure("full_grid") do
    ps = map(axiskeys(X, :k)) do k
        heatmap(X(k=k), clim=(0, .05), cbar=false, title="k = $k")
    end
    plot(ps..., size=(900,300), layout=(1,3), bottom_margin=4mm)
end


# %% --------
X = deserialize("tmp/k-β_u-0")
figure("k-β_u") do
    plot(axiskeys(X, 2), X', 
        label=reshape(["k=$k" for k in axiskeys(X, 1)], 1, :),
        palette=collect(cgrad(:blues, 6, rev=true, categorical=true)),
        xlabel="β_u", ylabel="Reward", legend=:topleft, 
    )
end

# %% --------
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