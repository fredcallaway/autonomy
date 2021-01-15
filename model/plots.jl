using Plots.Measures

include("figure.jl")
include("utils.jl")

function Plots.heatmap(X::KeyedArray{<:Real,2}; kws...)
    ylabel, xlabel = dimnames(X)
    heatmap(reverse(axiskeys(X))..., X; xlabel, ylabel, kws...)
end

function Plots.plot(x::KeyedArray{<:Real,1}; kws...)
    plot(axiskeys(x, 1), collect(x); xlabel=string(dimnames(x, 1)), kws...)
end

function Plots.plot(X::KeyedArray{<:Real,2}; kws...)
    plot(axiskeys(X, 1), collect(X);
        xlabel=dimnames(X, 1),
        label=reshape(["k=$k" for k in axiskeys(X, 2)], 1, :),
        palette=collect(cgrad(:viridis, size(X, 2), categorical = true)),
        kws...
    )
end

# %% ==================== Abs vs exp ====================

X = deserialize("tmp/abs_exp_det")(dispersion=1e10, α=0)
figure("abs_exp_det") do
    clim = (0, maximum(X))
    ps = map(Iterators.product(axiskeys(X, :k), axiskeys(X, :s))) do (k, s)
        x = X(;k,s)
        clim = (0, maximum(x))
        heatmap(x'; title="k=$k, s=$s", clim, cbar=false)
    end
    plot(ps..., size=(900,900), layout=(3,3), bottom_margin=4mm)
end
# %% --------
X = deserialize("tmp/abs_exp_detp")(dispersion=1e10, α=0)
figure("abs_exp_detp") do
    clim = (0, maximum(X))
    ps = map(Iterators.product(axiskeys(X, :k), axiskeys(X, :s))) do (k, s)
        x = X(;k,s)
        clim = (0, maximum(x))
        heatmap(x'; title="k=$k, s=$s", clim, cbar=false)
    end
    plot(ps..., size=(900,900), layout=(3,3), bottom_margin=4mm)
end

# %% --------

X = deserialize("tmp/abs_exp_importance_equalprob")(dispersion=1e10, importance=0)
figure("abs_exp_equalprob") do
    ps = map(Iterators.product(axiskeys(X, :k), axiskeys(X, :s))) do (k, s)
        x = X(;k,s)
        plot(x; title="k=$k, s=$s", xlabel="β", ylabel="Reward")
    end
    plot(ps..., size=(900,900), layout=(3,3), bottom_margin=4mm)
end

# %% --------
X = deserialize("tmp/abs_exp_importance_equalprob")(dispersion=1e10)
figure("abs_exp_importance_equalprob_grids") do
    clim = (0, maximum(X))
    ps = map(Iterators.product(axiskeys(X, :k), axiskeys(X, :s))) do (k, s)
        x = X(;k,s)
        clim = (0, maximum(x))
        heatmap(x'; title="k=$k, s=$s", clim, cbar=false, xlabel="β", ylabel="α")
    end
    plot(ps..., size=(900,900), layout=(3,3), bottom_margin=4mm)
end


# %% --------

X = deserialize("tmp/abs_exp_importance")
figure("abs_exp_importance") do
    clim = (0, maximum(X))
    ps = map(Iterators.product(axiskeys(X, :k), axiskeys(X, :s))) do (k, s)
        x = X(;k,s)
        clim = (0, maximum(x))
        heatmap(x'; title="k=$k, s=$s", clim, cbar=false)
    end
    plot(ps..., size=(900,900), layout=(3,3), bottom_margin=4mm)
end


# %% --------
X = deserialize("tmp/abs_exp_sk")

figure("abs_exp_sk") do
    clim = (0, maximum(X))
    ps = map(Iterators.product(axiskeys(X, :k), axiskeys(X, :s))) do (k, s)
        plot(X(;k,s)', title="k=$k, s=$s")
    end
    plot(ps..., size=(900,900), layout=(3,3), bottom_margin=4mm)
end

# %% --------
X = deserialize("tmp/abs_exp")
figure() do
    plot(axiskeys(X, 2), X', 
        label=reshape(["k=$k" for k in axiskeys(X, 1)], 1, :),
        palette=collect(cgrad(:viridis, size(X, 1), categorical = true)),
        lw=2, xlabel="Weight on Value", ylabel="Reward", legend=:topleft,
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