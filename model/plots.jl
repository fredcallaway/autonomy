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
    k = dimnames(X, 2)
    plot(axiskeys(X, 1), collect(X);
        xlabel=dimnames(X, 1),
        label=reshape(["$k=$v" for v in axiskeys(X, 2)], 1, :),
        palette=collect(cgrad(:viridis, size(X, 2), categorical = true)),
        kws...
    )
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
        plot(maximum(x; dims=(:d, :α)) |> dropdims((:d, :α)))
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


