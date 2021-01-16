using Plots
using Plots.Measures
using Dates
mkpath("fighist")
mkpath("figs")
gr(label="", dpi=200, size=(400,300), lw=2)
ENV["GKSwstype"] = "nul"

function figure(f, name="tmp"; kws...)
    plot(;kws...)
    f()
    dt = Dates.format(now(), "m-d-H-M-S")
    p = "fighist/$dt-$name.png"
    savefig(p)
    if name != "tmp"
        cp(p, "figs/$name.png"; force=true)
    end
end

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