using Revise
using CSV
using Optim
using StatsFuns
using TypedTables
using SplitApplyCombine
using Query

includet("utils.jl")
includet("model.jl")
includet("figure.jl")

Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2])

# %% --------
data = CSV.read("../data/bear2020_exp2.csv", Table) |> @mutate(
    bimodal = Bool(_.bimodal),
    passComp = Bool(_.passComp),
    ideal = [:low, :high, :middle][_.ideal+1]
) |> @filter(_.passComp) |> Table

data |> @groupby((_.ideal, _.bimodal))
G = group(x->(x.ideal, x.bimodal), data) .|> Table

@code_warntype group(mapview(x->(x.ideal, x.bimodal), data), data)

# %% --------

ideals = let
    X = [zeros(20); 1:60; 60*ones(20)];
    Dict(
        :low => maximum(X) .- [0; X],
        :middle => 50 .- abs.(50 .- (0:100)),
        :high => [0; X]
    )
end
pdfs = Dict(
    false => normalize(pdf.(Normal(45, 15), 0:100)),
    true => normalize(0.5pdf.(Normal(35, 5), 0:100) .+ 0.5pdf(Normal(75, 5), 0:100))
)
get_problem(ideal, bimodal) = (ideals[ideal], pdfs[bimodal])
get_problem(datum) = get_problem(datum.ideal, datum.bimodal)

# %% --------

function logp(W::Weighter, datum::NamedTuple)
    u, p = get_problem(datum)
    log(weight(W, u, p)[datum.sample+1])
end

function logp(W::Weighter, data::AbstractVector{<:NamedTuple})
    mapreduce(+, data) do datum
        logp(W, datum)
    end
end

function find_mle(make_model::Function, data; x0=[-20, -20.])
    res = optimize(x0, BFGS(), autodiff=:forward) do x
        -logp(make_model(x), data)
    end
    make_model(res.minimizer)
end

mle_bear = map(G) do g
    find_mle(g; x0=invsoftplus.([1/15, .1])) do x
        β_u, C = softplus.(x)
        Softmax(;β_u, C)
    end
end

mle_ana = map(pairs(G)) do ((ideal, bimodal), g)
    μ, σ = DiscreteNonParametric(0:100, pdfs[bimodal]) |> juxt(mean, std)
    find_mle(g; x0=[2., -20]) do (k, C)
        Analytic(k, μ, σ, softplus(C))
    end
end

mle_absexp = map(G) do g
    find_mle(g; x0=[1., invsoftplus(1/15), invsoftplus(.1)]) do (β, d, C)
        β = logistic(β)
        d = softplus(d)
        C = softplus(C)
        AbsExp(;β, d, C)
    end
end

# %% --------

nll_bear = map(mle_bear, G) do model, g
    -logp(model, g)
end

nll_ana = map(mle_ana, G) do model, g
    -logp(model, g)
end

nll_absexp = map(mle_absexp, G) do model, g
    -logp(model, g)
end

display(nll_ana .- nll_bear)
display(nll_absexp .- nll_bear)

sum(nll_ana) - sum(nll_bear)
sum(nll_absexp) - sum(nll_bear)
# %% --------
conds = [
    (:low, false),
    (:low, true),
    (:high, false),
    (:high, true),
    (:middle, false),
    (:middle, true),
]
using Printf
println("  τ       C     -> NLL")
foreach(conds) do c
    @unpack β, C = mle_bear[c]
    @printf "%7.2f %7.2f -> %4.3f\n" 1/β C nll_bear[c]
end  

# %% --------

figure("data_and_model") do
    plot_grid((size=(300,200), no_title=true), ideal=[:low, :middle, :high], bimodal=[false, true]) do ideal, bimodal
        plt = histogram(G[(ideal, bimodal)].sample, normalize=:pdf, color=:lightgray)
        u, p = get_problem(ideal, bimodal)
        plot!(weight(mle_ana[(ideal, bimodal)], u, p), label="Analytic", color=1, alpha=0.7)
        plot!(weight(mle_bear[(ideal, bimodal)], u, p), label="Bear (2020)", color=2, alpha=0.7)
        plot!(weight(mle_absexp[(ideal, bimodal)], u, p), label="AbsExp", color=3, alpha=0.7)
        title!((bimodal ? "Bimodal" : "Unimodal") * " $ideal ideal")
        if (ideal, bimodal) ≠ (:low, false)
            plot!(legend=false)
        end
        plt
    end
end

# %% --------

figure("likelihood_grids") do
    plot_grid((size=(500,400), no_title=true), ideal=[:low, :middle, :high], bimodal=[false, true]) do ideal, bimodal
        L = map(grid(k=1:.1:8, C=.0001:.0001:.005)) do (β, C)
            μ, σ = DiscreteNonParametric(0:100, pdfs[bimodal]) |> juxt(mean, std)
            logp(AnalyticC(β, μ, σ, C), G[(ideal, bimodal)])
        end
        heatmap(L)
        title!((bimodal ? "Bimodal" : "Unimodal") * " $ideal ideal")
    end
end
