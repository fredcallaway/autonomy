using CSV, DataFrames, DataFramesMeta
using Optim
using StatsFuns
using SplitApplyCombine

include("utils.jl")
include("model.jl")
include("figure.jl")
include("box.jl")

Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2])

# %% --------

df = CSV.read("../data/processed.csv", DataFrame)

all_trials = map(collect(groupby(df, [:wid, :scenario]))) do g
    control = only(unique(g.control))
    (;control, g.evaluation, g.considered)
end

trials = group(t->t.control, all_trials)

# %% --------

function logp(W::Weighter, trial::NamedTuple)
    p = weight(W, trial.evaluation)
    sum(log.(p[trial.considered]))
end

function logp(W::Weighter, trials::AbstractVector{<:NamedTuple})
    mapreduce(+, trials) do trial
        logp(W, trial)
    end
end

# %% --------

function find_mle(make_model::Function, trials, box)
    x0 = box.lower .+ rand(length(box.lower)) .* box.upper .- box.lower
    unsquash!(box, x0)
    res = optimize(x0) do x
        squash!(box, x)
        -logp(make_model(x), trials)
    end
    make_model(squash!(box, res.minimizer))
end

# %% --------

mle = find_mle(trials, Box([0, 0], [10, 10])) do (β, C)
    SoftmaxUWS(;β, C)
end

logp(mle, trials)

logp(NullWeighter(), trials)

absexp = find_mle(trials, Box([0, 0, 0], [1, 1, 20])) do (β, d, C)
    AbsExp(;β, d, C)
end

logp(absexp, trials)
logp(AbsExp(;β=.5, d=0, C=0), trials)
logp(NullWeighter(), trials)

# %% --------

soft = find_mle(trials, Box([0], [1])) do (β_u,)
    Softmax(;β_u, C=1)
end
# %% --------
res = optimize(0, 1) do β_u
    -logp(Softmax(;β_u, C=1), trials)
end

soft = Softmax(;β_u=res.minimizer, C=1)

logp(soft, trials)
logp(NullWeighter(), trials)
logp(Softmax(;β_u=0, C=1), trials)


# %% --------

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
