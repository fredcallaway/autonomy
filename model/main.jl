using CSV, DataFrames, DataFramesMeta
using Optim
using StatsFuns
using SplitApplyCombine

include("utils.jl")
include("simple_model.jl")
include("figure.jl")
include("box.jl")

Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2])

# %% --------

df = CSV.read("../data/processed.csv", DataFrame)

all_trials = map(collect(groupby(df, [:wid, :scenario]))) do g
    control = only(unique(g.control))
    (;control, g.evaluation, g.accessibility, g.considered)
end

split_trials = group(t->t.control, all_trials)

# %% --------
trials = split_trials["low"]
model = find_mle(Model{UWS}, trials)
logp(model, trials)

model2 = find_mle(Model{AbsExp}, trials)
logp(model2, trials)

baseline = find_mle(Model{NullWeighter}, trials)
logp(baseline, trials)

# %% --------



# %% --------
models = (
    baseline=Model{NullWeighter},
    softmax=Model{Softmax},
    uws=Model{UWS},
    softmax_uws=Model{SoftmaxUWS},
    absexp=Model{AbsExp},
)
results = map(split_trials) do trials
    mle = map(models) do M
        find_mle(M, trials; n_restart=200)
    end

    lp = map(mle) do model
        logp(model, trials)
    end

    (;mle, lp)
end

# %% --------


results["low"].lp
results["high"].lp

results["high"].mle.softmax_uws
results["low"].mle.softmax_uws

results["low"].mle.absexp

# %% --------
m = results["high"].mle.softmax_uws
trials = split_trials["high"]
logp(m, trials)
β, β_abs, ev, ε, β_acc = flatten(m)
β = 0.
logp(reconstruct(m, (β, β_abs, ev, ε, β_acc)), trials)

results["high"].lp.uws


# %% --------
trials = split_trials["low"]
model = find_mle(Model{AbsExp}, trials)
logp(model, trials)


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
