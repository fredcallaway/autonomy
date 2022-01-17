using CSV, DataFrames, DataFramesMeta
using Optim
using StatsFuns
using SplitApplyCombine

using DataFramesMeta: @chain
using Statistics, StatsBase

include("utils.jl")
include("simple_model.jl")
include("figure.jl")
include("box.jl")

Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2])

# %% --------

raw_df = CSV.read("../data/processed.csv", DataFrame)
zscore(x) = (x .- mean(x)) ./ std(x)

all_trials = @chain raw_df begin
    groupby(:wid)
    @transform(:evaluation = zscore(:evaluation))
    groupby([:wid, :scenario])
    map(collect(_)) do g
        control = only(unique(g.control))
        (;control, g.evaluation, g.accessibility, g.considered)
    end
end

all_trials = map(collect(groupby(df, [:wid, :scenario]))) do g
    control = only(unique(g.control))
    (;control, g.evaluation, g.accessibility, g.considered)
end

split_trials = group(t->t.control, all_trials)

# %% --------
trials = split_trials["low"]
model = find_mle(Model{UWS}(), trials)
logp(model, trials)

model = find_mle(Model{SoftmaxUWS}(), trials)
logp(model, trials)

model2 = find_mle(Model{AbsExp}, trials)
logp(model2, trials)

baseline = find_mle(Model{NullWeighter}(), trials)
logp(baseline, trials)


# %% --------



# %% --------
trials = split_trials["high"]
mle = find_mle(Model{SoftmaxUWS}(), trials; n_restart=10)
logp(mle, trials)
logp(baseline, trials)

fieldnameflatten(mle)
x0 = collect(flatten(mle))

# %% --------

G = grid(β=0:.2:5, β_abs=0:.2:5)
X = map(G) do kws
    logp(update(model; kws...), trials)
end

figure() do
    heatmap(X)
end

find_mle(Model{SoftmaxUWS}(), split_trials["low"]; n_restart=10)

# %% --------
models = (
    accessibility=Model{NullWeighter}(),
    softmax=Model{Softmax}(),
    uws=Model{UWS}(),
    softmax_uws=Model{SoftmaxUWS}(),
    absexp=Model{AbsExp}(),
)
results = map(split_trials) do trials
    mle = map(models) do model
        find_mle(model, trials; n_restart=100)
    end

    lp = map(mle) do model
        logp(model, trials)
    end

    (;mle, lp)
end

# %% --------

chance_logp(trials) = logp(Model{NullWeighter}(β_acc=0, ε=1), trials)

map(["low", "high"]) do k
    (control=k, chance=chance_logp(split_trials[k]), results[k].lp...)
end |> DataFrame

results["high"].mle.softmax_uws
results["low"].mle.softmax_uws

results["high"].mle.absexp
results["low"].mle.absexp

results["low"].mle.absexp



# %% ==================== Recovery ====================

true_model = Model{SoftmaxUWS}(β=6., β_abs=4., ε=0., β_acc=0.3)

fake_trials = map(split_trials["high"]) do t
    p = likelihood(true_model, t)
    n = length(p)
    # chosen = sample(1:n, Weights(p), sum(t.considered), replace=false)
    chosen = sample(1:n, Weights(p), 1, replace=false)
    considered = zeros(Bool, n)
    considered[chosen] .= 1
    (;t..., considered)
end

@show logp(true_model, fake_trials)

mle = find_mle(Model{SoftmaxUWS}(ε=0.), fake_trials)
@show logp(mle, fake_trials)
@show mle
nothing



# %% ==================== OLD ====================




m = results["high"].mle.softmax_uws
trials = split_trials["high"]
logp(m, trials)
β, β_abs, ev, ε, β_acc = flatten(m)
β = 0.
logp(reconstruct(m, (β, β_abs, ev, ε, β_acc)), trials)

results["high"].lp.uws


map(models) do model
    length(flatten(model, Missing))
end

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
