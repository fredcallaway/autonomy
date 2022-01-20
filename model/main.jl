@everywhere begin
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
end
Base.active_repl.options.iocontext[:displaysize] = (20, displaysize(stdout)[2])

# %% --------
raw_df = CSV.read("../data/processed-old.csv", DataFrame)
zscore(x) = (x .- mean(x)) ./ std(x)

wid2pid = Dict(zip(unique(raw_df.wid), 1:10000))

all_trials = @chain raw_df begin
    # groupby(:wid)
    # @transform(:evaluation = zscore(:evaluation))
    groupby([:wid, :scenario])
    map(collect(_)) do g
        pid = wid2pid[only(unique(g.wid))]
        control = only(unique(g.control))
        (;pid, control, g.evaluation, g.accessibility, g.considered)
    end
end

filter!(all_trials) do t
    sum(t.considered) > 0
end

split_trials = group(t->t.control, all_trials)

@everywhere all_trials  = $all_trials
@everywhere split_trials  = $split_trials

# %% --------
models = map(x->Model(x, ε=0.), (
    # accessibility=Model{NullWeighter}(),
    # softmax=Softmax(),
    # uws=UWS(),
    # switch=SwitchWeighter(UWS(), Softmax()),
    switch_softmax=SwitchWeighter(Softmax(), Softmax()),
    switch_uws=SwitchWeighter(UWS(), UWS()),
    # multiplicative=Model(SoftmaxUWS()),
    switch_joint=SwitchWeighter(ProductWeighter(Softmax(), UWS()), ProductWeighter(Softmax(), UWS())),
    # switch_multiplicative=SwitchingSoftmaxUWS(β_low=0.),
    switch_multiplicative_flex=SwitchingSoftmaxUWS(),
    mixture=MixtureWeighter(UWS(), Softmax())
    # switch_multiplicative2=SwitchingSoftmaxUWS2(),
    # switch_multiplicative_alt=SwitchingSoftmaxUWS(β_low = 0., C_exp=0.),
))

n_param = map(models) do model
    length(flatten(model, Missing))
end

@time mle = pmap(models) do model
    find_mle(model, all_trials; n_restart=30)
end

lp = map(mle) do model
    logp(model, all_trials)
end

aic = map(lp, n_param) do l, k
    2k - 2l
end

foreach(collect(pairs(aic))) do (k, v)
    print(rpad(k, 25), "  ")
    println(round(v; digits=1) / 2)
end

# %% --------
t = all_trials[6]
likelihood(mutate(mle.switch_softmax; β_acc=0), t) |> sort

score(mle.switch_softmax.weighter.high, t.evaluation)  |> sort

weighter = mle.switch_softmax.weighter.high

exp(weighter.β*9) + weighter.C
t.evaluation |> sort

# %% --------
@everywhere using Turing
@everywhere using Turing: @addlogprob!
@everywhere using ArviZ
@everywhere Turing.setprogress!(false)

@everywhere @model turing_model(base_model, trials, ::Type{T} = Float64) where T = begin
    priors = metaflatten(base_model, prior2, Missing)
    params = Vector{T}(undef, length(priors))
    for i = 1:length(priors)
        params[i] ~ priors[i]
    end
    model = reconstruct(base_model, params, Missing)
    map(trials) do trial
        lp = logp(model, trial)
        @addlogprob! lp
        lp
    end
end

turing_results = pmap(models) do base_model
    tm = turing_model(base_model, all_trials)
    chain = sample(tm, NUTS(), 5000);
    # chain = sample(tm, NUTS(), MCMCDistributed(), 2000, 4);
    L = generated_quantities(tm, MCMCChains.get_sections(chain, :parameters));
    L = permutedims(combinedims(L), (3, 2, 1))
    idata = from_mcmcchains(chain; log_likelihood=Dict("L" => L))
    (;chain, idata)
end

compare(Dict(pairs(invert(turing_results).idata)))

turing_results.switch_joint.chain
fieldnameflatten(models.switch_joint, Missing)

# %% --------
using Random
model = Model(NullWeighter(), β_acc=1.)

fake_trials = map(all_trials) do t
    idx = randperm(length(t.evaluation))
    (;t.control, evaluation=t.evaluation[idx], accessibility=t.accessibility[idx])
end

trials = split_trials["low"]

map(group(t->t.evaluation[1], trials) |> pairs |> sort |> collect) do (v, trials)
    length(trials)
end

# %% --------
model = mle.mixture

fake_trials = map(all_trials) do t
    idx = randperm(length(t.evaluation))
    (;t.control, evaluation=t.evaluation[idx], accessibility=t.accessibility[idx])
end

preds = map(group(t->t.control, fake_trials)) do trials
    p = map(-10:1:10) do v
        total = mapreduce(+, trials) do t
            t.evaluation[1] = v
            likelihood(model, t)[1]
        end
        total / length(trials)
    end
end

figure(ylim=(0, 0.3)) do
    plot!(-10:1:10, preds["low"], label="low")
    plot!(-10:1:10, preds["high"], label="high")
end

# compare(Dict(pairs(invert(turing_results_bad).idata)))

# %% --------
figure() do
    plot(Beta(1.1, 2))
end

mode(Beta(2, 9))

cdf(Beta(2, 9), .001)

# %% --------
turing_results.switch_softmax.chain
fieldnameflatten(models.switch_softmax, Missing)


turing_results.switch_multiplicative.chain
fieldnameflatten(models.switch_multiplicative, Missing)

# %% --------
@everywhere using Turing
@everywhere import Turing: @addlogprob!


priors = Dict(:β_low => Exponential(2))

@model main(trials) = begin
    base_model = Model(SwitchingSoftmaxUWS())

    β_low ~ priors[:β_low]
    β_high ~ Exponential(2)
    C_abs ~ truncated(Cauchy(0,10), 0, Inf)
    C_exp ~ truncated(Cauchy(0,1000), 0, Inf)
    β_acc ~ Exponential(2)
    model = reconstruct(base_model, [β_low, β_high, C_abs, C_exp, β_acc], Missing)
    map(trials) do trial
        lp = logp(model, trial)
        @addlogprob! lp
        lp
    end
end

model = main(all_trials);
chain = sample(model, NUTS(), MCMCDistributed(), 5000, 4)

L = generated_quantities(model, MCMCChains.get_sections(chain, :parameters));
L = permutedims(combinedims(L), (3, 2, 1));

loo(from_mcmcchains(chain; log_likelihood=Dict("obs" => L)))

# mean(sum(L; dims=3))
# mle = find_mle(Model(SwitchingSoftmaxUWS()), all_trials)
# logp(mle, all_trials)

# %% --------
using ArviZ
idata = from_mcmcchains(chain; log_likelihood=Dict("obs" => L))
loo(idata)


chain[:β_low]
plot_posterior(chain[:β_high][:])

# %% --------
@everywhere using Turing
import Turing: @addlogprob!

data = map(group(x->x.pid, all_trials)) do trials
    (;trials[1].pid, trials[1].control, trials)
end |> collect

@model main(data, ::Type{T} = Float64) where {T} = begin
    μ_β_low ~ Normal(0, 10)
    μ_β_high ~ Normal(0, 10)

    σ_β_low ~ Exponential(10)
    σ_β_high ~ Exponential(10)
    σ_β_acc ~ Exponential(10)

    dist_β_low = Normal(μ_β_low, σ_β_low)
    dist_β_high = Normal(μ_β_high, σ_β_high)

    λ_β_acc ~ Exponential(10)
    λ_C_abs ~ Exponential(10)
    λ_C_exp ~ Exponential(1000)

    # C_abs ~ truncated(Cauchy(0,10), 0, Inf)
    # C_exp ~ truncated(Cauchy(0,1000), 0, Inf)
    # β_acc ~ Exponential(2)

    β = Vector{T}(undef, length(data))
    
    C_abs ~ filldist(Exponential(λ_C_abs), length(data))
    C_exp ~ filldist(Exponential(λ_C_exp), length(data))
    β_acc ~ filldist(Exponential(λ_β_acc), length(data))

    map(data) do dd
        i = dd.pid
        β[i] ~ dd.control == "high" ? dist_β_high : dist_β_low
        model = Model(ProductWeighter(UWS(C=C_abs[i]), Softmax(;β=β[i], C=C_exp[i])); β_acc=β_acc[i])
        lp = logp(model, dd.trials)
        @addlogprob! lp
        lp
    end
end

chain = sample(main(data), NUTS(), MCMCDistributed(), 5000, 4)

# %% --------
mean(chain[:μ_β_high])
mean(chain[:μ_β_low])



L = generated_quantities(main(data), MCMCChains.get_sections(chain, :parameters))
L = permutedims(combinedims(L), (3, 2, 1))

mean(sum(L; dims=3))
mle = find_mle(Model(SwitchingSoftmaxUWS()), all_trials)
logp(mle, all_trials)

idata = from_mcmcchains(chain; log_likelihood=Dict("obs" => L))
loo(idata)
# %% --------


# %% --------

invert(get(chain; section=:parameters))

lpfun = function f(chain::Chains) # function to compute the logpdf values
    niter, nparams, nchains = size(chain)


    lp = zeros(niter + nchains) # resulting logpdf values
    for i = 1:nparams
        lp += map(p -> logpdf( ... , x), Array(chain[:,i,:]))
    end
    return lp
end


Turing.dic(chain)

# %% --------

switch2 = find_mle(Model(SwitchWeighter(UWS(), Softmax())), all_trials); logp(switch2, all_trials)
switch2 = find_mle(Model(SwitchWeighter(UWS(), Softmax())), all_trials); logp(switch2, all_trials)


soft = find_mle(Model{UWS}(), split_trials["high"]); logp(uws, split_trials["high"])


uws = find_mle(Model{UWS}(), split_trials["high"]); logp(uws, split_trials["high"])

uws = find_mle(Model{Softmax}(), all_trials); logp(uws, all_trials)
# %% --------
logp(switch, all_trials)
logp(switch2, all_trials)  # exponentiated performs better

logp(soft, all_trials)
logp(uws2, all_trials)  # exponentiated performs better
logp(uws, all_trials)  # exponentiated performs better
logp(switch, all_trials)  # exponentiated performs better

logp(update(switch, β_abs=1.), all_trials)

logp(soft, all_trials)
logp(uws, all_trials)
logp(uws2, all_trials)
# m4 = find_mle(Model{SoftmaxUWS}(), all_trials)


logp(m1, all_trials)
logp(m2, all_trials)
logp(m3, all_trials)
logp(m4, all_trials)

# %% --------
figure() do
    x = -3:.05:3
    plot(x, abs.(x)  .+ exp.(.05*x))
    plot!(x, abs.(x) .+ exp.(.1*x))
    plot!(x, abs.(x) .+ exp.(.2*x))
    plot!(x, abs.(x) .+ exp.(.4*x))
    plot!(x, abs.(x) .+ exp.(1*x))
end

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
