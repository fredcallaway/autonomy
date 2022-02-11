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
raw_df = CSV.read("../data/processed.csv", DataFrame)

# Each datum corresponds to one considered outcome
all_data = @chain raw_df begin
    # groupby(:wid)
    # @transform(:evaluation = zscore(:evaluation))
    groupby(:trial_id)
    mapreduce(vcat, collect(_)) do g
        control = only(unique(g.control))
        trial_id = only(unique(g.trial_id))
        evaluation = collect(g.evaluation)
        accessibility = collect(g.accessibility)
        map(1:sum(g.considered)) do order
            trial = (; trial_id, order, control, evaluation=copy(evaluation), accessibility=copy(accessibility), considered=1)
            popfirst!(evaluation); popfirst!(accessibility)
            trial
        end
    end
end

# %% ==================== Maximum Likelihood Estimation ====================


models = map(x->Model(x, ε=0.), (
    switching_softmax_uws=SwitchingSoftmaxUWS(),
    accessibility=NullWeighter(),
    # switch_softmax=SwitchWeighter(Softmax(), Softmax()),
    # switch_uws=SwitchWeighter(UWS(), UWS()),
    # softmax_uws=SoftmaxUWS(),
))

n_param = map(models) do model
    length(flatten(model, Missing))
end

@time mle = pmap(models) do model
    find_mle(model, all_data; n_restart=30)
end

lp = map(mle) do model
    logp(model, all_data)
end

aic = map(lp, n_param) do l, k
    2k - 2l
end

foreach(collect(pairs(aic))) do (k, v)
    print(rpad(k, 25), "  ")
    println(round(v; digits=1) / 2)
end

# %% --------
logp(mle.switching_softmax_uws, all_data)

pred_models = (;mle..., stronger=update(mle.switching_softmax_uws, β_low=0., C_abs=3))
map(all_data) do d
    x = map(pred_models) do model
        logp(model, d)
    end
    (;d.trial_id, d.order, x..., chance=log(1/length(d.evaluation)), eval_check=d.evaluation[1])
end |> CSV.write("results/mle_likelihoods.csv")

# %% --------
@chain raw_df begin
    # groupby(:wid)
    # @transform(:evaluation = zscore(:evaluation))
    groupby(:trial_id)
    mapreduce(vcat, collect(_)) do g
        control = only(unique(g.control))
        trial_id = only(unique(g.trial_id))
        evaluation = collect(g.evaluation)
        accessibility = collect(g.accessibility)
        map(1:sum(g.considered)) do i
            trial = (; trial_id, control, evaluation=copy(evaluation), accessibility=copy(accessibility), considered=1)
            popfirst!(evaluation); popfirst!(accessibility)
            trial
        end
    end
end



# %% ==================== Bayesian posterior  ====================

@everywhere using Turing
@everywhere using Turing: @addlogprob!
@everywhere using ArviZ
@everywhere Turing.setprogress!(true)

@everywhere @model turing_model(base_model, data, trial_wise=false, ::Type{T} = Float64) where T = begin
    priors = metaflatten(base_model, prior2, Missing)
    params = Vector{T}(undef, length(priors))
    for i = 1:length(priors)
        params[i] ~ priors[i]
    end
    model = reconstruct(base_model, params, Missing)
    if trial_wise
        map(data) do trial
            lp = logp(model, trial)
            @addlogprob! lp
            lp
        end
    else
        @addlogprob! logp(model, data)
    end
end

turing_results = pmap(models) do base_model
    # chain = sample(turing_model(base_model, all_data), NUTS(), 5000);
    chain = sample(tm, NUTS(), MCMCDistributed(), 5000, 4);
    L = generated_quantities(turing_model(base_model, all_data, true), 
                             MCMCChains.get_sections(chain, :parameters));
    L = permutedims(combinedims(L), (3, 2, 1))
    idata = from_mcmcchains(chain; log_likelihood=Dict("L" => L))
    (;chain, idata)
end
mle.softmax_uws
mle.switching_softmax_uws
aic
comp = compare(Dict(pairs(invert(turing_results).idata)))

comp |> CSV.write("results/comparison.csv")

# %% --------

ch = turing_results.switching_softmax_uws.chain



β_low = ch["params[1]"]
β_high = ch["params[2]"]

df = DataFrame(ch)[:, Cols(r"params")]
rename!(df, collect(fieldnameflatten(models.switching_softmax_uws, Missing)))
df |> CSV.write("results/chain.csv")




(df, )



std(β_low)
std(β_high)
mean(β_high .> β_low)


# %% ==================== Posterior predictive checks ====================

using Random
group = SplitApplyCombine.group

function eval_counts(models, data)
    X = Dict(
        "low" => zeros(Int, 21),
        "high" => zeros(Int, 21)
    )
    for model in models
        for t in data
            considered = sample(Weights(likelihood(model, t)))
            evaluation = t.evaluation[considered]
            X[t.control][evaluation+11] += 1
        end
    end
    X
end

model = :switching_softmax_uws
P = collect(cat(get(getfield(turing_results, model).chain; section=:parameters).params...; dims=3))
model_samples = map(eachrow(reshape(P, 5000, 5))) do x
    reconstruct(base_model, x, Missing)
end

cc = eval_counts(model_samples, all_data)

preds = DataFrame(cc)
preds.evaluation = -10:1:10

preds |> CSV.write("results/ppc.csv")



# %% --------
evaluation_vectors = invert(all_data).evaluation

function value_curves(model)
    # model = mle.mixture
    fake_data = map(all_data) do t
        (;t.control, evaluation=shuffle(t.evaluation), accessibility=ones(length(t.evaluation)))
    end
    map(group(t->t.control, fake_data)) do data
        p = map(-10:1:10) do v
            total = mapreduce(+, data) do t
                t.evaluation[1] = v
                likelihood(model, t)[1]
            end
            total / length(data)
        end
    end
end


model = :switching_softmax_uws
P = collect(cat(get(getfield(turing_results, model).chain; section=:parameters).params...; dims=3))

modl = model_samples[1]
simulate(modl, all_data)

base_model = getfield(models, model)
model_samples = map(eachrow(reshape(P, 5000, 5))) do x
    reconstruct(base_model, x, Missing)
end

value_curves(mod)


mean_model = reconstruct(models.switching_softmax_uws, collect(DataFrame(mean(ch)).mean), Missing)

figure() do
    plot!(-10:1:10, value_curves(mle.softmax_uws)["low"], label="shared β", color="#aaaaaa")
    preds = value_curves(mean_model)
    plot!(-10:1:10, preds["low"], label="low", color="#31ADF4")
    plot!(-10:1:10, preds["high"], label="high", color="#F5CE47")
end


# %% --------
model = mle.switch_softmax
figure() do
    preds = value_curves(model)
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

@model main(data) = begin
    base_model = Model(SwitchingSoftmaxUWS())

    β_low ~ priors[:β_low]
    β_high ~ Exponential(2)
    C_abs ~ truncated(Cauchy(0,10), 0, Inf)
    C_exp ~ truncated(Cauchy(0,1000), 0, Inf)
    β_acc ~ Exponential(2)
    model = reconstruct(base_model, [β_low, β_high, C_abs, C_exp, β_acc], Missing)
    map(data) do trial
        lp = logp(model, trial)
        @addlogprob! lp
        lp
    end
end

model = main(all_data);
chain = sample(model, NUTS(), MCMCDistributed(), 5000, 4)

L = generated_quantities(model, MCMCChains.get_sections(chain, :parameters));
L = permutedims(combinedims(L), (3, 2, 1));

loo(from_mcmcchains(chain; log_likelihood=Dict("obs" => L)))

# mean(sum(L; dims=3))
# mle = find_mle(Model(SwitchingSoftmaxUWS()), all_data)
# logp(mle, all_data)

# %% --------
using ArviZ
idata = from_mcmcchains(chain; log_likelihood=Dict("obs" => L))
loo(idata)


chain[:β_low]
plot_posterior(chain[:β_high][:])

# %% --------
@everywhere using Turing
import Turing: @addlogprob!

data = map(group(x->x.pid, all_data)) do data
    (;data[1].pid, data[1].control, data)
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
        lp = logp(model, dd.data)
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
mle = find_mle(Model(SwitchingSoftmaxUWS()), all_data)
logp(mle, all_data)

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

split_data = group(t->t.control, all_data)

@everywhere all_data  = $all_data
@everywhere split_data  = $split_data

switch2 = find_mle(Model(SwitchWeighter(UWS(), Softmax())), all_data); logp(switch2, all_data)
switch2 = find_mle(Model(SwitchWeighter(UWS(), Softmax())), all_data); logp(switch2, all_data)


soft = find_mle(Model{UWS}(), split_data["high"]); logp(uws, split_data["high"])


uws = find_mle(Model{UWS}(), split_data["high"]); logp(uws, split_data["high"])

uws = find_mle(Model{Softmax}(), all_data); logp(uws, all_data)
# %% --------
logp(switch, all_data)
logp(switch2, all_data)  # exponentiated performs better

logp(soft, all_data)
logp(uws2, all_data)  # exponentiated performs better
logp(uws, all_data)  # exponentiated performs better
logp(switch, all_data)  # exponentiated performs better

logp(update(switch, β_abs=1.), all_data)

logp(soft, all_data)
logp(uws, all_data)
logp(uws2, all_data)
# m4 = find_mle(Model{SoftmaxUWS}(), all_data)


logp(m1, all_data)
logp(m2, all_data)
logp(m3, all_data)
logp(m4, all_data)

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
data = split_data["low"]
model = find_mle(Model{UWS}(), data)
logp(model, data)

model = find_mle(Model{SoftmaxUWS}(), data)
logp(model, data)

model2 = find_mle(Model{AbsExp}, data)
logp(model2, data)

baseline = find_mle(Model{NullWeighter}(), data)
logp(baseline, data)


# %% --------


# %% --------
data = split_data["high"]
mle = find_mle(Model{SoftmaxUWS}(), data; n_restart=10)
logp(mle, data)
logp(baseline, data)

fieldnameflatten(mle)
x0 = collect(flatten(mle))

# %% --------

G = grid(β=0:.2:5, β_abs=0:.2:5)
X = map(G) do kws
    logp(update(model; kws...), data)
end

figure() do
    heatmap(X)
end

find_mle(Model{SoftmaxUWS}(), split_data["low"]; n_restart=10)

# %% --------
models = (
    accessibility=Model{NullWeighter}(),
    softmax=Model{Softmax}(),
    uws=Model{UWS}(),
    softmax_uws=Model{SoftmaxUWS}(),
    absexp=Model{AbsExp}(),
)
results = map(split_data) do data
    mle = map(models) do model
        find_mle(model, data; n_restart=100)
    end

    lp = map(mle) do model
        logp(model, data)
    end

    (;mle, lp)
end

# %% --------

chance_logp(data) = logp(Model{NullWeighter}(β_acc=0, ε=1), data)

map(["low", "high"]) do k
    (control=k, chance=chance_logp(split_data[k]), results[k].lp...)
end |> DataFrame

results["high"].mle.softmax_uws
results["low"].mle.softmax_uws

results["high"].mle.absexp
results["low"].mle.absexp

results["low"].mle.absexp



# %% ==================== Recovery ====================

true_model = Model{SoftmaxUWS}(β=6., β_abs=4., ε=0., β_acc=0.3)

fake_data = map(split_data["high"]) do t
    p = likelihood(true_model, t)
    n = length(p)
    # chosen = sample(1:n, Weights(p), sum(t.considered), replace=false)
    chosen = sample(1:n, Weights(p), 1, replace=false)
    considered = zeros(Bool, n)
    considered[chosen] .= 1
    (;t..., considered)
end

@show logp(true_model, fake_data)

mle = find_mle(Model{SoftmaxUWS}(ε=0.), fake_data)
@show logp(mle, fake_data)
@show mle
nothing



# %% ==================== OLD ====================




m = results["high"].mle.softmax_uws
data = split_data["high"]
logp(m, data)
β, β_abs, ev, ε, β_acc = flatten(m)
β = 0.
logp(reconstruct(m, (β, β_abs, ev, ε, β_acc)), data)

results["high"].lp.uws


map(models) do model
    length(flatten(model, Missing))
end

# %% --------
data = split_data["low"]
model = find_mle(Model{AbsExp}, data)
logp(model, data)


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
