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
    switch_multiplicative=SwitchingSoftmaxUWS(),
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
    println(round(v; digits=1))
end
