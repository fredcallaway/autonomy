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

# %% --------
from_vec(T::Type{<:Weighter}, x) = T(x...)
from_vec(T::Type{Bear2020}, x) = T(softplus.(x)...)
# to_vec(W::Bear2020) = invsoftplus.([W.β, W.C])

function find_mle(T::Type{<:Weighter}, data; x0=[-20, -20.])
    res = optimize(x0, BFGS()) do x
        -logp(from_vec(T, x), data)
    end
    from_vec(T, res.minimizer)
end

function find_mle(make_model::Function, data; x0=[-20, -20.])
    res = optimize(x0, BFGS()) do x
        -logp(make_model(x), data)
    end
    make_model(res.minimizer)
end

mle_bear = map(G) do g
    find_mle(Bear2020, g; x0=invsoftplus.([1/15, .1]))
end

nll_bear = map(mle_bear, G) do model, g
    -logp(model, g)
end

mle_ana = map(pairs(G)) do ((ideal, bimodal), g)
    μ, σ = DiscreteNonParametric(0:100, pdfs[bimodal]) |> juxt(mean, std)
    find_mle(g; x0=[2., -20]) do (k, C)
        AnalyticC(k, μ, σ, softplus(C))
    end
end

nll_ana = map(mle_ana, G) do model, g
    -logp(model, g)
end

round.([sum(nll_ana), sum(nll_bear)])

nll_ana .- nll_bear
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
        plot!(weight(mle_ana[(ideal, bimodal)], u, p), label="Analytic", color=1)
        plot!(weight(mle_bear[(ideal, bimodal)], u, p), label="Bear (2020)", color=2)
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

# %% --------
u, p = get_problem(:high, false)
figure() do
    μ, σ = DiscreteNonParametric(0:100, pdfs[bimodal]) |> juxt(mean, std)
    plot(score(Analytic(2, μ, σ), u, p)./ p) 
end



# %% ==================== Scratch ====================


figure() do
    row = data[1, :]
    u, p = get_problem(row)
    # plot(weight(Bear2020(.1, 0.1), u, p))
    # plot!(weight(Bear2020(.1, 0.2), u, p))
    plot!(weight(mle_bear, u, p))
end

# %% --------
figure() do
    βs = 0:.001:1
    y = map(βs) do x
        logp(Bear2020(x, 1e-11), first(G))
    end
    plot(βs, y)
end
# %% --------
res = optimize([-20., -20], BFGS(), Optim.Options(store_trace=true, extended_trace=true)) do x
    -logp(from_vec(Bear2020, x), first(G))
end
trc = invert(Optim.x_trace(res))

L = map(grid(β=0:.001:.08, C=.0001:.0001:.003)) do (β, C)
    logp(Bear2020(β, C), first(G))
end

# %% --------
res = optimize([0., 0], BFGS(), Optim.Options(store_trace=true, extended_trace=true)) do x
    -logp(from_vec(Bear2020, x), first(G))
end
trc = invert(Optim.x_trace(res))
L = map(grid(β=0:.01:1, C=0:.01:1)) do (β, C)
    logp(Bear2020(β, C), first(G))
end
figure() do
    heatmap(L)
    plot!(softplus.(trc[2]), softplus.(trc[1]))
end


# %% --------
# logp(Bear2020(1, 0.1), sim_data)
W = Bear2020(1, 0.1)
datum = sim_data[1]

# %% --------
figure() do
    row = data[1, :]
    u, p = get_problem(row)
    g = first(G)
    histogram(g.sample, normalize=:pdf)
    plot!(weight(mle_bear, u, p))
    # plot!(weight(Bear2020(.1, 0.1), u, p))
end
# %% --------


# %% --------
sim_data = let
    row = data[1, :]
    u, p = get_problem(row)
    model = Bear2020(1, 0.1)
    w = weight(model, u, p)
    map(1:1000) do i
        (;row.bimodal, row.ideal, sample=sample(Weights(w)))
    end
end |> DataFrame
# %% --------

% LOW IDEAL
idealL = max(X)-[0 X];
idealL_PDF = @(x,T) exp(idealL(x+1)/T);



%   Softmax Model
uniLSM = @(x,I) (idealL_PDF(x,I(1)).*unimodalPDF(x+1) + I(2));
biLSM = @(x,I) (idealL_PDF(x,I(1)).*bimodalPDF(x+1) + I(2));

NLLL_uni = @(I) -sum(log(uniLSM(sample(ideal==0 & bimodal==0 & passComp==1),I)/sum(uniLSM(0:100,I))));
NLLL_bi = @(I) -sum(log(biLSM(sample(ideal==0 & bimodal==1 & passComp==1),I)/sum(biLSM(0:100,I))));


# %% --------
idealL =;
findfirst(x->x≠60, idealL)
findfirst(x->x≠60, _value[:low])


idealL_PDF = @(x,T) exp(idealL(x+1)/T);

unimodalPDF = normpdf(0:100,45,15);
bimodalPDF = (normpdf(0:100,35,5) + normpdf(0:100,75,5));
uniLSM = @(x,I) (idealL_PDF(x,I(1)).*unimodalPDF(x+1) + I(2));
biLSM = @(x,I) (idealL_PDF(x,I(1)).*bimodalPDF(x+1) + I(2));

NLLL_uni = @(I) -sum(log(uniLSM(sample(ideal==0 & bimodal==0 & passComp==1),I)/sum(uniLSM(0:100,I))));
NLLL_bi = @(I) -sum(log(biLSM(sample(ideal==0 & bimodal==1 & passComp==1),I)/sum(biLSM(0:100,I))));


G = groupby(data, [:ideal, :bimodal])

g = first(G)



0: low
1: high
2: middle


idealL = max(X)-[0 X];


# NLL_Softmax = NLLL_uni(outputsL_uni) + NLLL_bi(outputsL_bi) + ...
#             NLLH_uni(outputsH_uni) + NLLH_bi(outputsH_bi) + ...
#             NLLM_uni(outputsM_uni) + NLLM_bi(outputsM_bi)