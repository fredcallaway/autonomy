using Revise
includet("utils.jl")
includet("model.jl")
includet("figure.jl")

using StatsFuns
using Optim

# %% --------
d = Normal(0, 1)

figure("pdf_max_vs_mean") do
    x = -4:.001:4
    p1 = plot(xlabel="Value", ylabel="P(Max)", legend=false)
    for k in 1:4
        y = @. k * pdf(d, x) * cdf(d,x)^(k-1)
        plot!(x, y, color=k)
    end
    p2 = plot(xlabel="Value", ylabel="P(Max) / P(Occur)", legend=:topleft)
    for k in 1:4
        y = @. k * pdf(d, x) * cdf(d,x)^(k-1) / pdf(d, x)
        plot!(x, y, label="k=$k", color=k)
    end
    plot(p1, p2, size=(700, 300))
end


# %% ==================== Compare to Bear ====================

function fit_bear(k, weighter)
    u = -10:.01:10
    p = pdf.(Normal(0, 1), u)
    p ./= sum(p)
    w_ana = weight(weighter, u, p)
    res = optimize([0, 0.], LBFGS()) do x
        B = Bear2020(softplus.(x)...)
        w = weight(B, u, p)
        # sum(@. (w - w_ana)^2)
        -sum(@. w_ana * log(w))
    end
    Bear2020(softplus.(res.minimizer)...)
end

function plot_comparison(weighter)
    u = -3:.01:3
    p = pdf.(Normal(0, 1), u)
    p ./= sum(p)
    k = weighter.k
    w_ana = weight(weighter, u, p)
    B = fit_bear(k, weighter)
    p1 = let
        plot(xlabel="Value", ylabel="P(Sample)", legend=:topleft)
        plot!(u, p, color=colorant"#aaa", alpha=0.5, label="Default")
        plot!(u, weight(weighter, u, p), label="Analytic", color=1)
        plot!(u, weight(B, u, p), label="Softmax", color=2)
    end
    p2 = let
        plot(xlabel="Value", ylabel="P(Sample) / P(Occur)", legend=false)
        plot!(u, ones(length(u)), color=colorant"#aaa", alpha=0.5, label="Default")
        plot!(u, weight(weighter, u, p) ./ p, label="Analytic", color=1)
        plot!(u, weight(B, u, p) ./ p, label="Softmax", color=2)
    end
    plot(p1, p2, size=(700, 300))
end

figure("analytic_vs_softmax") do
    plot_comparison(Analytic(2))
end

figure("analytic_vs_softmax_uws") do
    plot_comparison(AnalyticUWS(2))
end

# %% ==================== Finite sets ====================
using Combinatorics

figure("analytic_finite") do
    u = 1:20
    p = ones(length(u)) / length(u)
    F = ecdf(u).(u)
    plot(legend=:topleft,  xflip=true, xlabel="Rank", ylabel="P(Max) / P(Occur)",)
    foreach(1:4) do k
        maxs = map(maximum, combinations(u, k))
        pmax = counts(maxs, u) ./ length(maxs)
        plot!(reverse(u), pmax ./ p, color=k, label="k=$k")
        plot!(reverse(u), k .* F.^(k-1), color=k, ls=:dot)
    end
end

# %% --------
xs = -10:.05:10
function get_curves(n, k)
    us = map(1:1000) do i
        randn(n)
    end
    p = ones(n) / n
    A = Analytic(k)
    B = fit_bear(k, A)

    map([A, B]) do W
        map(xs) do x
            length(us) \ mapreduce(+, us) do u
                u[1] = x
                weight(W, u, p)[1]
            end
        end
    end
end

figure() do
    plots = map([2,4,8,16]) do n
        curves = get_curves(n, 2)
        plot(xs, curves[1], color=1)
        plot!(xs, curves[2], color=2)
    end
    plot(plots..., size=(600,600))
end

# %% --------
u = randn(10)
figure() do
    x = -10:.1:10
    y = map(x) do x
        u[1] = x
        n .* weight(W, u, p)[1]
    end
    plot(x, y)
end



# %% --------
figure() do
    x = -4:.01:1
    plot(x, exp.(x))
end

# %% --------
figure() do
    plot()
    for i in 1:10
        scatter!(u, weight(A, u, p), color=1)
        scatter!(u, weight(B, u, p), color=2)
    end
end
# %% --------
figure("analytic_finite") do
    plot(legend=:topleft,  xflip=true, xlabel="Rank", ylabel="P(Max) / P(Occur)",)
    foreach(1:4) do k
        maxs = map(maximum, combinations(u, k))
        pmax = counts(maxs, u) ./ length(maxs)
        plot!(reverse(u), pmax ./ p, color=k, label="k=$k")
        plot!(reverse(u), k .* F.^(k-1), color=k, ls=:dot)
    end
end
# %% --------



















# %% --------
figure("analytic_vs_softmax_multik") do
    # u = sort!(rand.(Normal(0, 1), 1000))
    # p = pdf.(Normal(0, 1), u)
    # p ./= sum(p)
    plots = map(bears, 1:4) do B, k
        plot(xlabel="Value", ylabel="P(Sample) / P(Occur)", legend=false)
        plot!(u, weight(Analytic(k), u, p) ./ p, label="Analytic")
        plot!(u, weight(B, u, p) ./ p, label="Softmax")
    end
    plot!(plots[1], legend=:topleft)
    plot(plots..., size=(600,600), layout=(2, 2))
    # B2 = mutate(B1, C=3)
    # plot!(u, weight(B2, u, p), label="Softmax alt")
end



# %% --------
k = 2
n = 10
u = sort!(randn(n) .* 0.5)
p = ones(n) ./ n
u[end] = 2

figure() do
    plot(xlabel="Value", ylabel="P(Sample)", legend=:topleft)
    scatter!(u, weight(Analytic(k), u, p), label="Analytic")
    scatter!(u, weight(bears[k], u, p), label="Softmax")
end
# %% --------
figure() do
    ps = map(zip(problems[1:10], analytic_weights)) do ((u,p), what)
        scatter(u, what, label="Analytic", legend=:topleft)
        scatter!(u, weight(B1, u, p), label="Bear")
    end
    plot(ps..., size= (1000, 1000))
end

# %% --------

# %% --------
@. logistic(x, (L, k, x0)) = L / (1 + exp(-k * (x - x0)))

using LsqFit
function logistic_fit(x, y, p0=[1., 1, 0])
    fit = curve_fit(logistic, x, y, p0)
    @show fit.param
    logistic(x, fit.param)
end

# %% --------
figure("pdf_max_vs_mean_numerical") do
    plot(xlabel="Value", ylabel="P(Max) / P(Occur)", legend=:topleft)
    x = xs[2:end]
    pp = pdf.(d, x)
    for (k, y) in enumerate(eachcol(1000 .* X ./ pp))
        plot!(x, y, label="k=$k")
        yhat = @. k * pdf(d, x) * cdf(d,x)^(k-1) / pdf(d, x)
        plot!(x, yhat, line=(:black, :dot))
        # yhat = logistic_fit(x, y)
        # plot!(x, yhat, line=(:black, :dot))
    end
end


# %% --------
k = 2
figure() do
    plot(xs, @. 4pdf(d, xs) * cdf(d,xs)^k / pdf(d, x))
end
# %% --------

function probmax(k, x)
    k * pdf(d, x) * cdf(d,x)^(k-1) / pdf(d, x)
end
function probmax2(k, x)
    k * pdf(d, x) * cdf(d,x)^(k-1) / pdf(d, x)
end

figure() do
    plot(xs, probmax.(2, xs))
end

# %% --------

env = Env(n=10)
problems = map(1:1000) do i
    sample_problem(env)
end

analytic_weights = map(problems) do (u, p)
    weight(Analytic(env.k), u, p)
end

res = optimize([0., 0.], LBFGS()) do x
    B = Bear2020(softplus.(x)...)
    mapreduce(+, problems, analytic_weights) do (u, p), what
        w = weight(B, u, p)
        # sum(@. (w - what)^2)
        -sum(@. what * log(w))
    end
end
B1 = Bear2020(softplus.(res.minimizer)...)



# %% --------
# d = Normal(0, 1)
# ks = 1:3
# xs = -6:.001:6
# X = mapreduce(hcat, ks) do N
#     mcdf(x) = cdf(d, x)^N
#     diff(mcdf.(xs))
# end
x = xs[2:end]
pp = pdf.(d, x)
P = map(ks, eachcol(1000 .* X ./ pp)) do k, y
    fit = curve_fit(logistic, x, y, [k, 1., 0])
    fit.param
end
using SplitApplyCombine
L, k, x0 = invert(P)
L
# %% --------

figure() do
    plot(ks, k)
    plot!(ks, x0)
end
# %% --------
figure() do
    plot(diff(L[1:end-12]) ./ 5)
end
# %% --------
