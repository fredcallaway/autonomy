using StatsBase
using Distributions
function p_sample(α, u, p)
    x = @. α * exp(u) + (1-α) * p
    # x = @. (α * exp(u) + (1-α) * abs(u)) * p
    x ./= sum(x)
end

N_MC = 1000

subjective_value(α, s, u, p) = mean(sample(u, Weights(p_sample(α, u, p)), s; replace=false))
objective_value(u, p, k) = maximum(sample(u, Weights(p), k; replace=false))

function sample_problem(n, signal=0.1)
    u = randn(n) .+ signal * randn()
    p = rand(Dirichlet(ones(n)))
    (u, p)
end

choice_prob(α, s, u, p) = N_MC \ mapreduce(+, 1:N_MC) do i
    subjective_value(α, s, u, p) > 0
end

choice_value(u, p, k) = N_MC \ mapreduce(+, 1:N_MC) do i
    objective_value(u, p, k)
end

function sample_value(n, k, s; αs=0:0.1:1)
    u, p = sample_problem(n)
    val = choice_value(u, p, k)
    prob = [choice_prob(α, s, u, p) for α in αs]
    prob .* val
end
