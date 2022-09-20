function weigh_ksets(est::Estimator, env::Env, s::State)
    @assert !est.replace
    idx = sortperm(s.u_true)
    u_true = s.u_true[idx]
    u_init = s.u_init[idx]
    w = weight(est.weighter, u_init).values
    ksets = collect(combinations(eachindex(u_true), env.k))
    kset_weights = map(ksets) do ks
        prod(w[ks])
    end
    kset_weights ./= sum(kset_weights)
    kset_vals = map(ks->u_true[ks[end]], ksets)
    (kset_weights, kset_vals)
end

weighter = AbsExp(;β=0.1, d=0)
est = SampleMeanEstimator(;n_sample=1, α=0, replace=false, weighter)

env = Env(n=10)
env = mutate(env, μ_outcome=-expected_value(env))
s = State(env)

ksw, ksv = weigh_ksets(est, env, s)

@test ksw' * ksv ≈ expected_maximum_srswor(env.k, s.u_true)

# %% ==================== check set weights ====================

function naive_sample_set_with_replace(w, k)
    out = Dict{Vector{Int},Float64}()
    sample_seqs = collect(product((eachindex(w) for i in 1:k)...))[:]
    for seq in sample_seqs
        seq = sort(collect(seq))
        out[seq] = get(out, seq, 0) + prod(w[seq])
    end
    return out
end
# https://www.sciencedirect.com/science/article/abs/pii/S002001900500298X
function sample_set_distribution(w, k; replace)
    if replace
        k = 3
        collect(with_replacement_combinations(eachindex(w), k))
        c = countmap(sort.(collect.(sample_seqs)))
        sort(collect(c), by=x->x[2])



        out = Dict{Vector{Int},Float64}()
        sample_seqs = collect(product((eachindex(w) for i in 1:k)...))[:]
        for seq in sample_seqs
            seq = sort(collect(seq))
            out[seq] = get(out, seq, 0) + prod(w[seq])
        end
        return out
    else
        sample_sets = collect(combinations(eachindex(w), k))
        set_weights = map(sample_sets) do ss
            mapreduce(+, permutations(ss)) do perm
                p = 1
                denom = sum(w)
                for i in perm
                    p *= (w[i] / denom)
                    denom -= w[i]
                end
                p
            end
        end
        sample_sets, set_weights
    end
end


@time sample_set_distribution(collect(1:5) ./ sum(1:5), 3; replace=true)
@time naive_sample_set_with_replace(collect(1:3) ./ sum(1:3), 3) |> sort
# @time naive_sample_set_with_replace(collect(1:5) ./ sum(1:5), 3) |> sort

# %% --------
env = Env(n=4); s = State(env)
s.u_true .= s.u_init .= 3 .*[float(10)^-(i-1) for i in 1:env.n]

function subjective_value_dist(est::SampleMeanEstimator, s::State)
    w = weight(est.weighter, s.u_init).values
    if est.replace
        product((eachindex(s.u_true) for i in 1:est.n_sample)...) |> length
    else
        sample_sets, set_weights = sample_set_distribution(w, est.n_sample; replace=false)
        set_vals = map(ss->mean(s.u_true[ss]), sample_sets)
        (set_vals, set_weights)
    end
end

est = SampleMeanEstimator(;n_sample=3, α=0, replace=false, weighter=AbsExp(;β=0.5, d=1))

# esv = round.([subjective_value(est, env, s) for i in 1:1000000]; digits=4);
# empirical = countmap(esv) |> sort |> values |> collect;
# empirical = empirical ./ sum(empirical)

sv, sw = subjective_value_dist(est, s);
sv = round.(sv; digits=4);
analytic = sw[sortperm(sv)]
analytic .- empirical
# @test Set(esv) == Set(sv)

# %% --------
env = Env(n=12); s = State(env)
est = SampleMeanEstimator(;n_sample=8, α=0, replace=false, weighter=AbsExp(;β=0.5, d=1))
@time subjective_value_dist(est, s)
nothing




# %% ==================== check evaluation ====================



ev = Evaluator(env, 1000)
ev(choice_objective, est)

# %% --------
map(ev.states, ev.true_vals) do s, true_val
    ksw, ksv = weigh_ksets(est, s.u_true, est.n_sample)
    mapreduce(+, ksw, ksv) do w, v
        w * choice_objective(v, true_val)
    end
end |> mean


