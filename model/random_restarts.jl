include("box.jl")
using Optim
using Sobol

function random_restarts(loss, hard_lower, soft_lower, soft_upper, hard_upper, n_restart; 
                         #algorithm=LBFGS(), 
                         algorithm=LBFGS(),
                         iterations=100, g_tol=1e-5, L2=1e-8,
                         max_err=30, max_timeout=50, max_finite=30, id="null")
    n_err = 0
    n_time = 0
    n_finite = 0
    box = Box(hard_lower, hard_upper)
    function wrap_loss(x)
        squashed = squash!(box, copy(x))
        loss(squashed) + L2 * sum(x .^ 2)  # tiny bit of regularization stabilizes optimization
    end

    function do_opt(x0)
        if !isfinite(loss(x0))  # hopeless!
            n_finite += 1
            @debug "nonfinite loss" n_finite
            if n_finite > max_finite
                @error "$id: Too many non-finite losses while optimizing"
                error("Optimization non-finite")
            end
            return missing
        end
        try
            # >~30s indicates that the optimizer is stuck, which means it's not likely to find a good minimum anyway
            res = optimize(wrap_loss, unsquash!(box, copy(x0)), algorithm,
                Optim.Options(;g_tol, iterations))
            if !(res.f_converged || res.g_converged) && res.iterations > iterations
                n_time += 1
                @debug "timeout" n_time
                if n_time > max_timeout
                    @error "$id: Too many timeouts while optimizing"
                    error("Optimization timeouts")
                end
                return missing
            elseif !isfinite(wrap_loss(res.minimizer))
                @error "Nonfinite final loss!" wrap_loss(res.minimizer)
                return missing
            else
                squash!(box, res.minimizer)
                return res
            end
        catch err
            err isa InterruptException && rethrow(err)
            n_err += 1
            #@warn "error" err n_err
            if n_err > max_err
                @error "$id: Too many errors while optimizing"
                rethrow(err)
            end
            return missing
        end
    end

    x0s = SobolSeq(soft_lower, soft_upper)
    results = Any[]
    while length(results) < n_restart
        res = do_opt(next!(x0s))
        if !ismissing(res)
            push!(results, res)
        end
    end
    if n_err > max_err/2 || n_time > max_timeout/2 || n_finite > max_finite/2
        @warn "$id: Difficulty optimizing" n_err n_time n_finite
    end
    losses = getfield.(results, :minimum)
    very_good = minimum(losses) * 1.05
    n_good = sum(losses .< very_good)
    #if n_good < 5
    #    best_losses = partialsort(losses, 1:5)
    #    @warn "$id: Only $n_good random restarts produced a very good minimum" best_losses
    #end
    partialsort(results, 1; by=o->o.minimum)  # best result
end