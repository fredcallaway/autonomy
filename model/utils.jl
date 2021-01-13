function mutate(x::T; kws...) where T
    for field in keys(kws)
        if !(field in fieldnames(T))
            error("$(T.name) has no field $field")
        end
    end
    return T([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end

function grid(;kws...)
    X = map(Iterators.product(values(kws)...)) do x
        (; Dict(zip(keys(kws), x))...)
    end
    KeyedArray(X; kws...)
end

function keyed(name, xs)
    KeyedArray(xs; Dict(name => xs)...)
end

function monte_carlo(f, N=10000)
    N \ mapreduce(+, 1:N) do i
        f()
    end
end

"Expected maximum of N samples from a Normal distribution"
function expected_maximum(N::Int, d::Normal)
    mcdf(x) = cdf(d, x)^N
    lo = d.μ - 10d.σ; hi = d.μ + 10d.σ
    - quadgk(mcdf, lo, 0, atol=1e-5)[1] + quadgk(x->1-mcdf(x), 0, hi, atol=1e-5)[1]
end
