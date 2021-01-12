using Plots
using Dates
mkpath("fighist")
mkpath("figs")
gr(label="", dpi=200, size=(400,300))
ENV["GKSwstype"] = "nul"

function figure(f, name="tmp"; kws...)
    plot(;kws...)
    f()
    dt = Dates.format(now(), "m-d-H-M-S")
    p = "fighist/$dt-$name.png"
    savefig(p)
    if name != "tmp"
        cp(p, "figs/$name.png"; force=true)
    end
end
