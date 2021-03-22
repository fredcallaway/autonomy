source("base.r")

# mc_absexp = read_csv("../model/results/mc_absexp.csv") %>% filter(dp == 1)
absexp = read_csv("../model/results/optim_de_absexp.csv")


# %% --------


plot_by = function(data, x) {
    ggplot(data, aes(k, {{ x }}, color=replace)) + 
        geom_line() +
        facet_grid(σ_init ~ n_sample, labeller = label_both)
    fig(glue("optim_{deparse(substitute(x))}"), 7, 7)
}

plot_by(absexp, β)
plot_by(absexp, α)
plot_by(absexp, d)
