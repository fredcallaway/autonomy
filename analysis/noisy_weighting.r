source("base.r")

# mc_absexp = read_csv("../model/results/mc_absexp.csv") %>% filter(dp == 1)
absexp = read_csv("../model/results/absexp.csv")

absexp %>% 
    select(-value, -se) %>% 
    map(~unique(.))



# %% ==================== Abs Exp ====================
absexp %>%
    group_by(objective, k, n_sample) %>%
    slice(which.max(value)) %>% print(n=100)

# %% --------
absexp %>%
    filter(objective=="choice") %>% 
    filter(d == 2) %>% 
    filter(α == 0) %>% 
    group_by(k, σ_init, n_sample, β) %>%
    ggplot(aes(β, value, color=factor(n_sample), group=factor(n_sample))) +
        geom_line() +
        facet_grid(σ_init ~ k, scales="free_y", labeller = label_both) +
        geom_hline(aes(yintercept = 0)) +
        theme_bw()

fig("absexp_choice", 10, 7)

# %% --------

absexp %>%
    group_by(objective, k, n_sample, α) %>%
    filter(k == 1, σ_init == 0) %>% 
    ggplot(aes(β, value, color=factor(n_sample), group=factor(n_sample))) +
        geom_line()

fig("")

# %% ==================== Investigate α ====================

absexp %>%
    filter(objective=="choice") %>% 
    group_by(k, n_sample, σ_init) %>%
    slice_max(value) %>%
    # filter(α > 0)
    identity
    # ungroup %>% count(α)

# %% --------
plot_by = function(data, x) {
    data %>% 
        filter(objective == "choice") %>% 
        group_by(k, n_sample, σ_init, {{ x }}) %>%
        summarise(value=max(value)) %>% 
        ggplot(aes({{ x }}, value, color=factor(n_sample), group=factor(n_sample))) +
            geom_line() +
            facet_grid(σ_init ~ k, scales="free_y", labeller = label_both) +
            geom_hline(aes(yintercept = 0)) +
            theme_bw()
    fig(glue("kσ_{deparse(substitute(x))}"), 10, 7)
}

plot_by(absexp, α)
plot_by(absexp, β)
plot_by(absexp, d)  3
# %% --------

# conclude: α is not useful

# %% --------

# absexp %>% group_by(α) %>% summarise(value=max(value))
foo = function(x) {absexp %>% group_by({{x}}) %>% summarise(value=max(value))}

foo(α)

# %% --------
absexp %>% filter(objective == "choice") %$% min(value)

    # slice(which.max(value)) %>% print(n=30)

```{r}
