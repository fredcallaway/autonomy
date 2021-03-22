
# %% ==================== Libraries ====================

library(tidyverse)
library(magrittr)
library(ggbeeswarm)

library(lme4)
library(stickylabeller)
library(lemon)
library(jtools)
library(jsonlite)

options(
    "summ-model.info"=FALSE, 
    "summ-model.fit"=FALSE, 
    "summ-re.table"=FALSE, 
    "summ-groups.table"=FALSE,
    "jtools-digits"=3,
    "pillar.subtle" = FALSE

)

kable = knitr::kable
glue = glue::glue

knitr::opts_chunk$set(warning=FALSE, message=FALSE, fig.width=5, fig.height=4, fig.align="center")


# %% ==================== Plotting ====================

RED =  "#E41A1C" 
BLUE =  "#377EB8" 
GREEN =  "#4DAF4A" 
PURPLE =  "#984EA3" 
ORANGE =  "#FF7F00" 
YELLOW =  "#FFDD47" 
GRAY = "#ADADAD"
BLACK = "#111111"

# theme_set(theme_bw(base_size = 14) + theme(strip.background = element_blank()))
# theme_set(theme_classic(base_size = 14) + theme(strip.background = element_blank()))

theme_set(theme_bw(base_size = 14))
theme_update(
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank()
)

update_geom_defaults("line", list(size = 1.2))

system('mkdir -p figs')
system('mkdir -p .fighist')
fig = function(name="tmp", w=4, h=4, dpi=320, ...) {
    if (isTRUE(getOption('knitr.in.progress'))) {
        show(last_plot())
        return()
    }
    ggsave("/tmp/fig.png", width=w, height=h, dpi=dpi, ...)
    stamp = format(Sys.time(), "%m-%d-%H-%M-%S")
    p = glue('.fighist/{name}-{stamp}.png')
    system(glue('mv /tmp/fig.png {p}'))
    system(glue('cp {p} figs/{name}.png'))
    # invisible(dev.off())
    # knitr::include_graphics(p)
}


geom_line_range = list(
    stat_summary(fun=mean, geom="line", position = position_dodge(width = .05)),
    stat_summary(fun.data=mean_cl_boot, position = position_dodge(width = .05))
)

plot_line_range = function(df, x, y, z) {
    if (missing(z)) {
        ggplot(df, aes({{x}}, {{y}}, group=0)) + geom_line_range
    } else {
        ggplot(df, aes({{x}}, {{y}}, color={{z}}, group={{z}})) + geom_line_range
    }
}

no_grid = theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
)

# %% ==================== String stuff ====================

sprintf_transformer <- function(text, envir) {
    m <- regexpr(":.+$", text)
    if (m != -1) {
        format <- substring(regmatches(text, m), 2)
        regmatches(text, m) <- ""
        res <- eval(parse(text = text, keep.source = FALSE), envir)
        do.call(sprintf, list(glue("%{format}"), res))
    } else {
        eval(parse(text = text, keep.source = FALSE), envir)
    }
}

# mimics python string formatting e.g. fmt("{pi:.2f}") == "3.14"
fmt <- function(..., .envir = parent.frame()) {
    glue(..., .transformer = sprintf_transformer, .envir = .envir)
}

pval = function(p) {
    if (p < .001) "p < .001" else glue("p = {str_sub(format(round(p, 3)), 2)}")
}

write_tex = function(tex, file) {
    print(fmt("{file}: {tex}"))
    writeLines("{tex}\\unskip", file)
}

# %% ==================== Miscellanea ====================

only = function(xs) {
    u = unique(xs)
    stopifnot(length(u) == 1)
    u[1]
}

zscore = function(x) (x - mean(x, na.rm=T)) / sd(x, na.rm=T)


