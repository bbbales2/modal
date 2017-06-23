library(tidyverse)
library(rstan)
library(ggplot2)
library(gridExtra)

setwd("~/modal/paper/cmsx4")

df = read_csv("hmc_30_noprior_data_warmup.csv", col_names = TRUE)
df %>% summary()

p1 = df %>% ggplot(aes(cu1, cu2)) + geom_point(alpha = 0.25) + geom_point(aes(x = 0, y = 0), col = "orange")
p2 = df %>% ggplot(aes(cu1, cu3)) + geom_point(alpha = 0.25) + geom_point(aes(x = 0, y = 0), col = "orange")
p3 = df %>% ggplot(aes(cu2, cu3)) + geom_point(alpha = 0.25) + geom_point(aes(x = 0, y = 0), col = "orange")
p4 = df %>% ggplot(aes(angle)) + geom_histogram() + xlab("Misoriention between posterior and measured")#Angle between posterior and\n measured orientations")

grid.arrange(p1, p2, p3, p4, nrow = 2, ncol = 2, top = "Misorientation")

q1 = df %>% group_by(chain) %>% mutate(sample = row_number()) %>% ungroup() %>%
  ggplot(aes(sample, c11)) +
  geom_line(aes(group = chain), alpha = 0.75) +
  facet_grid(chain ~ ., labeller = "label_both") +
  ylab(expression(c[11]))
q2 = df %>% group_by(chain) %>% mutate(sample = row_number()) %>% ungroup() %>%
  ggplot(aes(sample, a)) +
  geom_line(aes(group = chain), alpha = 0.75) +
  facet_grid(chain ~ ., labeller = "label_both") +
  ylab("Anisotropy ratio")
q3 = df %>% group_by(chain) %>% mutate(sample = row_number()) %>% ungroup() %>%
  ggplot(aes(sample, c44)) +
  geom_line(aes(group = chain), alpha = 0.75) +
  facet_grid(chain ~ ., labeller = "label_both") +
  ylab(expression(c[44]))
q4 = df %>% group_by(chain) %>% mutate(sample = row_number()) %>% ungroup() %>%
  ggplot(aes(sample, std)) +
  geom_line(aes(group = chain), alpha = 0.75) +
  facet_grid(chain ~ ., labeller = "label_both") +
  ylab(expression(sigma))

grid.arrange(q1, q2, q3, q4, nrow = 2, ncol = 2, top = "Elastic constants")

df3 = df %>%
  rename("c[11]" = "c11", "A" = "a", "c[44]" = "c44", "sigma" = "std") %>%
  gather(param, value, c("c[11]", "A", "c[44]", "sigma"))

df4 = df3 %>% group_by(param) %>%
      summarize(m = mean(value), s = sd(value), lx = min(value), ux = max(value))

df5 = tibble()
for(i in 1:nrow(df4)) {
  a = as_tibble(list(x = seq(df4$lx[[i]], df4$ux[[i]], length = 100)))
  a$y = dnorm(a$x, df4$m[[i]], df4$s[[i]])
  a$param = df4$param[[i]]
  df5 = bind_rows(df5, a)
}

df3 %>%
  ggplot(aes(value)) +
  geom_histogram(aes(y = ..density..), fill = "grey40", col = "grey77", size = 0.25) +
  geom_line(data = df5, aes(x, y), col = "orange", size = 1.5) +
  facet_wrap( ~ param, scales = "free", labeller = label_parsed) + ylab("") + xlab("") + labs(y = NULL, x = NULL) +
  theme(strip.text = element_text(size=12))


library(scales)

fmt_dcimals = function(){
  function(x) {
    if(abs(x - 1.0) < 0.1) {
      return(sprintf("%.4f", x))
    } else {
      return(sprintf("%.3f", x))
    }
  }
}

number_ticks = function(n) {
  function(limits) {
    m = mean(limits)
    l = quantile(limits, 0.15)
    u = quantile(limits, 0.85)
    c(l, m, u)#format(, nsmall = n, scientific = FALSE)
  }
}

test = list(c11 = expression(c[11]))

to_expression = function(x) {
  #out = list()
  #for(i in 1:length(x)) {
  #  out[i] = expression(c[11])
  #}
  #out
  rep(expression("c[11]"), length(x))
}

library(ggthemes)

df %>% group_by(chain) %>% mutate(sample = row_number()) %>% ungroup() %>%
  rename("c[11]" = "c11", "A" = "a", "c[44]" = "c44", "sigma" = "std") %>%
  gather(param, value, c("c[11]", "A", "c[44]", "sigma", "w", "x", "y", "z")) %>%
  ggplot(aes(sample, value)) +
  geom_line(alpha = 1.0, size = 0.15) +
  ylab("") + labs(y = NULL) +
  scale_y_continuous(breaks = number_ticks(3), labels = fmt_dcimals()) +
  facet_grid(param ~ chain, scales = "free_y",
             labeller = labeller(chain = label_both, param = label_parsed))# +
  #theme(panel.grid.major = element_line(colour = "grey75"),
  #      panel.background = element_rect(fill = 'grey99', colour = 'grey75'))#label_bquote(cols = chain: .(chain), rows = parse(.(param)))
  #theme_bw()

df %>% group_by(chain) %>% mutate(sample = row_number()) %>% ungroup() %>% gather(parameter, value, 2:8)

df %>% select(a, c11, c44, std, w, x, y, z, chain) %>% gather(param, value, 1:8) %>%
  group_by(param) %>%
  summarize(mean = mean(value), lower = quantile(value, 0.025), upper = quantile(value, 0.975), sd = sd(value)) %>%
  print(n = 100)

