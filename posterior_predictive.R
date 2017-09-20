library(tidyverse)
library(ggplot2)
library(rstan)

fit = read_stan_csv(c('/home/bbales2/Downloads/Output_Co-Ternary-A_12P-50N_cubic-rot-BRG7_1.csv'))

data = c(119.327,
            129.230,
            140.052,
            144.546,
            148.656,
            166.530,
            170.043,
            186.321,
            201.575,
            201.925,
            213.841,
            214.024,
            220.865,
            225.234,
            236.587,
            239.489,
            242.658,
            246.495,
            252.652,
            257.464,
            259.165,
            265.686,
            266.700,
            275.227,
            281.265,
            281.591,
            284.171,
            296.162,
            297.823,
            301.038,
            304.291,
            310.243,
            316.550,
            321.205,
            325.111,
            330.867,
            336.536,
            339.727,
            341.525,
            345.256,
            348.678,
            351.825,
            353.813,
            358.382,
            364.554,
            365.722,
            371.334,
            371.507,
            375.002,
            379.626)

# Postrior predictives
x = extract(fit, c('res', 'sigma')) %>%
  (function(x) {
    t(sapply(x$sigma, function(sig) rnorm(length(data), 0, sig))) + x$res
    }) %>%
  as.tibble %>%
  (function(yhat) yhat - t(replicate(nrow(yhat), data))) %>%
  as.tibble %>%
  setNames(1:length(data)) %>%
  gather(mode, error) %>%
  mutate(mode = as.integer(mode)) %>%
  group_by(mode) %>%
  summarize(median = median(error),
            q25 = quantile(error, 0.025),
            q975 = quantile(error, 0.975)) %>%
  ggplot(aes(mode)) +
  geom_linerange(aes(ymin = q25, ymax = q975)) +
  geom_point(aes(mode, median)) +
  geom_hline(aes(yintercept = 0.0), color = "red") +
  xlab("Resonance modes") +
  ylab("Error (Khz)") +
  ggtitle("Medians and 95% posterior intervals of error (yrep - y)\n(Red line is data)")
