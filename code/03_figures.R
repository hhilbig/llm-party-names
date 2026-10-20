#!/usr/bin/env Rscript
# Figures for the README and the results page. Run from the repository root
# after code/02_analyze.py. Writes PDF and PNG to figures/.

set.seed(20261010)

library(dplyr)
library(ggplot2)
library(readr)
library(scales)

input_dir <- "results"
out_dir <- "figures"
dir.create(out_dir, showWarnings = FALSE)

DARK <- "#1d4f77"
GREY <- "grey55"
LIGHT <- "grey85"

theme_clara_twitter <- function(fontsize = 13) {
  theme_bw(base_size = fontsize, base_family = "Helvetica") +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(color = "grey70", linewidth = 0.45),
      axis.ticks = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = fontsize - 1, hjust = 0),
      legend.position = "none",
      plot.title.position = "plot",
      plot.caption.position = "plot",
      plot.title = element_text(face = "bold", size = fontsize + 6),
      plot.subtitle = element_text(color = "grey30", lineheight = 1.15, margin = margin(b = 12)),
      plot.caption = element_text(
        color = "grey35", size = fontsize - 3, hjust = 0,
        lineheight = 1.15, margin = margin(t = 12)
      ),
      plot.margin = margin(12, 34, 10, 10)
    )
}

save_fig <- function(plot, stem, width, height) {
  pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
  png_path <- file.path(out_dir, paste0(stem, ".png"))
  ggsave(pdf_path, plot, width = width, height = height, device = cairo_pdf)
  ggsave(png_path, plot, width = width, height = height, dpi = 300, bg = "white")
  stopifnot(file.info(pdf_path)$size > 5000, file.info(png_path)$size > 10000)
}

benoit <- read_csv(file.path(input_dir, "benoit_masking.csv"), show_col_types = FALSE)
tweets <- read_csv(file.path(input_dir, "tweet_masking.csv"), show_col_types = FALSE) %>%
  mutate(group = recode(group, commercial = "Commercial", flagship = "Commercial",
                        `open-weight` = "Open-weight", `open-new` = "Open-weight"))
tweet_items <- read_csv(file.path(input_dir, "tweet_items.csv"), show_col_types = FALSE) %>%
  filter(group %in% c("commercial", "flagship"), model != "GPT-6.1 Sol")

model_order <- benoit %>%
  filter(domain == "economic", passage == "named") %>%
  arrange(con_lab_error_gap_orig) %>%
  pull(model)

# Reading guide under the x-axis zero line, placed in the bottom panel(s).
axis_guide <- function(group, versions) {
  tidyr::expand_grid(version = factor(versions, levels = versions),
                     tibble(x = c(0, 0), hjust = c(1.05, -0.05),
                            label = c("\u2190 less apart", "more apart \u2192"))) %>%
    mutate(group = group)
}

# Figure 1: Republican-minus-Democratic error gap against the human raters
# (model minus mean human rating), tweets that name someone, as written and
# masked. Rows: commercial and open-weight models; columns: the two versions.
excess <- bind_rows(
  tweets %>% filter(passage == "named") %>%
    transmute(group, model, version = "Names shown", est = rep_dem_error_gap_orig,
              lo = gap_orig_lo, hi = gap_orig_hi),
  tweets %>% filter(passage == "named") %>%
    transmute(group, model, version = "Names masked", est = rep_dem_error_gap_mask,
              lo = gap_mask_lo, hi = gap_mask_hi)
)
excess_order <- excess %>%
  filter(version == "Names shown") %>%
  arrange(group, est) %>%
  pull(model)
excess <- excess %>%
  mutate(model = factor(model, levels = excess_order),
         version = factor(version, levels = c("Names shown", "Names masked")))
fig_ex <- ggplot(excess, aes(est, model, color = version)) +
  geom_vline(xintercept = 0, color = "grey60", linewidth = 0.5) +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0, linewidth = 0.8, orientation = "y") +
  geom_point(size = 2.9) +
  geom_text(data = axis_guide("Open-weight", c("Names shown", "Names masked")),
            aes(x = x, y = -Inf, label = label, hjust = hjust), inherit.aes = FALSE,
            vjust = -0.7, size = 3.6, color = "grey35", family = "Helvetica") +
  facet_grid(group ~ version, scales = "free_y", space = "free_y") +
  scale_color_manual(values = c("Names shown" = DARK, "Names masked" = GREY)) +
  scale_y_discrete(expand = expansion(add = c(1.2, 0.6))) +
  labs(
    x = "Distance between the parties, model minus human coders (points)",
    y = NULL,
    caption = paste0(
      "144 tweets by members of Congress that name a party or politician (Le Mens and Gallego); human coders saw the names.\n",
      "0-100 left-right scale. Masked: names replaced by [PARTY] or [POLITICIAN]. 95% intervals."
    )
  ) +
  theme_clara_twitter() +
  theme(strip.text.y = element_text(angle = 0, hjust = 0), panel.spacing = unit(0.8, "lines"))
save_fig(fig_ex, "01_tweets_excess_separation", width = 10, height = 6)

# Figure 2: two example tweets, commercial models with the name shown and masked
# against the mean human rating.
examples <- tibble(
  item_id = c("lemens_tweet_226", "lemens_tweet_183"),
  text = c(
    "Republican: \"...Democrats' failed economic policies are hurting you.\"",
    "Democrat: \"...Kevin McCarthy once again playing the part of criminal defense counsel...\""
  )
)
rows <- c("Human coders", "AI models", "AI models, names masked")
ex <- tweet_items %>%
  inner_join(examples, by = "item_id") %>%
  select(text, model, human_position_mean, score_orig, score_mask)
stopifnot(nrow(ex) == 8)
ex_long <- bind_rows(
  ex %>% distinct(text, score = human_position_mean) %>% mutate(row = rows[1]),
  ex %>% transmute(text, model, score = score_orig, row = rows[2]),
  ex %>% transmute(text, model, score = score_mask, row = rows[3])
) %>%
  mutate(row = factor(row, levels = rev(rows)), text = factor(text, levels = examples$text))
ex_mean <- ex_long %>% group_by(text, row) %>% summarise(score = mean(score), .groups = "drop")

fig_example <- ggplot(ex_long, aes(score, row, color = row)) +
  geom_vline(data = ex_mean %>% filter(row == rows[1]),
             aes(xintercept = score), color = "grey80", linewidth = 0.5) +
  geom_point(data = ex_mean, size = 4.2, shape = 18) +
  geom_text(data = ex_mean, aes(label = round(score)), vjust = -1, size = 3.8,
            fontface = "bold", family = "Helvetica") +
  facet_wrap(~ text, ncol = 1) +
  scale_color_manual(values = setNames(c("black", DARK, GREY), rows)) +
  scale_x_continuous(limits = c(0, 100), breaks = c(0, 25, 50, 75, 100),
                     labels = c("0\nleft", "25", "50", "75", "100\nright")) +
  scale_y_discrete(expand = expansion(add = 0.7)) +
  labs(
    x = "Left-right position of the tweet (0-100)", y = NULL,
    caption = "AI models: mean of four commercial models. Human coders: mean rating (Le Mens and Gallego)."
  ) +
  theme_clara_twitter() +
  theme(strip.text = element_text(face = "plain", size = 12, margin = margin(t = 8, b = 4)))
save_fig(fig_example, "02_example_tweets", width = 9, height = 5.2)

# Figure 3: Conservative-minus-Labour error gap in passages that name a party or
# politician, as written and with names masked.
gap <- benoit %>%
  filter(domain == "economic", passage == "named") %>%
  select(model, con_lab_error_gap_orig, gap_orig_lo, gap_orig_hi,
         con_lab_error_gap_mask, gap_mask_lo, gap_mask_hi)
gap_long <- bind_rows(
  gap %>% transmute(model, version = "Names shown", est = con_lab_error_gap_orig,
                    lo = gap_orig_lo, hi = gap_orig_hi),
  gap %>% transmute(model, version = "Names masked", est = con_lab_error_gap_mask,
                    lo = gap_mask_lo, hi = gap_mask_hi)
) %>%
  mutate(
    model = factor(model, levels = model_order),
    version = factor(version, levels = c("Names masked", "Names shown")),
    y = as.numeric(model) + if_else(version == "Names shown", 0.15, -0.15)
  )
labels_1 <- gap_long %>% filter(model == first(model_order))

fig1 <- ggplot(gap_long, aes(est, y, color = version)) +
  geom_vline(xintercept = 0, color = "grey60", linewidth = 0.5) +
  geom_segment(aes(x = lo, xend = hi, yend = y), linewidth = 0.8) +
  geom_point(size = 3.2) +
  geom_text(
    data = labels_1, aes(x = hi + 0.03, label = version), hjust = 0, size = 4,
    fontface = "bold", family = "Helvetica"
  ) +
  scale_color_manual(values = c("Names shown" = DARK, "Names masked" = GREY)) +
  scale_y_continuous(breaks = seq_along(model_order), labels = model_order,
                     expand = expansion(add = 0.6)) +
  labs(
    title = "Party names pull Labour and Conservative codes apart",
    subtitle = paste0(
      "How much further right the model codes Conservative than Labour manifesto sentences\n",
      "that experts placed at the same economic position (steps on a 5-point scale)"
    ),
    x = "Conservative minus Labour (scale steps)", y = NULL,
    caption = paste0(
      "192 of 500 sampled sentences whose passage names a party or politician. Masked: names replaced by [PARTY] or [POLITICIAN],\n",
      "everything else identical. Experts saw the same passages, names included, and for about 80% of their codes also\n",
      "the manifesto title (Benoit et al. 2016). Gap net of the expert code and expert mean. 95% intervals, bootstrap over manifestos."
    )
  ) +
  theme_clara_twitter()
save_fig(fig1, "03_party_gap_named", width = 9, height = 7.6)

# Figure 4: the instruction to ignore names next to masking. Named tweets only:
# extra Republican-Democratic distance against the human raters, as written,
# with the instruction, and with names masked.
ign_path <- file.path(input_dir, "ignore_instruction_tweets.csv")
if (file.exists(ign_path)) {
  conds <- c(orig = "As written", ignore = "Told to ignore names", mask = "Names masked")
  ign <- read_csv(ign_path, show_col_types = FALSE) %>%
    filter(passage == "named", group %in% c("commercial", "flagship"))
  ign_long <- bind_rows(lapply(names(conds), function(v) {
    ign %>% transmute(model, version = conds[[v]],
                      est = .data[[paste0("rep_dem_error_gap_", v)]],
                      lo = .data[[paste0("lo_", v)]], hi = .data[[paste0("hi_", v)]])
  }))
  ign_order <- ign_long %>% filter(version == "As written") %>% arrange(est) %>% pull(model)
  ign_long <- ign_long %>%
    mutate(
      model = factor(model, levels = rev(ign_order)),
      version = factor(version, levels = conds),
      y = as.numeric(model) + c(0.25, 0, -0.25)[as.integer(version)]
    )
  fig_ign <- ggplot(ign_long, aes(est, y, color = version)) +
    geom_vline(xintercept = 0, color = "grey60", linewidth = 0.5) +
    geom_segment(aes(x = lo, xend = hi, yend = y), linewidth = 0.8) +
    geom_point(size = 2.9) +
    scale_color_manual(values = setNames(c(DARK, "#d08a2e", GREY), conds)) +
    geom_text(data = tibble(x = c(0, 0), hjust = c(1.05, -0.05),
                            label = c("\u2190 less apart", "more apart \u2192")),
              aes(x = x, y = -Inf, label = label, hjust = hjust), inherit.aes = FALSE,
              vjust = -0.7, size = 3.8, color = "grey35", family = "Helvetica") +
    scale_y_continuous(breaks = seq_along(ign_order), labels = rev(ign_order),
                       expand = expansion(add = c(1, 0.5))) +
    labs(
      x = "Distance between the parties, model minus human coders (points)",
      y = NULL,
      caption = "Commercial models, 144 tweets that name a party or politician. 0-100 left-right scale. 95% intervals."
    ) +
    theme_clara_twitter() +
    theme(
      legend.position = "top", legend.justification = "left", legend.title = element_blank(),
      legend.text = element_text(size = 12), legend.margin = margin(0, 0, 0, 0),
      axis.title.x = element_text(size = 11.5)
    )
  n_models <- length(ign_order)
  save_fig(fig_ign, "04_ignore_instruction", width = 9.5, height = 2.2 + 0.75 * n_models)
}

# Figure 5: how much the names push Republican and Democratic tweets apart in each
# model (as written minus masked; the negative of rep_minus_dem_change), named
# tweets only.
push <- tweets %>%
  filter(passage == "named") %>%
  transmute(model, push = -rep_minus_dem_change, push_lo = -hi, push_hi = -lo) %>%
  arrange(push) %>%
  mutate(model = factor(model, levels = model))
stopifnot(nrow(push) == 11, all(push$push_lo > 0), all(push$push_hi > push$push))

fig_push <- ggplot(push, aes(x = push, y = model)) +
  geom_vline(xintercept = 0, color = "grey60") +
  geom_errorbar(aes(xmin = push_lo, xmax = push_hi), width = 0, linewidth = 0.9, orientation = "y", color = DARK) +
  geom_point(size = 3.2, color = DARK) +
  annotate("text", x = Inf, y = -Inf, label = "more different \u2192", hjust = 1.05, vjust = -0.7,
           size = 3.8, color = "grey35", family = "Helvetica") +
  scale_x_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.04))) +
  scale_y_discrete(expand = expansion(add = c(1.2, 0.6))) +
  labs(
    title = "What happens if we hide or show names in political statements?",
    x = "How much showing names pushes the parties apart (points)",
    y = NULL,
    caption = "144 tweets that name a party or politician, rated as written and with the names masked. 0-100 left-right scale. 95% intervals."
  ) +
  theme_clara_twitter()

save_fig(fig_push, "05_names_push_apart", width = 8.5, height = 5.6)

# Figure 6: telling the model which party wrote a manifesto sentence ("Source:
# Conservative Party manifesto, 1992" above the passage), next to expert coders
# who coded the same sentences with and without the title. Common set of 1,033
# sentences coded by all 11 models; colour by the model-minus-experts interval.
title_path <- file.path(input_dir, "title_effects.csv")
if (file.exists(title_path)) {
  RED <- "#b5482f"
  ttl <- read_csv(title_path, show_col_types = FALSE) %>%
    filter(sentences == "common") %>%
    mutate(
      vs = case_when(coder == "Experts" ~ "Expert coders",
                     minus_experts_lo > 0 ~ "More than experts",
                     minus_experts_hi < 0 ~ "Less than experts",
                     TRUE ~ "About the same as experts"),
      coder = if_else(coder == "Experts", "Expert coders", coder)
    ) %>%
    arrange(party_effect) %>%
    mutate(coder = factor(coder, levels = coder))
  stopifnot(nrow(ttl) == 12, n_distinct(ttl$n) == 1)
  expert <- ttl %>% filter(coder == "Expert coders")
  fig_title <- ggplot(ttl, aes(party_effect, coder, color = vs)) +
    annotate("rect", xmin = expert$lo, xmax = expert$hi, ymin = -Inf, ymax = Inf,
             fill = "grey92") +
    geom_vline(xintercept = expert$party_effect, color = "grey55", linetype = "dashed") +
    geom_vline(xintercept = 0, color = "grey60") +
    geom_errorbar(aes(xmin = lo, xmax = hi), width = 0, linewidth = 0.9, orientation = "y") +
    geom_point(size = 3.2) +
    scale_color_manual(values = c("Expert coders" = "black", "More than experts" = RED,
                                  "Less than experts" = DARK,
                                  "About the same as experts" = GREY),
                       breaks = c("More than experts", "About the same as experts",
                                  "Less than experts")) +
    annotate("text", x = Inf, y = -Inf, label = "more different \u2192", hjust = 1.05, vjust = -0.7,
             size = 3.8, color = "grey35", family = "Helvetica") +
    scale_y_discrete(expand = expansion(add = c(1.2, 0.6))) +
    labs(
      x = "How much showing the party pushes the parties apart (points)",
      y = NULL,
      caption = paste0(
        format(expert$n, big.mark = ","), " economic-policy sentences from 12 UK manifestos (Benoit et al. 2016), 5-point left-right scale.\n",
        "Colour: compared with experts, 95% interval."
      )
    ) +
    theme_clara_twitter() +
    theme(legend.position = "top", legend.justification = "left", legend.title = element_blank(),
          legend.text = element_text(size = 11.5), legend.margin = margin(0, 0, 0, 0))
  save_fig(fig_title, "06_party_label_vs_experts", width = 9, height = 5.8)
}
