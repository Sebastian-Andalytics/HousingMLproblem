# ================================================
# 0. Setup & Load Libraries
# ================================================
library(tidyverse)      
library(corrplot)       
library(randomForest)   
library(caret)          
library(ranger)         
library(cluster)        
library(factoextra)     
library(purrr)
library(dplyr)
library(stargazer)
library(gridExtra)
library(ggplot2)
library(scales)
library(factoextra)
# ================================================
# 1. Data Import & Feature Engineering
# ================================================
energy <- read.csv("/Users/sebastian/Downloads/ENB2012_data.csv") %>%
  mutate(
    Orientation  = factor(Orientation),
    GlazingDist  = factor(Glazing.Area.Distribution),
    glazing_frac = Glazing.Area / Surface.Area,
    wall_frac    = Wall.Area    / Surface.Area
  )

## ------------------------------------------------
## 2. Exploratory Visualisations  (6‑up per page)
## ------------------------------------------------
num_vars <- energy |>
  select(-ID) |>                
  select(where(is.numeric)) |>  # keep only numeric columns
  names()


##  Display six per “page” in the Plots pane --------------------------
## ---------- 2.2 Histograms ----------
hist_list <- lapply(num_vars, function(v) {
  ggplot(energy, aes(.data[[v]])) +
    geom_histogram(bins = 30,
                   fill = "steelblue", colour = "white") +
    labs(title = paste("Histogram of", v), x = v, y = "Frequency") +
    theme_minimal(base_size = 11)
})
marrangeGrob(hist_list, nrow = 2, ncol = 2)

## ---------- 2.3 Box‑plots ----------

box_list <- lapply(num_vars, function(v) {
  ggplot(energy, aes(y = .data[[v]])) +
    geom_boxplot(fill = "lightgreen") +
    labs(title = paste("Boxplot of", v), y = v) +
    theme_minimal(base_size = 11)
})

marrangeGrob(box_list, nrow = 2, ncol = 3)

   

ggplot(energy, aes(factor(wall_frac))) +
  geom_bar(fill = "steelblue") +
  scale_x_discrete(
    labels = function(x) prettyNum(round(as.numeric(x), 3))  
  ) +
  labs(title = "Counts of distinct wall_frac values",
       x = "wall_frac (rounded to 3 d.p.)",
       y = "Frequency") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 2.4 Correlation Matrix (all numeric vars)
num_df <- energy |>
  select(where(is.numeric), -ID)

# 2. compute pairwise‑complete correlation matrix
M <- cor(num_df, use = "pairwise.complete.obs")

# 3. draw the plot directly in the Plots pane
corrplot(
  M,
  method      = "color",
  type        = "upper",
  addCoef.col = "black",
  tl.col      = "black",
  tl.srt      = 45,
  title       = "Correlation Matrix: All Numeric Features",
  mar         = c(0, 0, 1, 0)   # tighten margins so the title shows
)

# ================================================
# 3. Baseline: Simple Linear Regression
# ================================================
lm_baseline <- lm(
  Y1 ~ Relative.Compactness + Surface.Area + Wall.Area + Roof.Area +
    Overall.Height + Glazing.Area + glazing_frac + wall_frac +
    Orientation + GlazingDist,
  data = energy
)
summary(lm_baseline)

# ================================================
# 4. Random Forest on Full Dataset
# ================================================
# 1. Fit RF 
set.seed(123)
rf_full <- randomForest(
  Y1 ~ Relative.Compactness + Surface.Area + Wall.Area + Roof.Area +
    Overall.Height + Glazing.Area + glazing_frac + wall_frac +
    Orientation + GlazingDist,
  data       = energy,
  ntree      = 500,
  importance = TRUE
)

# 2. Extract importance from the model object
imp_mat <- rf_full$importance
imp_df  <- data.frame(
  Variable      = rownames(imp_mat),
  IncMSE        = imp_mat[, "%IncMSE"],
  IncNodePurity = imp_mat[, "IncNodePurity"],
  row.names     = NULL
)

# 3. Display with stargazer in console
library(stargazer)
stargazer(
  imp_df,
  type     = "html",
  summary  = FALSE,
  rownames = FALSE,
  digits   = 1,
  title    = "RF Full‐Model Variable Importance"
)


# 5. Print OOB MSE
cat("Full RF OOB MSE:", round(tail(rf_full$mse, 1), 3), "\n")
# ================================================
# 5. PCA on Geometry Features
# ================================================
geom_feats  <- energy %>%
  select(Relative.Compactness, Surface.Area, Wall.Area, Roof.Area, Overall.Height)
geom_scaled <- scale(geom_feats)

pca_res <- prcomp(geom_scaled, center = TRUE, scale. = FALSE)
fviz_eig(pca_res, addlabels = TRUE, ylim = c(0, 60)) +
  labs(title = "PCA: Percentage of Variance Explained")

pcs <- as.data.frame(pca_res$x[, 1:2]) %>%
  setNames(c("PC1", "PC2"))

# ================================================
# 6. Choose K via elbow plot
# ================================================
set.seed(123)
fviz_nbclust(pcs, FUNcluster = kmeans, method = "wss", k.max = 6) +
  labs(title = "Elbow Method: Optimal # of Clusters")

# ================================================
# 7. k‑Means Clustering on PCs
# ================================================
set.seed(123)
km_pca <- kmeans(pcs, centers = 3, nstart = 50)
energy <- energy %>%
  mutate(cluster = factor(km_pca$cluster))

# Visualize clusters on PC1/PC2
fviz_cluster(
  km_pca,
  data    = pcs,
  geom    = "point",
  ellipse = TRUE,
  palette = "jco",
  ggtheme = theme_minimal(),
  main    = "k‑Means on First Two Principal Components"
)

# Cluster profiling
cluster_profile <- energy %>%
  group_by(cluster) %>%
  summarise(
    size              = n(),
    mean_Y1           = mean(Y1),
    mean_Y2           = mean(Y2),
    mean_glazing_frac = mean(glazing_frac),
    mean_compactness  = mean(Relative.Compactness),
    mean_surface      = mean(Surface.Area),
    mean_wall         = mean(Wall.Area),
    mean_roof         = mean(Roof.Area),
    mean_height       = mean(Overall.Height)
  )
print(cluster_profile)

# ================================================
# 8. Cluster‑Specific Random Forests & CV Grid Search
# ================================================
# 8.1 Define a caret CV spec (5‑fold)
ctrl <- trainControl(
  method       = "cv",
  number       = 5,
  verboseIter  = FALSE,
  returnResamp = "all"
)

# 8.2 Build an expanded tuning grid
p <- 6  # number of predictors in cluster‐level model
rf_grid <- expand.grid(
  mtry          = seq(1, p, by = 1),             
  splitrule     = c("variance", "extratrees"),   
  min.node.size = c(1, 5, 10, 20, 30)            
)

cluster_models_cv <- list()
for(k in levels(energy$cluster)) {
  df_k <- energy %>% filter(cluster == k)
  cat("\n>>> Cluster", k, "(n =", nrow(df_k), ")\n")
  
  set.seed(123)
  rf_k_cv <- train(
    Y1 ~ Relative.Compactness + glazing_frac + wall_frac +
      Glazing.Area + Surface.Area + Overall.Height,
    data       = df_k,
    method     = "ranger",
    trControl  = ctrl,
    tuneGrid   = rf_grid,
    num.trees  = 500,
    importance = "permutation",
    metric     = "RMSE"
  )
  
  # store
  cluster_models_cv[[k]] <- rf_k_cv
  print(rf_k_cv)
  
  # plot variable importance
  plot(varImp(rf_k_cv), main = paste("Cluster", k, "— Variable Importance"))
  final_rmse <- min(rf_k_cv$results$RMSE)
    cat("Cluster", k, "- Final RMSE:", round(final_rmse, 3), "\n")
}
# ================================================
# 9. Summarize & stargazer the CV results
# ================================================


cluster_models <- list()

for (k in levels(energy$cluster)) {
  df_k <- filter(energy, cluster == k)
  message("Training RF for cluster ", k, " (n=", nrow(df_k), ")")
  
  rf_k <- randomForest(
    Y1 ~ Relative.Compactness + glazing_frac + wall_frac + Glazing.Area +
      Surface.Area + Overall.Height,
    data       = df_k,
    ntree      = 500,
    importance = TRUE
  )
  cluster_models[[k]] <- rf_k
  
  # 1) pull out the importance matrix
  imp_mat <- rf_k$importance
  imp_df  <- as.data.frame(imp_mat) %>% 
    rownames_to_column(var = "Variable") %>%
    arrange(desc(`%IncMSE`))  
  
  # 2) print top 5 by %IncMSE to console
  print(head(imp_df, 5))
  
  # 3) use varImpPlot for a quick visual
  varImpPlot(rf_k, main = paste("Cluster", k, "– Variable Importance"))
  
  # 4) stargazer() for visualization:
  cat("\nCluster", k, "Variable Importance:\n")
  stargazer(
    imp_df,
    type     = "html",
    summary  = FALSE,
    rownames = FALSE,
    title    = paste("Cluster", k, "Variable Importance"),
    digits   = 1
  )
}
# ================================================
# End of Script
# ================================================

