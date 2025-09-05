# Reverb Data Visualization Dashboard - FIXED VERSION
# Install required packages if not already installed
required_packages <- c("ggplot2", "dplyr", "readr", "plotly", "viridis", 
                       "lubridate", "scales", "gridExtra", "RColorBrewer",
                       "ggridges", "patchwork", "DT")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Custom EUR formatting function
euro_format <- function() {
  function(x) paste0("€", format(x, big.mark = ",", scientific = FALSE))
}

# Load and prepare data
# Update this path to your CSV file location
data <- read_csv("data/reverb/cleaned/reverb_tableau_202509-051553.csv")

# Data preprocessing
data <- data %>%
  mutate(
    # Convert dates
    created_at = as.POSIXct(created_at),
    published_at = as.POSIXct(published_at),
    snap_valid_from = as.POSIXct(snap_valid_from),
    snap_valid_to = as.POSIXct(snap_valid_to),
    
    # Calculate price difference
    price_diff = buyer_price - price,
    price_diff_pct = (price_diff / price) * 100,
    
    # Create date components for calendar heatmap
    created_date = as.Date(created_at),
    closed_date = as.Date(snap_valid_to),
    
    # Clean up categories and product types
    product_type = gsub("-", " ", product_type),
    product_type = tools::toTitleCase(product_type),
    category = gsub("-", " ", category),
    category = tools::toTitleCase(category)
  ) %>%
  # Remove extreme outliers for better visualization
  filter(
    price > 0 & price < quantile(price, 0.99, na.rm = TRUE),
    buyer_price > 0 & buyer_price < quantile(buyer_price, 0.99, na.rm = TRUE)
  )

# 1. VIOLIN PLOT OF PRICES BY CATEGORY AND PRODUCT TYPE
create_price_violin <- function(selected_categories = NULL, selected_product_types = NULL) {
  
  plot_data <- data
  
  # Apply filters if specified
  if (!is.null(selected_categories)) {
    plot_data <- plot_data %>% filter(category %in% selected_categories)
  }
  if (!is.null(selected_product_types)) {
    plot_data <- plot_data %>% filter(product_type %in% selected_product_types)
  }
  
  # Create violin plot
  p <- ggplot(plot_data, aes(x = product_type, y = price, fill = product_type)) +
    geom_violin(alpha = 0.7, draw_quantiles = c(0.25, 0.5, 0.75)) +
    geom_boxplot(width = 0.1, alpha = 0.3, outlier.alpha = 0.1) +
    scale_y_log10(labels = euro_format()) +
    scale_fill_viridis_d() +
    facet_wrap(~category, scales = "free_x") +
    labs(
      title = "Price Distribution by Product Type and Category",
      subtitle = "Violin plots with quartiles and box plots overlay",
      x = "Product Type",
      y = "Price (EUR, log scale)",
      fill = "Product Type"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none",
      strip.text = element_text(size = 10, face = "bold")
    )
  
  return(p)
}

# 2. PRICE DIFFERENCE ANALYSIS
create_price_diff_plots <- function() {
  
  # Distribution of absolute price difference
  p1 <- ggplot(data, aes(x = price_diff)) +
    geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
    geom_vline(xintercept = median(data$price_diff, na.rm = TRUE), 
               color = "red", linetype = "dashed", linewidth = 1) +
    labs(
      title = "Distribution of Price Difference (Buyer Price - Price)",
      x = "Price Difference (EUR)",
      y = "Count"
    ) +
    scale_x_continuous(labels = euro_format()) +
    theme_minimal()
  
  # Distribution of percentage price difference
  p2 <- ggplot(data, aes(x = price_diff_pct)) +
    geom_histogram(bins = 50, fill = "darkgreen", alpha = 0.7) +
    geom_vline(xintercept = median(data$price_diff_pct, na.rm = TRUE), 
               color = "red", linetype = "dashed", linewidth = 1) +
    labs(
      title = "Distribution of Price Difference (%)",
      x = "Price Difference (%)",
      y = "Count"
    ) +
    scale_x_continuous(labels = label_percent(scale = 1)) +
    theme_minimal()
  
  # Price difference by category
  p3 <- ggplot(data, aes(x = category, y = price_diff_pct, fill = category)) +
    geom_boxplot(alpha = 0.7) +
    scale_fill_viridis_d() +
    labs(
      title = "Price Difference by Category",
      x = "Category",
      y = "Price Difference (%)"
    ) +
    scale_y_continuous(labels = label_percent(scale = 1)) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none"
    )
  
  return(list(absolute = p1, percentage = p2, by_category = p3))
}

# 3. ITEM ROTATION (PERMANENCE) ANALYSIS
create_permanence_plots <- function() {
  
  # Filter for items with permanence data
  perm_data <- data %>%
    filter(!is.na(permanence_hours) & permanence_hours > 0) %>%
    mutate(
      permanence_days = permanence_hours / 24,
      permanence_category = case_when(
        permanence_days < 1 ~ "< 1 day",
        permanence_days < 7 ~ "1-7 days",
        permanence_days < 30 ~ "1-4 weeks",
        permanence_days < 90 ~ "1-3 months",
        TRUE ~ "3+ months"
      )
    )
  
  # Distribution of permanence
  p1 <- ggplot(perm_data, aes(x = permanence_days)) +
    geom_histogram(bins = 50, fill = "orange", alpha = 0.7) +
    scale_x_log10() +
    labs(
      title = "Distribution of Item Permanence",
      subtitle = "How long items stay active on the platform",
      x = "Permanence (days, log scale)",
      y = "Count"
    ) +
    theme_minimal()
  
  # Permanence by product type
  p2 <- ggplot(perm_data, aes(x = product_type, y = permanence_days, fill = product_type)) +
    geom_violin(alpha = 0.7) +
    geom_boxplot(width = 0.1, alpha = 0.3) +
    scale_y_log10() +
    scale_fill_viridis_d() +
    labs(
      title = "Item Permanence by Product Type",
      x = "Product Type",
      y = "Permanence (days, log scale)"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none"
    )
  
  # Permanence categories
  p3 <- perm_data %>%
    count(permanence_category) %>%
    mutate(
      permanence_category = factor(permanence_category, 
                                   levels = c("< 1 day", "1-7 days", "1-4 weeks", 
                                              "1-3 months", "3+ months"))
    ) %>%
    ggplot(aes(x = permanence_category, y = n, fill = permanence_category)) +
    geom_col(alpha = 0.8) +
    scale_fill_viridis_d() +
    labs(
      title = "Item Permanence Categories",
      x = "Permanence Category",
      y = "Number of Items"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none"
    )
  
  return(list(distribution = p1, by_type = p2, categories = p3))
}

# 4. CALENDAR HEATMAP
create_calendar_heatmap <- function(date_type = "created") {
  
  # Prepare data based on date type
  if (date_type == "created") {
    cal_data <- data %>%
      filter(!is.na(created_date)) %>%
      count(created_date, name = "count") %>%
      rename(date = created_date)
    title_suffix <- "Listing Creations"
  } else {
    cal_data <- data %>%
      filter(!is.na(closed_date)) %>%
      count(closed_date, name = "count") %>%
      rename(date = closed_date)
    title_suffix <- "Listing Closures"
  }
  
  # Create calendar data structure
  cal_data <- cal_data %>%
    mutate(
      year = year(date),
      month = month(date),
      day = day(date),
      week = week(date),
      wday = wday(date, label = TRUE),
      mday = mday(date)
    ) %>%
    complete(date = seq.Date(min(date), max(date), by = "day")) %>%
    replace_na(list(count = 0)) %>%
    mutate(
      year = year(date),
      month = month(date, label = TRUE),
      day = day(date),
      week = week(date),
      wday = wday(date, label = TRUE),
      mday = mday(date)
    )
  
  # Create heatmap
  p <- ggplot(cal_data, aes(x = wday, y = -week, fill = count)) +
    geom_tile(color = "white", linewidth = 0.1) +
    scale_fill_viridis_c(name = "Count", trans = "sqrt") +
    facet_wrap(~month, scales = "free") +
    labs(
      title = paste("Calendar Heatmap:", title_suffix),
      subtitle = "Daily activity patterns",
      x = "Day of Week",
      y = NULL
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid = element_blank(),
      strip.text = element_text(size = 10, face = "bold")
    )
  
  return(p)
}

# INTERACTIVE FUNCTIONS
# Function to create interactive filters
create_interactive_dashboard <- function() {
  
  # Get unique values for filters
  categories <- sort(unique(data$category))
  product_types <- sort(unique(data$product_type))
  
  cat("Available Categories:\n")
  cat(paste(1:length(categories), categories, sep = ". ", collapse = "\n"))
  cat("\n\nAvailable Product Types:\n")
  cat(paste(1:length(product_types), product_types, sep = ". ", collapse = "\n"))
  
  return(list(categories = categories, product_types = product_types))
}

# MAIN EXECUTION
cat("Reverb Data Visualization Dashboard\n")
cat("=====================================\n\n")

# Create filter options
filters <- create_interactive_dashboard()

cat("\n\n--- GENERATING VISUALIZATIONS ---\n")

# 1. Price violin plots (show all by default)
cat("1. Creating price violin plots...\n")
price_violin <- create_price_violin()
print(price_violin)

# 2. Price difference analysis
cat("2. Creating price difference analysis...\n")
price_diff_plots <- create_price_diff_plots()
grid.arrange(price_diff_plots$absolute, price_diff_plots$percentage, ncol = 2)
print(price_diff_plots$by_category)

# 3. Permanence analysis
cat("3. Creating permanence analysis...\n")
perm_plots <- create_permanence_plots()
print(perm_plots$distribution)
print(perm_plots$by_type)
print(perm_plots$categories)

# 4. Calendar heatmaps
cat("4. Creating calendar heatmaps...\n")
cal_created <- create_calendar_heatmap("created")
cal_closed <- create_calendar_heatmap("closed")
print(cal_created)
print(cal_closed)

# SUMMARY STATISTICS
cat("\n--- SUMMARY STATISTICS ---\n")
cat("Total records:", nrow(data), "\n")
cat("Date range:", as.character(min(data$created_date, na.rm = TRUE)), 
    "to", as.character(max(data$created_date, na.rm = TRUE)), "\n")
cat("Unique categories:", length(unique(data$category)), "\n")
cat("Unique product types:", length(unique(data$product_type)), "\n")
cat("Price range:", euro_format()(min(data$price, na.rm = TRUE)), 
    "to", euro_format()(max(data$price, na.rm = TRUE)), "\n")

# EXAMPLE: Filter for specific categories/product types
# Uncomment and modify these lines to filter visualizations:
# 
# selected_cats <- c("Electric Guitars", "Bass Guitars")
# selected_types <- c("Electric Guitars", "Bass Guitars") 
# filtered_violin <- create_price_violin(selected_cats, selected_types)
# print(filtered_violin)