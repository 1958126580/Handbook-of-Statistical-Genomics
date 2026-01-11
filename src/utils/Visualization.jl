# ============================================================================
# Visualization.jl - Plotting Functions for Genomic Data
# ============================================================================

"""
    manhattan_plot(chromosomes::AbstractVector, positions::AbstractVector, 
                  pvalues::AbstractVector; kwargs...)

Create a Manhattan plot for GWAS results.

# Arguments
- `chromosomes`: Chromosome identifiers
- `positions`: Genomic positions
- `pvalues`: P-values from association tests

# Keyword Arguments
- `significance_threshold`: Genome-wide significance threshold (default: 5e-8)
- `suggestive_threshold`: Suggestive significance threshold (default: 1e-5)
- `title`: Plot title
- `highlight`: Indices of variants to highlight
"""
function manhattan_plot(chromosomes::AbstractVector, positions::AbstractVector,
                       pvalues::AbstractVector;
                       significance_threshold::Float64=5e-8,
                       suggestive_threshold::Float64=1e-5,
                       title::String="Manhattan Plot",
                       highlight::Vector{Int}=Int[])
    
    n = length(pvalues)
    @assert length(chromosomes) == n && length(positions) == n
    
    # Convert to -log10(p)
    logp = -log10.(max.(pvalues, 1e-300))
    
    # Calculate cumulative positions for x-axis
    chr_numeric = [c isa Int ? c : findfirst(==(c), unique(chromosomes)) for c in chromosomes]
    unique_chrs = sort(unique(chr_numeric))
    
    chr_offsets = Dict{Int, Int64}()
    cumulative = Int64(0)
    chr_centers = Dict{Int, Float64}()
    
    for chr in unique_chrs
        chr_mask = chr_numeric .== chr
        chr_positions = positions[chr_mask]
        chr_offsets[chr] = cumulative
        chr_centers[chr] = cumulative + (maximum(chr_positions) - minimum(chr_positions)) / 2
        cumulative += maximum(chr_positions) + 1_000_000
    end
    
    x_positions = [positions[i] + chr_offsets[chr_numeric[i]] for i in 1:n]
    
    # Create color palette (alternating by chromosome)
    colors = [isodd(chr_numeric[i]) ? :steelblue : :darkorange for i in 1:n]
    
    # Highlight selected variants
    for idx in highlight
        colors[idx] = :red
    end
    
    # Create plot
    p = scatter(x_positions, logp, 
               color=colors, 
               markersize=2, 
               markerstrokewidth=0,
               xlabel="Chromosome",
               ylabel="-log₁₀(p)",
               title=title,
               legend=false,
               size=(1200, 400))
    
    # Add significance lines
    hline!(p, [-log10(significance_threshold)], color=:red, linestyle=:dash, linewidth=1)
    hline!(p, [-log10(suggestive_threshold)], color=:blue, linestyle=:dot, linewidth=1)
    
    # Add chromosome labels
    xticks = [chr_centers[chr] for chr in unique_chrs]
    xtick_labels = string.(unique_chrs)
    plot!(p, xticks=(xticks, xtick_labels))
    
    return p
end

"""
    qq_plot(pvalues::AbstractVector; title::String="QQ Plot")

Create a quantile-quantile plot for p-values.

# Arguments
- `pvalues`: Observed p-values

# Returns
- Plot object
"""
function qq_plot(pvalues::AbstractVector; title::String="QQ Plot")
    # Remove missing and invalid p-values
    valid_p = filter(p -> !ismissing(p) && !isnan(p) && p > 0 && p <= 1, pvalues)
    n = length(valid_p)
    
    # Sort observed p-values
    observed = sort(valid_p)
    expected = (1:n) ./ (n + 1)
    
    # Convert to -log10
    obs_log = -log10.(observed)
    exp_log = -log10.(expected)
    
    # Calculate genomic inflation factor (lambda GC)
    lambda_gc = median(obs_log) / median(exp_log)
    
    # Create plot
    max_val = max(maximum(obs_log), maximum(exp_log)) * 1.05
    
    p = scatter(exp_log, obs_log,
               xlabel="Expected -log₁₀(p)",
               ylabel="Observed -log₁₀(p)",
               title="$title\nλ_GC = $(round(lambda_gc, digits=3))",
               legend=false,
               markersize=3,
               color=:steelblue,
               alpha=0.6)
    
    # Add identity line
    plot!(p, [0, max_val], [0, max_val], color=:red, linestyle=:dash, linewidth=1)
    
    return p
end

"""
    pca_plot(pc1::AbstractVector, pc2::AbstractVector;
            groups::Union{AbstractVector, Nothing}=nothing,
            title::String="PCA Plot")

Create a PCA scatter plot.

# Arguments
- `pc1`: First principal component values
- `pc2`: Second principal component values
- `groups`: Optional group labels for coloring
"""
function pca_plot(pc1::AbstractVector, pc2::AbstractVector;
                 groups::Union{AbstractVector, Nothing}=nothing,
                 title::String="PCA Plot",
                 var_explained::Tuple{Float64, Float64}=(0.0, 0.0))
    
    xlabel_str = var_explained[1] > 0 ? "PC1 ($(round(var_explained[1]*100, digits=1))%)" : "PC1"
    ylabel_str = var_explained[2] > 0 ? "PC2 ($(round(var_explained[2]*100, digits=1))%)" : "PC2"
    
    if groups === nothing
        p = scatter(pc1, pc2,
                   xlabel=xlabel_str,
                   ylabel=ylabel_str,
                   title=title,
                   legend=false,
                   markersize=4,
                   color=:steelblue)
    else
        unique_groups = unique(groups)
        colors = distinguishable_colors(length(unique_groups))
        color_map = Dict(g => colors[i] for (i, g) in enumerate(unique_groups))
        point_colors = [color_map[g] for g in groups]
        
        p = scatter(pc1, pc2,
                   xlabel=xlabel_str,
                   ylabel=ylabel_str,
                   title=title,
                   group=groups,
                   markersize=4,
                   legend=:topright)
    end
    
    return p
end

"""Generate distinguishable colors."""
function distinguishable_colors(n::Int)
    if n <= 10
        return [:steelblue, :darkorange, :green, :red, :purple, 
                :brown, :pink, :gray, :olive, :cyan][1:n]
    else
        return [RGB(rand(), rand(), rand()) for _ in 1:n]
    end
end

"""
    heatmap_plot(data::AbstractMatrix; 
                row_labels::Vector{String}=String[],
                col_labels::Vector{String}=String[],
                title::String="Heatmap")

Create a heatmap visualization.
"""
function heatmap_plot(data::AbstractMatrix;
                     row_labels::Vector{String}=String[],
                     col_labels::Vector{String}=String[],
                     title::String="Heatmap",
                     colorscheme::Symbol=:viridis)
    
    m, n = size(data)
    
    if isempty(row_labels)
        row_labels = ["R$i" for i in 1:m]
    end
    if isempty(col_labels)
        col_labels = ["C$j" for j in 1:n]
    end
    
    p = heatmap(data,
               xlabel="",
               ylabel="",
               title=title,
               xticks=(1:n, col_labels),
               yticks=(1:m, row_labels),
               color=colorscheme,
               xrotation=45)
    
    return p
end

"""
    ld_heatmap(ld_matrix::AbstractMatrix; variant_ids::Vector{String}=String[])

Create an LD heatmap visualization.
"""
function ld_heatmap(ld_matrix::AbstractMatrix; 
                   variant_ids::Vector{String}=String[],
                   title::String="Linkage Disequilibrium")
    
    heatmap_plot(ld_matrix, 
                row_labels=variant_ids, 
                col_labels=variant_ids,
                title=title,
                colorscheme=:YlOrRd)
end

"""
    forest_plot(effects::AbstractVector, ses::AbstractVector, 
               labels::Vector{String}; title::String="Forest Plot")

Create a forest plot for effect size visualization.
"""
function forest_plot(effects::AbstractVector, ses::AbstractVector,
                    labels::Vector{String}; 
                    title::String="Forest Plot",
                    ci_level::Float64=0.95)
    
    n = length(effects)
    z = quantile(Normal(), 1 - (1 - ci_level) / 2)
    
    lower = effects .- z .* ses
    upper = effects .+ z .* ses
    
    y_positions = n:-1:1
    
    p = scatter(effects, y_positions,
               xerror=(effects .- lower, upper .- effects),
               yticks=(y_positions, labels),
               xlabel="Effect Size",
               title=title,
               legend=false,
               markersize=6,
               color=:steelblue)
    
    # Add vertical line at zero
    vline!(p, [0], color=:gray, linestyle=:dash)
    
    return p
end
