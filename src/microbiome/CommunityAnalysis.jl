# ============================================================================
# CommunityAnalysis.jl - Microbial Community Analysis
# ============================================================================

"""
    community_profile(abundance::Matrix{Float64}, taxa_names::Vector{String})

Summarize microbial community composition.
"""
function community_profile(abundance::Matrix{Float64}, taxa_names::Vector{String})
    n_taxa, n_samples = size(abundance)
    
    # Relative abundance
    rel_abundance = abundance ./ sum(abundance, dims=1)
    
    # Mean abundance per taxon
    mean_abundance = mean(rel_abundance, dims=2)[:]
    prevalence = [count(x -> x > 0, abundance[t, :]) / n_samples for t in 1:n_taxa]
    
    order = sortperm(mean_abundance, rev=true)
    
    return (taxa_names=taxa_names[order], mean_abundance=mean_abundance[order], 
            prevalence=prevalence[order], relative_abundance=rel_abundance)
end

"""
    taxonomic_abundance(otu_table::Matrix{Float64}, taxonomy::Dict{String, String})

Aggregate OTU abundance to taxonomic levels.
"""
function taxonomic_abundance(otu_table::Matrix{Float64}, 
                            taxonomy::Dict{String, String},
                            otu_names::Vector{String},
                            level::Symbol=:genus)
    aggregated = Dict{String, Vector{Float64}}()
    
    for (i, otu) in enumerate(otu_names)
        if haskey(taxonomy, otu)
            taxon = taxonomy[otu]
            if !haskey(aggregated, taxon)
                aggregated[taxon] = zeros(size(otu_table, 2))
            end
            aggregated[taxon] .+= otu_table[i, :]
        end
    end
    
    return aggregated
end
