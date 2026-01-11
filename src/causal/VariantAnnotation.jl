# ============================================================================
# VariantAnnotation.jl - Variant Functional Annotation
# ============================================================================

"""
    variant_impact(variant::VariantInfo)

Predict functional impact of a variant.
"""
function variant_impact(variant::VariantInfo)
    impacts = Dict{Symbol, Float64}()
    
    # Basic classification
    if is_snp(variant)
        if is_transition(variant)
            impacts[:transition] = 1.0
        else
            impacts[:transversion] = 1.0
        end
    elseif is_indel(variant)
        indel_size = abs(length(variant.ref) - length(variant.alt))
        impacts[:frameshift] = (indel_size % 3 != 0) ? 1.0 : 0.0
        impacts[:indel_size] = Float64(indel_size)
    end
    
    return impacts
end

"""
    conservation_score(position::Int, conservation_track::Dict{Int, Float64})

Get conservation score for a genomic position.
"""
function conservation_score(position::Int, conservation_track::Dict{Int, Float64})
    return get(conservation_track, position, NaN)
end

"""
    regulatory_annotation(position::Int, regulatory_regions::Vector{Tuple{Int, Int, String}})

Annotate variants with regulatory elements.
"""
function regulatory_annotation(position::Int, 
                              regulatory_regions::Vector{Tuple{Int, Int, String}})
    annotations = String[]
    
    for (start, stop, label) in regulatory_regions
        if start <= position <= stop
            push!(annotations, label)
        end
    end
    
    return annotations
end
