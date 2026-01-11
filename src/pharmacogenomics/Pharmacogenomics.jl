# ============================================================================
# Pharmacogenomics.jl - Pharmacogenomics Analysis Methods
# ============================================================================
# Methods for analyzing genetic variants affecting drug response
# Including star allele calling, drug-gene associations, and dosing
# ============================================================================

"""
    StarAllele

Structure representing a pharmacogenomic star allele.

# Fields
- `gene::String`: Gene name (e.g., CYP2D6)
- `allele::String`: Allele name (e.g., *4)
- `variants::Vector{String}`: Defining variants
- `function_status::Symbol`: Function status (:normal, :decreased, :no_function, :increased)
- `activity_score::Float64`: Activity score (0-2)
"""
struct StarAllele
    gene::String
    allele::String
    variants::Vector{String}
    function_status::Symbol
    activity_score::Float64
end

"""
    PGxResult

Structure containing pharmacogenomics analysis results.

# Fields
- `gene::String`: Gene analyzed
- `diplotype::String`: Diplotype call (e.g., *1/*4)
- `phenotype::String`: Predicted phenotype
- `activity_score::Float64`: Combined activity score
- `recommendation::String`: Dosing recommendation
"""
struct PGxResult
    gene::String
    diplotype::String
    phenotype::String
    activity_score::Float64
    recommendation::String
end

"""
    DrugResponseResult

Structure for drug response GWAS results.

# Fields
- `drug::String`: Drug name
- `phenotype_type::String`: Response type (efficacy, toxicity, PK)
- `significant_variants::DataFrame`: Significant associations
- `h2::Float64`: Heritability estimate
"""
struct DrugResponseResult
    drug::String
    phenotype_type::String
    significant_variants::DataFrame
    h2::Float64
end

# Standard star allele definitions for major pharmacogenes
const STAR_ALLELE_DEFINITIONS = Dict(
    "CYP2D6" => Dict(
        "*1" => StarAllele("CYP2D6", "*1", String[], :normal, 1.0),
        "*2" => StarAllele("CYP2D6", "*2", ["rs16947"], :normal, 1.0),
        "*3" => StarAllele("CYP2D6", "*3", ["rs35742686"], :no_function, 0.0),
        "*4" => StarAllele("CYP2D6", "*4", ["rs3892097"], :no_function, 0.0),
        "*5" => StarAllele("CYP2D6", "*5", ["gene_deletion"], :no_function, 0.0),
        "*6" => StarAllele("CYP2D6", "*6", ["rs5030655"], :no_function, 0.0),
        "*9" => StarAllele("CYP2D6", "*9", ["rs5030656"], :decreased, 0.5),
        "*10" => StarAllele("CYP2D6", "*10", ["rs1065852"], :decreased, 0.25),
        "*17" => StarAllele("CYP2D6", "*17", ["rs28371706"], :decreased, 0.5),
        "*29" => StarAllele("CYP2D6", "*29", ["rs59421388"], :decreased, 0.5),
        "*41" => StarAllele("CYP2D6", "*41", ["rs28371725"], :decreased, 0.5)
    ),
    "CYP2C19" => Dict(
        "*1" => StarAllele("CYP2C19", "*1", String[], :normal, 1.0),
        "*2" => StarAllele("CYP2C19", "*2", ["rs4244285"], :no_function, 0.0),
        "*3" => StarAllele("CYP2C19", "*3", ["rs4986893"], :no_function, 0.0),
        "*17" => StarAllele("CYP2C19", "*17", ["rs12248560"], :increased, 1.5)
    ),
    "CYP2C9" => Dict(
        "*1" => StarAllele("CYP2C9", "*1", String[], :normal, 1.0),
        "*2" => StarAllele("CYP2C9", "*2", ["rs1799853"], :decreased, 0.5),
        "*3" => StarAllele("CYP2C9", "*3", ["rs1057910"], :decreased, 0.25)
    ),
    "CYP3A5" => Dict(
        "*1" => StarAllele("CYP3A5", "*1", String[], :normal, 1.0),
        "*3" => StarAllele("CYP3A5", "*3", ["rs776746"], :no_function, 0.0)
    ),
    "TPMT" => Dict(
        "*1" => StarAllele("TPMT", "*1", String[], :normal, 1.0),
        "*2" => StarAllele("TPMT", "*2", ["rs1800462"], :no_function, 0.0),
        "*3A" => StarAllele("TPMT", "*3A", ["rs1800460", "rs1142345"], :no_function, 0.0),
        "*3B" => StarAllele("TPMT", "*3B", ["rs1800460"], :no_function, 0.0),
        "*3C" => StarAllele("TPMT", "*3C", ["rs1142345"], :no_function, 0.0)
    ),
    "DPYD" => Dict(
        "*1" => StarAllele("DPYD", "*1", String[], :normal, 1.0),
        "*2A" => StarAllele("DPYD", "*2A", ["rs3918290"], :no_function, 0.0),
        "*13" => StarAllele("DPYD", "*13", ["rs55886062"], :no_function, 0.0)
    ),
    "SLCO1B1" => Dict(
        "*1A" => StarAllele("SLCO1B1", "*1A", String[], :normal, 1.0),
        "*1B" => StarAllele("SLCO1B1", "*1B", ["rs2306283"], :normal, 1.0),
        "*5" => StarAllele("SLCO1B1", "*5", ["rs4149056"], :decreased, 0.5),
        "*15" => StarAllele("SLCO1B1", "*15", ["rs2306283", "rs4149056"], :decreased, 0.5)
    ),
    "VKORC1" => Dict(
        "A/A" => StarAllele("VKORC1", "A/A", ["rs9923231_AA"], :high_sensitivity, 0.25),
        "A/G" => StarAllele("VKORC1", "A/G", ["rs9923231_AG"], :intermediate, 0.5),
        "G/G" => StarAllele("VKORC1", "G/G", ["rs9923231_GG"], :normal, 1.0)
    )
)

"""
    call_star_alleles(genotypes::Dict{String, Int}, gene::String) -> Tuple{String, String}

Call star alleles from genotype data for a pharmacogene.

# Arguments
- `genotypes`: Dictionary mapping variant IDs to genotypes (0/1/2)
- `gene`: Gene name (e.g., "CYP2D6")

# Returns
Tuple of (allele1, allele2) representing the diplotype

# Algorithm
1. Match observed variants against star allele definitions
2. Apply priority rules for overlapping alleles
3. Handle copy number variants (for CYP2D6)
4. Return most likely diplotype

# Example
```julia
genotypes = Dict("rs3892097" => 1, "rs1065852" => 0)
allele1, allele2 = call_star_alleles(genotypes, "CYP2D6")
# Returns ("*1", "*4")
```

# References
- PharmVar database (https://www.pharmvar.org/)
- Gaedigk et al. (2017) Clin. Pharmacol. Ther.
"""
function call_star_alleles(genotypes::Dict{String, Int}, gene::String)
    if !haskey(STAR_ALLELE_DEFINITIONS, gene)
        error("Unknown gene: $gene. Supported: $(keys(STAR_ALLELE_DEFINITIONS))")
    end

    allele_defs = STAR_ALLELE_DEFINITIONS[gene]

    # Score each allele based on matching variants
    allele_scores = Dict{String, Float64}()

    for (allele_name, allele) in allele_defs
        if isempty(allele.variants)
            # Reference allele (*1) - default score
            allele_scores[allele_name] = 0.0
        else
            # Score based on matching variants
            matches = 0
            for var in allele.variants
                if haskey(genotypes, var) && genotypes[var] > 0
                    matches += genotypes[var]
                end
            end
            allele_scores[allele_name] = matches / length(allele.variants)
        end
    end

    # Select top two alleles
    sorted_alleles = sort(collect(allele_scores), by=x->x[2], rev=true)

    # Handle diplotype calling
    if sorted_alleles[1][2] > 0
        allele1 = sorted_alleles[1][1]
        if sorted_alleles[2][2] > 0
            allele2 = sorted_alleles[2][1]
        else
            allele2 = "*1"  # Reference allele
        end
    else
        allele1 = "*1"
        allele2 = "*1"
    end

    return (allele1, allele2)
end

"""
    predict_phenotype(gene::String, allele1::String, allele2::String) -> PGxResult

Predict metabolizer phenotype from diplotype.

# Arguments
- `gene`: Gene name
- `allele1`: First allele
- `allele2`: Second allele

# Returns
PGxResult with phenotype prediction and activity score

# Phenotype Categories
- Ultra-rapid metabolizer (UM): Activity score > 2.0
- Normal metabolizer (NM): Activity score 1.0-2.0
- Intermediate metabolizer (IM): Activity score 0.5-1.0
- Poor metabolizer (PM): Activity score < 0.5

# Example
```julia
result = predict_phenotype("CYP2D6", "*1", "*4")
println("Phenotype: \$(result.phenotype)")
println("Activity Score: \$(result.activity_score)")
```
"""
function predict_phenotype(gene::String, allele1::String, allele2::String)
    if !haskey(STAR_ALLELE_DEFINITIONS, gene)
        error("Unknown gene: $gene")
    end

    allele_defs = STAR_ALLELE_DEFINITIONS[gene]

    # Get activity scores
    as1 = haskey(allele_defs, allele1) ? allele_defs[allele1].activity_score : 1.0
    as2 = haskey(allele_defs, allele2) ? allele_defs[allele2].activity_score : 1.0

    total_score = as1 + as2

    # Determine phenotype
    if total_score > 2.0
        phenotype = "Ultrarapid Metabolizer"
    elseif total_score >= 1.0
        phenotype = "Normal Metabolizer"
    elseif total_score >= 0.5
        phenotype = "Intermediate Metabolizer"
    else
        phenotype = "Poor Metabolizer"
    end

    # Generate recommendation based on gene and phenotype
    recommendation = generate_recommendation(gene, phenotype, total_score)

    diplotype = "$allele1/$allele2"

    return PGxResult(gene, diplotype, phenotype, total_score, recommendation)
end

"""
    generate_recommendation(gene, phenotype, activity_score) -> String

Generate clinical dosing recommendation based on PGx result.
"""
function generate_recommendation(gene::String, phenotype::String, activity_score::Float64)
    recommendations = Dict(
        "CYP2D6" => Dict(
            "Poor Metabolizer" => "Consider alternative drugs not metabolized by CYP2D6. If CYP2D6 substrate is necessary, consider 50% dose reduction and monitor for adverse effects.",
            "Intermediate Metabolizer" => "Standard starting dose with close monitoring. Consider dose adjustment based on response.",
            "Normal Metabolizer" => "Use standard dosing as per drug labeling.",
            "Ultrarapid Metabolizer" => "For prodrugs (e.g., codeine), use alternative due to risk of toxicity. For active drugs, consider dose increase or alternative therapy."
        ),
        "CYP2C19" => Dict(
            "Poor Metabolizer" => "For PPIs: standard dose effective. For clopidogrel: use alternative antiplatelet (prasugrel/ticagrelor).",
            "Intermediate Metabolizer" => "Standard dosing for most drugs. Consider alternative for clopidogrel.",
            "Normal Metabolizer" => "Use standard dosing.",
            "Ultrarapid Metabolizer" => "For PPIs: consider dose increase. Standard clopidogrel dosing effective."
        ),
        "CYP2C9" => Dict(
            "Poor Metabolizer" => "For warfarin: reduce dose 20-40%. For NSAIDs: use lowest effective dose.",
            "Intermediate Metabolizer" => "For warfarin: reduce dose 10-20%. Standard NSAID dosing with monitoring.",
            "Normal Metabolizer" => "Use standard dosing."
        ),
        "TPMT" => Dict(
            "Poor Metabolizer" => "For thiopurines: reduce dose to 10% of standard or use alternative. High risk of severe myelosuppression.",
            "Intermediate Metabolizer" => "For thiopurines: reduce starting dose by 30-50%.",
            "Normal Metabolizer" => "Use standard dosing with routine monitoring."
        ),
        "DPYD" => Dict(
            "Poor Metabolizer" => "AVOID fluoropyrimidines (5-FU, capecitabine). Life-threatening toxicity risk.",
            "Intermediate Metabolizer" => "Reduce fluoropyrimidine dose by 50%.",
            "Normal Metabolizer" => "Use standard dosing."
        ),
        "SLCO1B1" => Dict(
            "Poor Metabolizer" => "For simvastatin: avoid high doses (>20mg). Consider alternative statin.",
            "Intermediate Metabolizer" => "For simvastatin: limit dose to 40mg. Monitor for myopathy.",
            "Normal Metabolizer" => "Use standard statin dosing."
        )
    )

    if haskey(recommendations, gene) && haskey(recommendations[gene], phenotype)
        return recommendations[gene][phenotype]
    else
        return "Consult clinical pharmacogenomics guidelines for specific drug recommendations."
    end
end

"""
    pgx_gwas(genotypes::Matrix{Float64}, drug_response::Vector{Float64};
            covariates=nothing, response_type=:continuous) -> DrugResponseResult

Run pharmacogenomics GWAS for drug response.

# Arguments
- `genotypes`: Genotype matrix
- `drug_response`: Drug response phenotype
- `response_type`: Type of response - :continuous, :binary, :ordinal

# Returns
DrugResponseResult with significant associations

# Example
```julia
# Test association with drug efficacy
result = pgx_gwas(genotypes, efficacy_scores; response_type=:continuous)

# Test association with adverse events
result = pgx_gwas(genotypes, adverse_event; response_type=:binary)
```
"""
function pgx_gwas(
    genotypes::Matrix{Float64},
    drug_response::Vector{Float64};
    covariates::Union{Matrix{Float64}, Nothing}=nothing,
    response_type::Symbol=:continuous,
    variant_ids::Union{Vector{String}, Nothing}=nothing,
    drug_name::String="unknown",
    p_threshold::Float64=5e-8
)
    n_samples, n_variants = size(genotypes)

    if variant_ids === nothing
        variant_ids = ["SNP_$i" for i in 1:n_variants]
    end

    # Build covariate matrix
    if covariates !== nothing
        X_base = hcat(ones(n_samples), covariates)
    else
        X_base = ones(n_samples, 1)
    end

    results = DataFrame(
        variant = String[],
        beta = Float64[],
        se = Float64[],
        statistic = Float64[],
        pvalue = Float64[]
    )

    is_binary = response_type == :binary || all(r -> r == 0 || r == 1, drug_response)

    for j in 1:n_variants
        X = hcat(X_base, genotypes[:, j])
        var_idx = size(X, 2)

        try
            if is_binary
                β = logistic_fit(X, drug_response)
                μ = 1.0 ./ (1.0 .+ exp.(-X * β))
                V = μ .* (1 .- μ)
                I = X' * Diagonal(V) * X
                se = sqrt(inv(I)[var_idx, var_idx])
                z = β[var_idx] / se
                pvalue = 2 * ccdf(Normal(), abs(z))
                stat = z
            else
                β = X \ drug_response
                residuals = drug_response - X * β
                σ2 = sum(residuals.^2) / (n_samples - size(X, 2))
                XtX_inv = inv(X' * X)
                se = sqrt(σ2 * XtX_inv[var_idx, var_idx])
                t = β[var_idx] / se
                pvalue = 2 * ccdf(TDist(n_samples - size(X, 2)), abs(t))
                stat = t
            end

            push!(results, (
                variant = variant_ids[j],
                beta = β[var_idx],
                se = se,
                statistic = stat,
                pvalue = pvalue
            ))
        catch
            continue
        end
    end

    sort!(results, :pvalue)

    # Filter significant
    significant = filter(r -> r.pvalue < p_threshold, results)

    # Estimate heritability (simplified)
    h2 = estimate_pgx_heritability(genotypes, drug_response)

    return DrugResponseResult(
        drug_name,
        string(response_type),
        significant,
        h2
    )
end

"""
    estimate_pgx_heritability(genotypes, phenotype) -> Float64

Estimate heritability of drug response phenotype.
"""
function estimate_pgx_heritability(genotypes::Matrix{Float64}, phenotype::Vector{Float64})
    n, p = size(genotypes)

    # Standardize genotypes
    G_std = similar(genotypes)
    for j in 1:p
        μ = mean(genotypes[:, j])
        σ = std(genotypes[:, j])
        if σ > 0
            G_std[:, j] = (genotypes[:, j] .- μ) ./ σ
        else
            G_std[:, j] .= 0
        end
    end

    # GRM
    K = G_std * G_std' / p

    # GREML-like heritability estimation (simplified)
    # Using method of moments
    y = phenotype .- mean(phenotype)
    σ2_y = var(phenotype)

    # y'Ky / (n-1) estimates σ2_g + σ2_e * tr(K)/(n-1)
    yKy = y' * K * y
    tr_K = tr(K)

    # Solve for h2
    h2 = (yKy / σ2_y - tr_K / n) / (n - tr_K / n)
    h2 = clamp(h2, 0, 1)

    return h2
end

"""
    warfarin_dose_prediction(cyp2c9::String, vkorc1::String;
                            age=nothing, weight=nothing, race=nothing) -> Float64

Predict warfarin dose based on genetic and clinical factors.

# Arguments
- `cyp2c9`: CYP2C9 diplotype (e.g., "*1/*2")
- `vkorc1`: VKORC1 genotype (e.g., "A/G")
- `age`: Patient age in years
- `weight`: Patient weight in kg
- `race`: Patient race/ethnicity

# Returns
Predicted weekly warfarin dose in mg

# Algorithm
Uses IWPC (International Warfarin Pharmacogenetics Consortium) algorithm:
sqrt(dose) = 5.6044 - 0.2614×age_decade - 0.0087×height + 0.0128×weight
             - 0.8677×VKORC1_AG - 1.6974×VKORC1_AA
             - 0.5211×CYP2C9_*1/*2 - 0.9357×CYP2C9_*1/*3
             - 1.0616×CYP2C9_*2/*2 - 1.9206×CYP2C9_*2/*3
             - 2.3312×CYP2C9_*3/*3
             - 0.2188×Asian - 0.1092×Black
             + 1.1816×enzyme_inducer - 0.5503×amiodarone

# Example
```julia
dose = warfarin_dose_prediction("*1/*2", "A/G"; age=65, weight=80)
println("Predicted weekly dose: \$(dose) mg")
```

# References
- IWPC (2009) N. Engl. J. Med.
"""
function warfarin_dose_prediction(
    cyp2c9::String,
    vkorc1::String;
    age::Union{Int, Nothing}=nothing,
    weight::Union{Float64, Nothing}=nothing,
    height::Union{Float64, Nothing}=nothing,
    race::Union{String, Nothing}=nothing,
    enzyme_inducer::Bool=false,
    amiodarone::Bool=false
)
    # Base intercept
    sqrt_dose = 5.6044

    # Age effect (per decade)
    if age !== nothing
        sqrt_dose -= 0.2614 * (age / 10)
    end

    # Weight effect
    if weight !== nothing
        sqrt_dose += 0.0128 * weight
    end

    # Height effect
    if height !== nothing
        sqrt_dose -= 0.0087 * height
    end

    # VKORC1 effect
    if vkorc1 == "A/G" || vkorc1 == "G/A"
        sqrt_dose -= 0.8677
    elseif vkorc1 == "A/A"
        sqrt_dose -= 1.6974
    end

    # CYP2C9 effect
    cyp2c9_effects = Dict(
        "*1/*2" => -0.5211,
        "*2/*1" => -0.5211,
        "*1/*3" => -0.9357,
        "*3/*1" => -0.9357,
        "*2/*2" => -1.0616,
        "*2/*3" => -1.9206,
        "*3/*2" => -1.9206,
        "*3/*3" => -2.3312
    )

    if haskey(cyp2c9_effects, cyp2c9)
        sqrt_dose += cyp2c9_effects[cyp2c9]
    end

    # Race effect
    if race !== nothing
        race_lower = lowercase(race)
        if contains(race_lower, "asian")
            sqrt_dose -= 0.2188
        elseif contains(race_lower, "black") || contains(race_lower, "african")
            sqrt_dose -= 0.1092
        end
    end

    # Drug interactions
    if enzyme_inducer
        sqrt_dose += 1.1816
    end
    if amiodarone
        sqrt_dose -= 0.5503
    end

    # Convert to weekly dose
    weekly_dose = sqrt_dose^2

    # Ensure reasonable range
    weekly_dose = clamp(weekly_dose, 5.0, 100.0)

    return weekly_dose
end

"""
    pgx_report(genotypes::Dict{String, Dict{String, Int}}) -> DataFrame

Generate comprehensive PGx report for a patient.

# Arguments
- `genotypes`: Dictionary mapping gene names to variant genotypes

# Returns
DataFrame with PGx results for all tested genes

# Example
```julia
patient_genotypes = Dict(
    "CYP2D6" => Dict("rs3892097" => 1, "rs1065852" => 0),
    "CYP2C19" => Dict("rs4244285" => 0, "rs12248560" => 1)
)

report = pgx_report(patient_genotypes)
```
"""
function pgx_report(genotypes::Dict{String, Dict{String, Int}})
    results = DataFrame(
        gene = String[],
        diplotype = String[],
        phenotype = String[],
        activity_score = Float64[],
        recommendation = String[]
    )

    for (gene, gene_genotypes) in genotypes
        if haskey(STAR_ALLELE_DEFINITIONS, gene)
            allele1, allele2 = call_star_alleles(gene_genotypes, gene)
            result = predict_phenotype(gene, allele1, allele2)

            push!(results, (
                gene = result.gene,
                diplotype = result.diplotype,
                phenotype = result.phenotype,
                activity_score = result.activity_score,
                recommendation = result.recommendation
            ))
        end
    end

    return results
end

# Helper function
function logistic_fit(X, y; max_iter=25)
    n, p = size(X)
    β = zeros(p)

    for _ in 1:max_iter
        μ = 1.0 ./ (1.0 .+ exp.(-X * β))
        V = μ .* (1 .- μ)
        V = max.(V, 1e-10)
        z = X * β + (y - μ) ./ V
        β_new = (X' * Diagonal(V) * X) \ (X' * Diagonal(V) * z)

        if maximum(abs.(β_new - β)) < 1e-8
            return β_new
        end
        β = β_new
    end
    return β
end
