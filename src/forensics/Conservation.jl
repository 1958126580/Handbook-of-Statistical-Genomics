# ============================================================================
# Conservation.jl - Conservation Genetics
# ============================================================================

"""
    inbreeding_coefficient(gm::GenotypeMatrix)

Calculate individual inbreeding coefficients.
"""
function inbreeding_coefficient_individual(gm::GenotypeMatrix)
    n_samp = n_samples(gm)
    F = Vector{Float64}(undef, n_samp)
    
    for i in 1:n_samp
        obs_het = 0
        exp_het = 0.0
        n_valid = 0
        
        for j in 1:n_variants(gm)
            g = gm.data[i, j]
            if ismissing(g)
                continue
            end
            
            # Get allele frequency
            genos = collect(skipmissing(gm.data[:, j]))
            if isempty(genos)
                continue
            end
            
            p = sum(genos) / (2 * length(genos))
            if p == 0 || p == 1
                continue
            end
            
            n_valid += 1
            obs_het += g == 1 ? 1 : 0
            exp_het += 2 * p * (1 - p)
        end
        
        if exp_het > 0
            F[i] = 1 - obs_het / exp_het
        else
            F[i] = NaN
        end
    end
    
    return F
end

"""
    population_viability(Ne::Float64, generations::Int; 
                        lambda::Float64=1.01, var_lambda::Float64=0.1)

Simple population viability analysis.
"""
function population_viability(Ne::Float64, generations::Int;
                             lambda::Float64=1.01, var_lambda::Float64=0.1,
                             n_simulations::Int=1000)
    extinction_count = 0
    final_sizes = Float64[]
    
    for sim in 1:n_simulations
        N = Ne
        for gen in 1:generations
            # Stochastic growth
            r = rand(Normal(log(lambda), var_lambda))
            N *= exp(r)
            
            # Demographic stochasticity for small populations
            if N < 50
                N = rand(Poisson(N))
            end
            
            if N < 2
                extinction_count += 1
                break
            end
        end
        push!(final_sizes, N)
    end
    
    extinction_prob = extinction_count / n_simulations
    
    return (extinction_probability=extinction_prob,
            mean_final_size=mean(final_sizes),
            median_final_size=median(final_sizes))
end
