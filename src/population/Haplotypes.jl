# ============================================================================
# Haplotypes.jl - Haplotype Estimation and Phasing
# ============================================================================

"""
    HaplotypeResult

Container for haplotype estimation results.
"""
struct HaplotypeResult
    haplotypes::Matrix{Int8}      # Estimated haplotypes (2*n_samples × n_variants)
    frequencies::Vector{Float64}  # Estimated haplotype frequencies
    unique_haplotypes::Matrix{Int8}  # Unique haplotype patterns
    posterior::Matrix{Float64}    # Posterior probabilities of haplotype pairs
    log_likelihood::Float64       # Final log-likelihood
    converged::Bool               # Whether EM converged
end

"""
    estimate_haplotypes(gm::GenotypeMatrix; method::Symbol=:em, 
                       maxiter::Int=100, tol::Float64=1e-6)

Estimate haplotypes from genotype data.

# Arguments
- `gm`: GenotypeMatrix
- `method`: Estimation method (:em for Expectation-Maximization)
- `maxiter`: Maximum iterations
- `tol`: Convergence tolerance

# Returns
- HaplotypeResult with estimated haplotypes
"""
function estimate_haplotypes(gm::GenotypeMatrix; method::Symbol=:em,
                            maxiter::Int=100, tol::Float64=1e-6)
    if method == :em
        return em_haplotype_estimation(gm, maxiter, tol)
    else
        throw(ArgumentError("Unknown method: $method"))
    end
end

"""
    em_haplotype_estimation(gm::GenotypeMatrix, maxiter::Int, tol::Float64)

EM algorithm for haplotype frequency estimation.
Based on Excoffier & Slatkin (1995) approach.
"""
function em_haplotype_estimation(gm::GenotypeMatrix, maxiter::Int, tol::Float64)
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    
    # For computational tractability, limit to small regions
    if n_var > 20
        @warn "Many variants ($n_var), consider windowed approach"
    end
    
    # Generate all possible haplotypes
    n_possible = 2^n_var
    if n_possible > 10000
        # Use sampling-based approach for large haplotype space
        return sampled_haplotype_estimation(gm, maxiter, tol)
    end
    
    # Generate haplotype matrix
    all_haps = Matrix{Int8}(undef, n_possible, n_var)
    for h in 0:(n_possible-1)
        for j in 1:n_var
            all_haps[h+1, j] = (h >> (j-1)) & 1
        end
    end
    
    # Initialize frequencies uniformly
    hap_freq = fill(1.0 / n_possible, n_possible)
    
    # EM iterations
    prev_ll = -Inf
    converged = false
    
    for iter in 1:maxiter
        # E-step: Calculate expected haplotype pair frequencies for each individual
        expected_counts = zeros(n_possible)
        log_likelihood = 0.0
        
        for i in 1:n_samp
            # Get genotypes for this individual
            geno = collect(gm.data[i, :])
            
            # Find compatible haplotype pairs
            compatible_pairs = Tuple{Int, Int, Float64}[]  # (h1, h2, probability)
            
            for h1 in 1:n_possible
                for h2 in h1:n_possible
                    if is_compatible(all_haps[h1, :], all_haps[h2, :], geno)
                        # Probability of this pair
                        if h1 == h2
                            prob = hap_freq[h1]^2
                        else
                            prob = 2 * hap_freq[h1] * hap_freq[h2]
                        end
                        push!(compatible_pairs, (h1, h2, prob))
                    end
                end
            end
            
            # Normalize probabilities
            total_prob = sum(p[3] for p in compatible_pairs)
            
            if total_prob > 0
                log_likelihood += log(total_prob)
                
                for (h1, h2, prob) in compatible_pairs
                    weight = prob / total_prob
                    expected_counts[h1] += weight
                    expected_counts[h2] += weight
                end
            end
        end
        
        # M-step: Update haplotype frequencies
        total_chroms = 2 * n_samp
        hap_freq = expected_counts ./ total_chroms
        
        # Add small regularization to avoid zero frequencies
        hap_freq .+= 1e-10
        hap_freq ./= sum(hap_freq)
        
        # Check convergence
        if abs(log_likelihood - prev_ll) < tol
            converged = true
            break
        end
        prev_ll = log_likelihood
    end
    
    # Extract most likely haplotypes for each individual
    estimated_haps = Matrix{Int8}(undef, 2 * n_samp, n_var)
    posteriors = zeros(n_samp, n_possible * n_possible)
    
    for i in 1:n_samp
        geno = collect(gm.data[i, :])
        best_h1, best_h2 = 1, 1
        best_prob = 0.0
        
        for h1 in 1:n_possible
            for h2 in h1:n_possible
                if is_compatible(all_haps[h1, :], all_haps[h2, :], geno)
                    prob = h1 == h2 ? hap_freq[h1]^2 : 2*hap_freq[h1]*hap_freq[h2]
                    if prob > best_prob
                        best_prob = prob
                        best_h1, best_h2 = h1, h2
                    end
                end
            end
        end
        
        estimated_haps[2*i-1, :] = all_haps[best_h1, :]
        estimated_haps[2*i, :] = all_haps[best_h2, :]
    end
    
    # Get unique haplotypes with non-negligible frequency
    sig_haps = findall(f -> f > 0.01, hap_freq)
    
    HaplotypeResult(
        estimated_haps,
        hap_freq,
        all_haps[sig_haps, :],
        posteriors[:, 1:min(100, size(posteriors, 2))],
        prev_ll,
        converged
    )
end

"""Check if two haplotypes are compatible with observed genotype."""
function is_compatible(hap1::AbstractVector, hap2::AbstractVector, 
                      genotype::AbstractVector)
    for j in eachindex(genotype)
        if ismissing(genotype[j])
            continue  # Missing genotypes are compatible with anything
        end
        expected_geno = hap1[j] + hap2[j]
        if expected_geno != genotype[j]
            return false
        end
    end
    return true
end

"""Sampling-based haplotype estimation for large variant sets."""
function sampled_haplotype_estimation(gm::GenotypeMatrix, maxiter::Int, tol::Float64)
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    
    # Initialize with pseudo-random phasing
    haplotypes = Matrix{Int8}(undef, 2 * n_samp, n_var)
    
    for i in 1:n_samp
        for j in 1:n_var
            g = gm.data[i, j]
            if ismissing(g)
                haplotypes[2*i-1, j] = 0
                haplotypes[2*i, j] = 0
            elseif g == 0
                haplotypes[2*i-1, j] = 0
                haplotypes[2*i, j] = 0
            elseif g == 2
                haplotypes[2*i-1, j] = 1
                haplotypes[2*i, j] = 1
            else  # g == 1, heterozygous
                # Random initial phase
                if rand() < 0.5
                    haplotypes[2*i-1, j] = 0
                    haplotypes[2*i, j] = 1
                else
                    haplotypes[2*i-1, j] = 1
                    haplotypes[2*i, j] = 0
                end
            end
        end
    end
    
    # Gibbs sampling to refine phases
    for iter in 1:maxiter
        for i in 1:n_samp
            for j in 1:n_var
                g = gm.data[i, j]
                if ismissing(g) || g != 1
                    continue  # Only need to phase heterozygotes
                end
                
                # Calculate probability of each phase given neighbors
                # Simplified: use local haplotype frequencies
                
                # Current configuration
                h1_current = copy(haplotypes[2*i-1, :])
                h2_current = copy(haplotypes[2*i, :])
                
                # Alternative configuration
                h1_alt = copy(h1_current)
                h2_alt = copy(h2_current)
                h1_alt[j], h2_alt[j] = h2_alt[j], h1_alt[j]
                
                # Count matching haplotypes in population
                match_current = count_haplotype_matches(haplotypes, h1_current, 2*i-1) +
                               count_haplotype_matches(haplotypes, h2_current, 2*i)
                match_alt = count_haplotype_matches(haplotypes, h1_alt, 2*i-1) +
                           count_haplotype_matches(haplotypes, h2_alt, 2*i)
                
                # Update with probability proportional to matches
                prob_alt = match_alt / (match_current + match_alt + 1e-10)
                
                if rand() < prob_alt
                    haplotypes[2*i-1, :] = h1_alt
                    haplotypes[2*i, :] = h2_alt
                end
            end
        end
    end
    
    # Estimate haplotype frequencies
    hap_strings = [join(haplotypes[h, :]) for h in 1:(2*n_samp)]
    unique_haps = unique(hap_strings)
    freq_dict = Dict{String, Float64}()
    for hs in hap_strings
        freq_dict[hs] = get(freq_dict, hs, 0.0) + 1.0
    end
    
    freqs = [freq_dict[h] / (2*n_samp) for h in unique_haps]
    unique_hap_matrix = Matrix{Int8}(undef, length(unique_haps), n_var)
    for (i, hs) in enumerate(unique_haps)
        for (j, c) in enumerate(hs)
            unique_hap_matrix[i, j] = parse(Int8, c)
        end
    end
    
    HaplotypeResult(
        haplotypes,
        freqs,
        unique_hap_matrix,
        zeros(n_samp, 1),  # Simplified posterior
        0.0,
        true
    )
end

"""Count how many haplotypes in the population match the query."""
function count_haplotype_matches(all_haps::Matrix{Int8}, query::Vector{Int8}, 
                                exclude_idx::Int)
    count = 0
    for h in 1:size(all_haps, 1)
        if h == exclude_idx
            continue
        end
        if all_haps[h, :] == query
            count += 1
        end
    end
    return count
end

"""
    phase_genotypes(gm::GenotypeMatrix; window_size::Int=100, overlap::Int=20)

Phase genotypes using windowed approach for chromosome-length data.

# Arguments
- `gm`: GenotypeMatrix
- `window_size`: Number of variants per window
- `overlap`: Overlap between adjacent windows

# Returns
- Matrix of phased haplotypes (2*n_samples × n_variants)
"""
function phase_genotypes(gm::GenotypeMatrix; window_size::Int=100, overlap::Int=20)
    n_samp = n_samples(gm)
    n_var = n_variants(gm)
    
    phased = Matrix{Int8}(undef, 2 * n_samp, n_var)
    
    # Process each chromosome separately
    for chr in unique(gm.chromosomes)
        chr_idx = findall(c -> c == chr, gm.chromosomes)
        sort!(chr_idx, by=j -> gm.positions[j])
        
        # Phase in windows
        window_start = 1
        while window_start <= length(chr_idx)
            window_end = min(window_start + window_size - 1, length(chr_idx))
            window_vars = chr_idx[window_start:window_end]
            
            # Create sub-matrix for this window
            window_data = gm.data[:, window_vars]
            window_gm = GenotypeMatrix(window_data)
            
            # Phase this window
            result = estimate_haplotypes(window_gm; maxiter=50)
            
            # Copy results (respecting overlap with previous window)
            if window_start == 1
                copy_start = 1
            else
                copy_start = overlap ÷ 2 + 1
            end
            
            for (local_j, global_j) in enumerate(window_vars[copy_start:end])
                phased[:, global_j] = result.haplotypes[:, copy_start + local_j - 1]
            end
            
            window_start += window_size - overlap
        end
    end
    
    return phased
end

"""
    haplotype_frequencies(phased_haps::Matrix{Int8})

Calculate haplotype frequencies from phased data.

# Returns
- DataFrame with haplotype strings and frequencies
"""
function haplotype_frequencies(phased_haps::Matrix{Int8})
    n_haps = size(phased_haps, 1)
    
    # Convert to strings
    hap_strings = [join(phased_haps[h, :]) for h in 1:n_haps]
    
    # Count frequencies
    counts = Dict{String, Int}()
    for hs in hap_strings
        counts[hs] = get(counts, hs, 0) + 1
    end
    
    # Create result
    unique_haps = collect(keys(counts))
    freqs = [counts[h] / n_haps for h in unique_haps]
    
    # Sort by frequency
    order = sortperm(freqs, rev=true)
    
    DataFrame(
        haplotype = unique_haps[order],
        count = [counts[h] for h in unique_haps[order]],
        frequency = freqs[order]
    )
end
