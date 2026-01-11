# ============================================================================
# Parallel.jl - Parallel Computing Utilities
# ============================================================================

"""
    parallel_map(f::Function, collection; n_threads::Int=Threads.nthreads())

Apply function f to each element of collection in parallel.

# Arguments
- `f`: Function to apply
- `collection`: Iterable collection

# Returns
- Vector of results
"""
function parallel_map(f::Function, collection; n_threads::Int=Threads.nthreads())
    n = length(collection)
    results = Vector{Any}(undef, n)
    
    Threads.@threads for i in 1:n
        results[i] = f(collection[i])
    end
    
    return results
end

"""
    progress_map(f::Function, collection; desc::String="Processing")

Apply function with progress bar.

# Arguments
- `f`: Function to apply
- `collection`: Iterable collection
- `desc`: Description for progress bar

# Returns
- Vector of results
"""
function progress_map(f::Function, collection; desc::String="Processing")
    n = length(collection)
    results = Vector{Any}(undef, n)
    
    @showprogress desc=desc for i in 1:n
        results[i] = f(collection[i])
    end
    
    return results
end

"""
    chunked_parallel(f::Function, collection, chunk_size::Int)

Process collection in parallel chunks.
"""
function chunked_parallel(f::Function, collection, chunk_size::Int)
    n = length(collection)
    n_chunks = ceil(Int, n / chunk_size)
    
    results = Vector{Vector{Any}}(undef, n_chunks)
    
    Threads.@threads for chunk_idx in 1:n_chunks
        start_idx = (chunk_idx - 1) * chunk_size + 1
        end_idx = min(chunk_idx * chunk_size, n)
        chunk = collection[start_idx:end_idx]
        results[chunk_idx] = [f(x) for x in chunk]
    end
    
    return reduce(vcat, results)
end

"""
    timed_execution(f::Function; desc::String="")

Execute function with timing information.
"""
function timed_execution(f::Function; desc::String="")
    start_time = time()
    result = f()
    elapsed = time() - start_time
    
    if !isempty(desc)
        @info "$(desc): $(round(elapsed, digits=2)) seconds"
    end
    
    return (result=result, elapsed=elapsed)
end

"""
    batch_process(f::Function, data::AbstractMatrix; 
                 batch_dim::Int=2, batch_size::Int=1000)

Process matrix data in batches along specified dimension.
"""
function batch_process(f::Function, data::AbstractMatrix;
                      batch_dim::Int=2, batch_size::Int=1000,
                      show_progress::Bool=true)
    
    n = size(data, batch_dim)
    n_batches = ceil(Int, n / batch_size)
    
    results = Vector{Any}(undef, n_batches)
    
    progress = show_progress ? Progress(n_batches, desc="Processing batches: ") : nothing
    
    for batch_idx in 1:n_batches
        start_idx = (batch_idx - 1) * batch_size + 1
        end_idx = min(batch_idx * batch_size, n)
        
        if batch_dim == 1
            batch_data = data[start_idx:end_idx, :]
        else
            batch_data = data[:, start_idx:end_idx]
        end
        
        results[batch_idx] = f(batch_data)
        
        if show_progress
            next!(progress)
        end
    end
    
    return results
end

"""
    memory_efficient_apply(f::Function, gm::GenotypeMatrix; 
                          variant_chunk_size::Int=100)

Apply function to genotype matrix in memory-efficient chunks.
"""
function memory_efficient_apply(f::Function, gm::GenotypeMatrix;
                               variant_chunk_size::Int=100)
    n_vars = n_variants(gm)
    n_chunks = ceil(Int, n_vars / variant_chunk_size)
    
    all_results = []
    
    @showprogress "Processing variants: " for chunk_idx in 1:n_chunks
        start_j = (chunk_idx - 1) * variant_chunk_size + 1
        end_j = min(chunk_idx * variant_chunk_size, n_vars)
        
        chunk_data = gm.data[:, start_j:end_j]
        chunk_result = f(chunk_data)
        push!(all_results, chunk_result)
    end
    
    return all_results
end
