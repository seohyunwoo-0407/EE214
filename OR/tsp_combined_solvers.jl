using JuMP
using Gurobi
using Random
using Printf
using Concorde # Added for Concorde TSP Solver
using LKH # Added for LKH TSP Solver
# using LightGraphs # Not strictly needed for the current find_subtours, but good to note if extending

"""
    generate_tsp_instance(n::Int, max_cost::Int=100)

Generates a random symmetric TSP cost matrix with integer costs.

Arguments:
- n: The number of cities.
- max_cost: The maximum possible cost between any two distinct cities (default 100).

Returns:
- A n x n symmetric matrix of integer costs, where C[i,i] = 0.
"""
function generate_tsp_instance(n::Int, max_cost::Int=100)
    if n <= 0
        error("Number of cities n must be positive.")
    end
    if max_cost <= 0
        error("Maximum cost must be positive.")
    end
    cost_matrix = zeros(Int, n, n)
    for i in 1:n
        for j in (i+1):n
            cost = rand(1:max_cost)
            cost_matrix[i, j] = cost
            cost_matrix[j, i] = cost
        end
    end
    return cost_matrix
end

"""
    solve_tsp_mtz(cost_matrix::Matrix{Int})

Solves the Traveling Salesman Problem using the Miller-Tucker-Zemlin (MTZ) formulation
with JuMP and Gurobi.

Arguments:
- cost_matrix: An n x n matrix where C[i,j] is the cost from city i to city j.

Returns:
- A tuple: (optimal_tour, tour_length, computation_time)
"""
function solve_tsp_mtz(cost_matrix::Matrix{Int})
    n = size(cost_matrix, 1)
    if n == 0
        return [], 0.0, 0.0
    end

    model = Model(Gurobi.Optimizer)
    set_silent(model)

    @variable(model, x[1:n, 1:n], Bin)
    @variable(model, u[1:n], Int)

    @objective(model, Min, sum(cost_matrix[i,j] * x[i,j] for i=1:n, j=1:n if i != j))

    @constraint(model, [i=1:n], sum(x[i,j] for j=1:n if i != j) == 1)
    @constraint(model, [j=1:n], sum(x[i,j] for i=1:n if i != j) == 1)
    @constraint(model, [i=1:n], x[i,i] == 0)
    @constraint(model, u[1] == 1)
    @constraint(model, [i=2:n], u[i] >= 2)
    @constraint(model, [i=2:n], u[i] <= n)
    @constraint(model, [i=1:n, j=2:n; i != j], u[i] - u[j] + n * x[i,j] <= n-1)

    start_time = time()
    optimize!(model)
    computation_time = time() - start_time

    tour_length = Inf
    optimal_tour = []

    if termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.LOCALLY_SOLVED || (termination_status(model) == MOI.TIME_LIMIT && has_values(model))
        tour_length = objective_value(model)
        solution_edges = []
        for i in 1:n
            for j in 1:n
                if i != j && value(x[i,j]) > 0.99
                    push!(solution_edges, (i,j))
                end
            end
        end

        if !isempty(solution_edges)
            curr_city = 1
            push!(optimal_tour, curr_city)
            while length(optimal_tour) < n
                found_next = false
                for (edge_u,edge_v) in solution_edges # Renamed to avoid conflict
                    if edge_u == curr_city && !(edge_v in optimal_tour)
                        push!(optimal_tour, edge_v)
                        curr_city = edge_v
                        found_next = true
                        break
                    end
                end
                if !found_next
                    println("Error: Could not reconstruct full tour from MTZ edges.")
                    optimal_tour = [Symbol("ReconstructionFailedMTZ")]
                    append!(optimal_tour, solution_edges)
                    break 
                end
            end
            if length(optimal_tour) == n
                 push!(optimal_tour, optimal_tour[1])
            elseif !(Symbol("ReconstructionFailedMTZ") in optimal_tour) && !isempty(solution_edges)
                 println("Warning: MTZ Tour reconstruction might be incomplete.")
            end
        else
            println("No solution edges found in MTZ.")
        end
    else
        println("MTZ: Optimal solution not found. Status: ", termination_status(model))
    end
    return optimal_tour, tour_length, computation_time
end

"""
    find_subtours(edges::Vector{Tuple{Int, Int}}, n_nodes::Int)

Finds all subtours in a given set of edges for a graph with n_nodes.
"""
function find_subtours(edges::Vector{Tuple{Int, Int}}, n_nodes::Int)
    if isempty(edges) || n_nodes == 0
        return []
    end
    adj = [[] for _ in 1:n_nodes]
    for (u, v) in edges
        if 1 <= u <= n_nodes && 1 <= v <= n_nodes
            push!(adj[u], v)
            # For DFJ, x_ij is directed, but we build undirected graph for component finding
            # However, the solution from x_ij should form directed paths. Let's assume directed for adj construction initially.
            # If tour reconstruction logic assumes directed edges, this is fine.
            # If a general component finder is used for undirected graph, then push!(adj[v],u) might be needed.
            # For now, sticking to how it might be if edges are from x_ij solution directly.
        end
    end
    # For component finding, it's better to have an undirected representation if edges are just pairs
    # Rebuilding adj for undirected graph for component finding:
    adj = [Set{Int}() for _ in 1:n_nodes]
    for (u,v) in edges
        if 1 <= u <= n_nodes && 1 <= v <= n_nodes
            push!(adj[u], v)
            push!(adj[v], u) # Make it undirected for component search
        end
    end
    adj_list_of_vectors = [collect(s) for s in adj] # Convert Set to Vector for iteration

    visited = falses(n_nodes)
    subtours = Vector{Vector{Int}}()
    for i in 1:n_nodes
        if !visited[i]
            component = Int[]
            q = [i]
            visited[i] = true
            head = 1
            while head <= length(q)
                u_node = q[head] # Renamed to avoid conflict
                head += 1
                push!(component, u_node)
                if u_node <= length(adj_list_of_vectors)
                    for v_node in adj_list_of_vectors[u_node] # Renamed to avoid conflict
                        if 1 <= v_node <= n_nodes && !visited[v_node]
                            visited[v_node] = true
                            push!(q, v_node)
                        end
                    end
                end
            end
            if !isempty(component) && length(component) < n_nodes
                push!(subtours, component)
            elseif !isempty(component) && length(component) == n_nodes && count(visited) == n_nodes
                return [] # Full tour
            end
        end
    end
    return subtours
end

"""
    solve_tsp_dfj(cost_matrix::Matrix{Int})

Solves TSP using Dantzig-Fulkerson-Johnson (DFJ) formulation.
"""
function solve_tsp_dfj(cost_matrix::Matrix{Int})
    n = size(cost_matrix, 1)
    if n == 0
        return [], 0.0, 0.0, 0
    end

    model = Model(Gurobi.Optimizer)
    set_silent(model)

    @variable(model, x[1:n, 1:n], Bin)
    @objective(model, Min, sum(cost_matrix[i,j] * x[i,j] for i=1:n, j=1:n if i != j))
    @constraint(model, [i=1:n], sum(x[i,j] for j=1:n if i != j) == 1)
    @constraint(model, [j=1:n], sum(x[i,j] for i=1:n if i != j) == 1)
    @constraint(model, [i=1:n], x[i,i] == 0)

    start_time = time()
    iterations = 0
    while true
        iterations += 1
        optimize!(model)

        if !(termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
            println("DFJ Error: Model not solved in iteration $iterations. Status: ", termination_status(model))
            return [Symbol("SolverErrorDFJ")], Inf, time() - start_time, iterations
        end

        current_edges = Tuple{Int,Int}[]
        for i in 1:n
            for j in 1:n
                if i != j && value(x[i,j]) > 0.99
                    push!(current_edges, (i,j))
                end
            end
        end
        
        sub_tours = find_subtours(current_edges, n)

        if isempty(sub_tours)
            break 
        end

        for tour_nodes in sub_tours
            if !isempty(tour_nodes) && length(tour_nodes) < n
                @constraint(model, sum(x[i,j] for i in tour_nodes, j in tour_nodes if i!=j) <= length(tour_nodes) - 1)
            end
        end
    end

    computation_time = time() - start_time
    tour_length = objective_value(model)
    optimal_tour = []
    solution_edges = [] # Re-fetch edges for tour reconstruction, just in case.
     for i in 1:n
        for j in 1:n
            if i != j && value(x[i,j]) > 0.99
                push!(solution_edges, (i,j))
            end
        end
    end

    if !isempty(solution_edges)
        curr_city = 1
        push!(optimal_tour, curr_city)
        visited_count = 1
        while visited_count < n
            found_next = false
            # Iterate over a copy or manage indices carefully if modifying solution_edges
            for k_edge in 1:length(solution_edges) 
                edge_u,edge_v = solution_edges[k_edge]
                if edge_u == curr_city && !(edge_v in optimal_tour)
                    push!(optimal_tour, edge_v)
                    curr_city = edge_v
                    visited_count +=1
                    found_next = true
                    break
                end
            end
            if !found_next
                println("Error: Could not reconstruct full tour in DFJ.")
                optimal_tour = [Symbol("ReconstructionFailedDFJ")]
                append!(optimal_tour, solution_edges)
                break
            end
        end
        if length(optimal_tour) == n 
            push!(optimal_tour, optimal_tour[1])
        elseif !(Symbol("ReconstructionFailedDFJ") in optimal_tour) && !isempty(solution_edges)
            println("Warning: DFJ Tour reconstruction might be incomplete.")
        end
    else
        println("No solution edges found in DFJ result.")
    end
    
    return optimal_tour, tour_length, computation_time, iterations
end

"""
    calculate_tour_cost(tour::Vector{Int}, cost_matrix::Matrix{Int})

Calculates the total cost of a given tour.
The tour should be a permutation of cities, e.g., [1, 3, 2, 4] for n=4.
It automatically adds the cost from the last city back to the first.
"""
function calculate_tour_cost(tour::Vector{Int}, cost_matrix::Matrix{Int})
    n = length(tour)
    if n == 0
        return 0.0
    end
    if n != size(cost_matrix,1)
        # Check if the tour is like [1,2,3,1] (n+1 elements) or [1,2,3] (n elements)
        # For 2-opt, we typically work with permutations [1,2,3] and add closing edge separately.
        if n == size(cost_matrix,1) + 1 && tour[1] == tour[end]
            # Already a closed tour, calculate as is
        else
            error("Tour length does not match cost matrix dimensions or is not a simple permutation.")
        end
    end

    total_cost = 0.0
    for i in 1:(n-1)
        total_cost += cost_matrix[tour[i], tour[i+1]]
    end
    # Add cost from last city back to first, only if tour is a permutation (e.g. [1,2,3])
    # If tour is already [1,2,3,1], the loop above handles it.
    if tour[1] != tour[end] # It's a permutation like [1,2,3]
         total_cost += cost_matrix[tour[n], tour[1]]
    end
    return total_cost
end

"""
    solve_tsp_2_opt(cost_matrix::Matrix{Int}; max_iterations::Int=20000, time_limit_sec::Float64=20.0)

Solves the TSP using the 2-opt heuristic.

Arguments:
- cost_matrix: An n x n matrix of costs.
- max_iterations: Maximum number of iterations for the main loop.
- time_limit_sec: Time limit in seconds for the heuristic.

Returns:
- A tuple: (best_tour_closed, best_cost, computation_time)
  - best_tour_closed: The best tour found, including the return to the start city (e.g., [1, ..., 1]).
  - best_cost: The cost of the best tour.
  - computation_time: The time taken for the heuristic.
"""
function solve_tsp_2_opt(cost_matrix::Matrix{Int}; max_iterations::Int=20000, time_limit_sec::Float64=20.0)
    n = size(cost_matrix, 1)
    if n == 0
        return [], 0.0, 0.0
    end
    if n == 1
        return [1,1], 0.0, 0.0 # Tour for a single city
    end

    current_tour = collect(1:n) # Permutation [1, 2, ..., n]
    # Initial calculation of cost can be outside the timed block if preferred,
    # as it's setup rather than the heuristic itself.
    # However, for simplicity, keeping it as is, or can move initial cost calc out.
    
    best_tour = copy(current_tour)
    # Calculate initial cost for best_tour
    best_cost = calculate_tour_cost(best_tour, cost_matrix)
    current_cost = best_cost
        
    computation_time = @elapsed begin
        improved = true
        iter_count = 0
        start_heuristic_time = time() # For time_limit_sec check inside the loop

        while improved && iter_count < max_iterations && (time() - start_heuristic_time) < time_limit_sec
            improved = false
            iter_count += 1
            
            for i in 1:(n-1) 
                for k in (i+1):n 
                    if k == i || (k+1 > n && i == 1) 
                        continue
                    end
                    
                    new_tour = copy(current_tour)
                    # Reverse the segment from new_tour[i+1] to new_tour[k]
                    # Ensure indices are within bounds if new_tour is 1 to n
                    idx_start_reverse = i + 1
                    idx_end_reverse = k

                    if idx_start_reverse <= idx_end_reverse # Standard case
                        segment_to_reverse = new_tour[idx_start_reverse:idx_end_reverse]
                        reverse!(segment_to_reverse)
                        new_tour[idx_start_reverse:idx_end_reverse] = segment_to_reverse
                    else
                        # This case should not happen with i < k logic for permutation of length n
                        # If tour was [1,2,3,4,1] (n+1), then wrapping logic would be complex here.
                        # Current logic is for permutation [1,2,3,4]
                    end
                                    
                    new_cost = calculate_tour_cost(new_tour, cost_matrix)

                    if new_cost < current_cost
                        current_tour = new_tour
                        current_cost = new_cost
                        improved = true
                        
                        if new_cost < best_cost
                            best_tour = copy(new_tour)
                            best_cost = new_cost
                        end
                        # Using "first improvement" strategy implies breaking here to restart from the new tour
                        # To keep it as a full pass for all pairs then check improved:
                        # remove break statements inside these loops if that was the intention.
                        # For now, assume we want to test all pairs in one pass before 'improved' is checked by while.
                    end
                    if (time() - start_heuristic_time) > time_limit_sec
                        break # break k loop
                    end
                end # k loop
                if (time() - start_heuristic_time) > time_limit_sec || improved
                     # if improved, outer while loop will restart. If not, and time limit hit, break.
                    if improved # if an improvement was found in this i-pass, restart the i-loop effectively via while
                        break # break i loop to go to while condition
                    elseif (time() - start_heuristic_time) > time_limit_sec
                        break # break i loop due to time limit
                    end
                end
            end # i loop
        end # while loop
    end # @elapsed block
    
    final_best_tour_closed = [best_tour..., best_tour[1]]
    
    return final_best_tour_closed, best_cost, computation_time
end

"""
    solve_tsp_concorde(cost_matrix::Matrix{Int})

Solves the TSP using the Concorde.jl package.
Concorde requires integer edge weights.

Arguments:
- cost_matrix: An n x n symmetric matrix of integer costs.

Returns:
- A tuple: (optimal_tour, tour_length, computation_time)
  - optimal_tour: The optimal tour found by Concorde (e.g., [1, ..., 1]).
  - tour_length: The length of the optimal tour.
  - computation_time: The time taken for Concorde to solve.
"""
function solve_tsp_concorde(cost_matrix::Matrix{Int})
    n = size(cost_matrix, 1)
    if n == 0
        return [], 0.0, 0.0
    end
    if n == 1
        return [1,1], 0.0, 0.0
    end

    start_time = time()
    try
        # Using Concorde.solve_tsp instead of Concorde.solve_tsp_LOG
        tour_perm, tour_len = Concorde.solve_tsp(cost_matrix)
        
        computation_time = time() - start_time

        if isempty(tour_perm)
            println("Concorde did not return a tour.")
            return [], Float64(tour_len), computation_time
        end
        
        optimal_tour_closed = [tour_perm..., tour_perm[1]]
        
        return optimal_tour_closed, Float64(tour_len), computation_time
    catch e
        computation_time = time() - start_time
        println("Error running Concorde: $e")
        println("Ensure Concorde executable is installed and in PATH, and Concorde.jl is properly built.")
        println("Also, check the Concorde.jl API for the correct function to call (e.g., solve_tsp).")
        return [], Inf, computation_time
    end
end

"""
    solve_tsp_lkh(cost_matrix::Matrix{Int}; lkh_executable::Union{String, Nothing}=nothing, initial_tour_file::Union{String, Nothing}=nothing, runs::Int=1)

Solves the TSP using the LKH.jl package.
LKH requires integer edge weights.

Arguments:
- cost_matrix: An n x n symmetric matrix of integer costs.
- lkh_executable: Path to the LKH executable if not in default search paths or if specific one is needed.
- initial_tour_file: Path to an initial tour file in LKH format, if desired.
- runs: Number of LKH runs to perform.

Returns:
- A tuple: (optimal_tour, tour_length, computation_time)
  - optimal_tour: The best tour found by LKH (e.g., [1, ..., 1]).
  - tour_length: The length of the best tour.
  - computation_time: The time taken for LKH to solve.
"""
function solve_tsp_lkh(cost_matrix::Matrix{Int}; lkh_executable::Union{String, Nothing}=nothing, initial_tour_file::Union{String, Nothing}=nothing, runs::Int=1)
    n = size(cost_matrix, 1)
    if n == 0
        return [], 0.0, 0.0
    end
    if n == 1
        return [1,1], 0.0, 0.0
    end

    start_time = time()
    try
        # LKH.jl's solve_tsp function can take a cost matrix directly.
        # It returns a tuple: (tour_permutation, tour_length)
        # tour_permutation is 1-indexed.
        
        # Construct keyword arguments for LKH.solve_tsp based on provided parameters
        lkh_kwargs = Dict{Symbol, Any}()
        if lkh_executable !== nothing
            lkh_kwargs[:lkh_exe] = lkh_executable
        end
        if initial_tour_file !== nothing
            lkh_kwargs[:initial_tour_file] = initial_tour_file
        end
        if runs > 1
            lkh_kwargs[:runs] = runs
        end
        # Other parameters like :max_trials, :time_limit could be added here if needed.
        # For now, we'll keep it simple.

        tour_perm, tour_len = LKH.solve_tsp(cost_matrix; lkh_kwargs...)
        
        computation_time = time() - start_time

        if isempty(tour_perm) || tour_len == -1 # LKH might return -1 or Inf for errors/no solution
            println("LKH did not return a valid tour or tour length.")
            # LKH.jl might return tour_len as -1.0 for an error, ensure it's Float64 for consistency
            return [], (tour_len == -1 ? Inf : Float64(tour_len)), computation_time
        end
        
        # The returned tour_perm is a permutation, e.g., [1, 3, 2, 4]. We need to close it.
        optimal_tour_closed = [tour_perm..., tour_perm[1]]
        
        # LKH directly gives the tour length.
        return optimal_tour_closed, Float64(tour_len), computation_time

    catch e
        computation_time = time() - start_time
        println("Error running LKH: $e")
        println("Ensure LKH executable is installed and accessible by LKH.jl (e.g., in PATH or configured). Also check LKH.jl documentation.")
        return [], Inf, computation_time
    end
end

function main()
    n_cities = 10
    # Random.seed!(12345) # Use a fixed seed for comparable results
    cost_matrix = generate_tsp_instance(n_cities, 100)

    println("Generated TSP Instance (n=$n_cities):")
    display(cost_matrix)
    println("\n")

    # --- Solve using MTZ --- 
    println("Solving with MTZ formulation...")
    mtz_tour, mtz_len, mtz_time = solve_tsp_mtz(cost_matrix)
    println("--- MTZ Solver Results ---")
    if !isempty(mtz_tour) && mtz_len != Inf && !(Symbol("ReconstructionFailedMTZ") in mtz_tour)
        @printf "Optimal tour (MTZ): %s\n" join(mtz_tour, " -> ")
        @printf "Tour length (MTZ): %.2f\n" mtz_len
    elseif (Symbol("ReconstructionFailedMTZ") in mtz_tour)
        println("MTZ: Could not reconstruct a valid tour. Edges/Info: ", filter(x->x!=Symbol("ReconstructionFailedMTZ"), mtz_tour))
        @printf "MTZ Reported length: %.2f\n" mtz_len
    else
        println("MTZ: Could not find an optimal tour.")
    end
    @printf "Computation time (MTZ): %.4f seconds\n\n" mtz_time

    # --- Solve using DFJ --- 
    println("Solving with DFJ formulation...")
    dfj_tour, dfj_len, dfj_time, dfj_iters = solve_tsp_dfj(cost_matrix)
    println("--- DFJ Solver Results ---")
    if !isempty(dfj_tour) && dfj_len != Inf && !(Symbol("ReconstructionFailedDFJ") in dfj_tour)
        @printf "Optimal tour (DFJ): %s\n" join(dfj_tour, " -> ")
        @printf "Tour length (DFJ): %.2f\n" dfj_len
    elseif (Symbol("ReconstructionFailedDFJ") in dfj_tour)
        println("DFJ: Could not reconstruct a valid tour. Edges/Info: ", filter(x->x!=Symbol("ReconstructionFailedDFJ"), dfj_tour))
        @printf "DFJ Reported length: %.2f\n" dfj_len
    else
        println("DFJ: Could not find an optimal tour.")
    end
    @printf "Computation time (DFJ): %.4f seconds\n" dfj_time
    @printf "DFJ Iterations: %d\n\n" dfj_iters

    # --- Solve using 2-Opt Heuristic ---
    println("Solving with 2-Opt Heuristic...")
    opt2_tour, opt2_len, opt2_time = solve_tsp_2_opt(cost_matrix, max_iterations=30000, time_limit_sec=15.0)
    println("--- 2-Opt Heuristic Results ---")
    if !isempty(opt2_tour)
        @printf "Best tour (2-Opt): %s\n" join(opt2_tour, " -> ")
        @printf "Tour length (2-Opt): %.2f\n" opt2_len
    else
        println("2-Opt: Could not find a tour.")
    end
    @printf "Computation time (2-Opt): %.4f seconds\n\n" opt2_time
    
    # --- Solve using Concorde.jl ---
    println("Solving with Concorde.jl...")
    concorde_tour, concorde_len, concorde_time = solve_tsp_concorde(cost_matrix)
    println("--- Concorde.jl Results ---")
    if !isempty(concorde_tour) && concorde_len != Inf
        @printf "Optimal tour (Concorde): %s\n" join(concorde_tour, " -> ")
        @printf "Tour length (Concorde): %.2f\n" concorde_len
    else
        println("Concorde: Could not find an optimal tour or an error occurred.")
    end
    @printf "Computation time (Concorde): %.4f seconds\n\n" concorde_time

    # --- Solve using LKH.jl ---
    println("Solving with LKH.jl...")
    # Example: Specify path to LKH executable if not found automatically, and number of runs
    # lkh_exe_path = "C:/path/to/your/LKH-3.X.X/LKH" # Replace with your actual path if needed
    # lkh_tour, lkh_len, lkh_time = solve_tsp_lkh(cost_matrix, lkh_executable=lkh_exe_path, runs=5)
    lkh_tour, lkh_len, lkh_time = solve_tsp_lkh(cost_matrix, runs=1) # Default: 1 run
    println("--- LKH.jl Results ---")
    if !isempty(lkh_tour) && lkh_len != Inf
        @printf "Best tour (LKH): %s\n" join(lkh_tour, " -> ")
        @printf "Tour length (LKH): %.2f\n" lkh_len
    else
        println("LKH: Could not find a tour or an error occurred.")
    end
    @printf "Computation time (LKH): %.4f seconds\n" lkh_time

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 