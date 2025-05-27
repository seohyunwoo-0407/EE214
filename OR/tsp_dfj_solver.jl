using JuMP
using Gurobi
using Random # For generate_tsp_instance
using Printf # For formatted output
using LightGraphs # For subtour detection (will be replaced by simple DFS if not available or for simplicity)

"""
    generate_tsp_instance(n::Int, max_cost::Int=100)

Generates a random symmetric TSP cost matrix with integer costs.
Copied from previous scripts for self-contained execution.

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
    find_subtours(edges::Vector{Tuple{Int, Int}}, n_nodes::Int)

Finds all subtours in a given set of edges for a graph with n_nodes.
Uses a simple DFS-based approach.

Arguments:
- edges: A vector of tuples (i,j) representing selected edges.
- n_nodes: The total number of nodes in the graph.

Returns:
- A vector of vectors, where each inner vector is a subtour (a list of nodes in the subtour).
  Returns an empty vector if no subtours (i.e., a single tour involving all nodes, or no edges).
"""
function find_subtours(edges::Vector{Tuple{Int, Int}}, n_nodes::Int)
    if isempty(edges) || n_nodes == 0
        return []
    end

    adj = [[] for _ in 1:n_nodes] # Adjacency list
    for (u, v) in edges
        # Ensure nodes are within bounds (1 to n_nodes)
        if 1 <= u <= n_nodes && 1 <= v <= n_nodes
            push!(adj[u], v)
            push!(adj[v], u) # For TSP, we often get symmetric edges from x_ij variables
        else
            # This case should ideally not happen if x_ij are correctly defined
            # println("Warning: Edge with out-of-bounds node: ($u, $v) for $n_nodes nodes.")
        end
    end
    
    # Normalize adjacency list (remove duplicates)
    for i in 1:n_nodes
        adj[i] = unique(adj[i])
    end

    visited = falses(n_nodes)
    subtours = Vector{Vector{Int}}()
    
    for i in 1:n_nodes
        if !visited[i]
            component = []
            q = [i] # Queue for BFS/Stack for DFS
            visited[i] = true
            head = 1
            
            # Using BFS to find connected components (subtours)
            while head <= length(q)
                u = q[head]
                head += 1
                push!(component, u)
                
                if u <= length(adj) # Check if u is a valid index for adj
                    for v in adj[u]
                        if 1 <= v <= n_nodes && !visited[v]
                            visited[v] = true
                            push!(q, v)
                        end
                    end
                end
            end
            
            if !isempty(component) && length(component) < n_nodes # A true subtour
                push!(subtours, component)
            elseif !isempty(component) && length(component) == n_nodes && count(visited) == n_nodes
                # This is a full tour, not a subtour. Clear subtours if this is the first component found.
                # If other unvisited nodes exist later, they would form separate components/subtours.
                # However, with degree-2 constraints, a full tour should be the only component if no subtours.
                return [] # Indicates a full tour
            end
        end
    end
    return subtours
end


"""
    solve_tsp_dfj(cost_matrix::Matrix{Int})

Solves the Traveling Salesman Problem using the Dantzig-Fulkerson-Johnson (DFJ)
formulation with iterative subtour elimination.

Arguments:
- cost_matrix: An n x n matrix where C[i,j] is the cost from city i to city j.

Returns:
- A tuple: (optimal_tour, tour_length, computation_time, iterations)
"""
function solve_tsp_dfj(cost_matrix::Matrix{Int})
    n = size(cost_matrix, 1)
    if n == 0
        return [], 0.0, 0.0, 0
    end

    model = Model(Gurobi.Optimizer)
    set_silent(model) # Suppress Gurobi output

    @variable(model, x[1:n, 1:n], Bin)

    @objective(model, Min, sum(cost_matrix[i,j] * x[i,j] for i=1:n, j=1:n if i != j))

    # Degree-2 constraints: each city is entered and exited exactly once
    @constraint(model, [i=1:n], sum(x[i,j] for j=1:n if i != j) == 1) # Exit
    @constraint(model, [j=1:n], sum(x[i,j] for i=1:n if i != j) == 1) # Enter
    
    # No self-loops
    @constraint(model, [i=1:n], x[i,i] == 0)

    start_time = time()
    iterations = 0

    while true
        iterations += 1
        # println("DFJ Iteration: $iterations")
        optimize!(model)

        if !(termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
            println("Error: Model not solved to optimality in iteration $iterations. Status: ", termination_status(model))
            return [], Inf, time() - start_time, iterations
        end

        current_edges = []
        for i in 1:n
            for j in 1:n
                if i != j && value(x[i,j]) > 0.99
                    push!(current_edges, (i,j))
                end
            end
        end
        
        # Check for subtours using the simplified DFS/BFS component finder
        sub_tours = find_subtours(current_edges, n)

        if isempty(sub_tours)
            # println("No subtours found. Optimal solution achieved.")
            break # Optimal solution found (single tour)
        end

        # Add subtour elimination constraints
        # println("Found subtours: $sub_tours. Adding constraints.")
        for tour_nodes in sub_tours
            if !isempty(tour_nodes) && length(tour_nodes) < n
                 # sum_{i in S, j in S, i!=j} x_ij <= |S|-1
                @constraint(model, sum(x[i,j] for i in tour_nodes, j in tour_nodes if i!=j) <= length(tour_nodes) - 1)
            end
        end
    end # while loop

    computation_time = time() - start_time
    tour_length = objective_value(model)

    # Reconstruct the tour (similar to MTZ part, can be refactored)
    optimal_tour = []
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
        visited_count = 1
        while visited_count < n
            found_next = false
            for k_edge in 1:length(solution_edges)
                u,v = solution_edges[k_edge]
                if u == curr_city && !(v in optimal_tour)
                    push!(optimal_tour, v)
                    curr_city = v
                    visited_count +=1
                    found_next = true
                    # To avoid reusing edges if graph is not perfectly clean (e.g. if undirected edges were added to list)
                    # However, for x_ij values this should lead to a directed path forming the tour.
                    break
                end
            end
            if !found_next
                # println("Error: Could not reconstruct full tour in DFJ.")
                # This can happen if subtours were not fully eliminated or other issues.
                # Fallback to just listing edges if reconstruction fails badly.
                optimal_tour = [Symbol("ReconstructionFailed")] # Indicate failure
                append!(optimal_tour, solution_edges) 
                break
            end
        end
        if length(optimal_tour) == n # Tour reconstructed successfully
            push!(optimal_tour, optimal_tour[1]) # Close the loop
        elseif !occursin(Symbol("ReconstructionFailed"), optimal_tour) && !isempty(solution_edges)
            println("Warning: Tour reconstruction might be incomplete in DFJ.")
        end
    else
        println("No solution edges found in DFJ result.")
    end
    
    return optimal_tour, tour_length, computation_time, iterations
end


function main()
    n_cities = 10
    # Random.seed!(123) # For consistent testing
    cost_matrix = generate_tsp_instance(n_cities, 100)

    println("Generating a TSP instance with n = $n_cities cities...")
    println("Cost Matrix:")
    display(cost_matrix)
    
    println("\nSolving TSP using DFJ formulation (iterative subtour elimination)...")
    tour, len, time_taken, iters = solve_tsp_dfj(cost_matrix)

    println("\n--- DFJ Solver Results ---")
    if !isempty(tour) && len != Inf && !occursin(Symbol("ReconstructionFailed"), tour)
        @printf "Optimal tour: %s\n" join(tour, " -> ")
        @printf "Tour length: %.2f\n" len
    elseif occursin(Symbol("ReconstructionFailed"), tour)
        println("Could not reconstruct a valid tour. Edges found: ", filter(x->x!=Symbol("ReconstructionFailed"), tour))
        @printf "Reported length (may not correspond to a single tour): %.2f\n" len
    else
        println("Could not find an optimal tour or failed to reconstruct.")
    end
    @printf "Computation time: %.4f seconds\n" time_taken
    @printf "Number of iterations (subtour cuts): %d\n" iters
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 