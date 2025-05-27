using JuMP
using Gurobi
using Random # For generate_tsp_instance
using Printf # For formatted output

"""
    generate_tsp_instance(n::Int, max_cost::Int=100)

Generates a random symmetric TSP cost matrix with integer costs.
Copied from tsp_generator.jl for self-contained execution.

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
        for j in (i+1):n # Ensure symmetry and avoid self-loops initially
            cost = rand(1:max_cost)
            cost_matrix[i, j] = cost
            cost_matrix[j, i] = cost # Symmetric
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
  - optimal_tour: An array representing the sequence of cities in the optimal tour.
  - tour_length: The total length of the optimal tour.
  - computation_time: The time taken to solve the model.
"""
function solve_tsp_mtz(cost_matrix::Matrix{Int})
    n = size(cost_matrix, 1)
    if n == 0
        return [], 0.0, 0.0
    end

    model = Model(Gurobi.Optimizer)
    set_silent(model) # Suppress Gurobi output

    # Decision variables: x[i,j] is 1 if tour goes from i to j, 0 otherwise
    @variable(model, x[1:n, 1:n], Bin)

    # Auxiliary variables for MTZ subtour elimination
    # u[i] stores the position of city i in the tour (1-indexed for city 1)
    @variable(model, u[1:n], Int)

    # Objective function: minimize total cost
    @objective(model, Min, sum(cost_matrix[i,j] * x[i,j] for i=1:n, j=1:n if i != j))

    # Constraints:
    # 1. Each city must be exited exactly once
    @constraint(model, [i=1:n], sum(x[i,j] for j=1:n if i != j) == 1)

    # 2. Each city must be entered exactly once
    @constraint(model, [j=1:n], sum(x[i,j] for i=1:n if i != j) == 1)
    
    # 3. No self-loops (explicitly, though covered by i != j in sums)
    @constraint(model, [i=1:n], x[i,i] == 0)

    # 4. MTZ subtour elimination constraints
    # u[1] is the starting node, conventionally its position is 1
    @constraint(model, u[1] == 1)
    # For other nodes u[i] is between 2 and n
    @constraint(model, [i=2:n], u[i] >= 2)
    @constraint(model, [i=2:n], u[i] <= n)
    # MTZ constraints: u_i - u_j + n*x_ij <= n-1 for i=1..n, j=2..n, i!=j
    @constraint(model, [i=1:n, j=2:n; i != j], u[i] - u[j] + n * x[i,j] <= n-1)

    # Solve the model
    start_time = time()
    optimize!(model)
    computation_time = time() - start_time

    # Extract results
    tour_length = 0.0
    optimal_tour = []

    if termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.LOCALLY_SOLVED || (termination_status(model) == MOI.TIME_LIMIT && has_values(model))
        tour_length = objective_value(model)
        
        # Reconstruct the tour
        solution_edges = []
        for i in 1:n
            for j in 1:n
                if i != j && value(x[i,j]) > 0.99 # Check for 1
                    push!(solution_edges, (i,j))
                end
            end
        end

        if !isempty(solution_edges)
            curr_city = 1 # Start from city 1
            push!(optimal_tour, curr_city)
            while length(optimal_tour) < n
                found_next = false
                for (u,v) in solution_edges
                    if u == curr_city && !(v in optimal_tour)
                        push!(optimal_tour, v)
                        curr_city = v
                        found_next = true
                        break
                    end
                end
                if !found_next
                    # This might happen if solution is not a single tour (e.g. subtours not fully eliminated)
                    # or if problem is infeasible and we somehow got here.
                    println("Error: Could not reconstruct full tour from edges.")
                    # Fallback or error indication
                    optimal_tour = [] # Clear partial tour
                    for edge in solution_edges # just list edges if tour reconstruction fails
                        push!(optimal_tour, edge)
                    end
                    break 
                end
            end
            if length(optimal_tour) == n # if tour is complete (all n cities visited)
                 push!(optimal_tour, optimal_tour[1]) # Add starting city to complete the cycle
            elseif isempty(optimal_tour) && !isempty(solution_edges) # If we couldn't start at 1 or reconstruct
                println("Warning: Tour reconstruction might be incomplete or failed.")
            end
        else
            println("No solution edges found.")
        end

    else
        println("Optimal solution not found. Status: ", termination_status(model))
        # Fallback values
        tour_length = Inf
        optimal_tour = []
    end

    return optimal_tour, tour_length, computation_time
end

function main()
    n_cities = 10
    println("Generating a TSP instance with n = $n_cities cities...")
    # For repeatable tests, you might want to seed Random: Random.seed!(1234)
    cost_matrix = generate_tsp_instance(n_cities, 100) # max_cost = 100
    
    println("Cost Matrix:")
    display(cost_matrix)
    println("\nSolving TSP using MTZ formulation...")

    tour, len, time_taken = solve_tsp_mtz(cost_matrix)

    println("\n--- MTZ Solver Results ---")
    if !isempty(tour) && len != Inf
        @printf "Optimal tour: %s\n" join(tour, " -> ")
        @printf "Tour length: %.2f\n" len
    else
        println("Could not find an optimal tour.")
    end
    @printf "Computation time: %.4f seconds\n" time_taken
end

# Run the main function if the script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 