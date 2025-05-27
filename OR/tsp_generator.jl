using Random

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
        for j in (i+1):n # Ensure symmetry and avoid self-loops initially
            cost = rand(1:max_cost)
            cost_matrix[i, j] = cost
            cost_matrix[j, i] = cost # Symmetric
        end
    end
    return cost_matrix
end

function main()
    num_instances = 5
    n = 10
    tsp_instances = []

    println("Generating $num_instances TSP instances with n = $n:")
    for i in 1:num_instances
        instance = generate_tsp_instance(n)
        push!(tsp_instances, instance)
        println("\nInstance $i:")
        display(instance) # Or use println for a more compact view if preferred
    end
    
    # You might want to return tsp_instances if you plan to use them directly
    # in another part of a larger script, or save them to files.
    # For now, we are just printing them as per the initial understanding.
end

# Run the main function if the script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 