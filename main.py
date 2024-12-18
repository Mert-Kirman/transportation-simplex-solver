import numpy as np
import pulp

def generate_instance(num_supply_node, num_demand_node, max_cost, max_amount):
    """"
    Function that generates a feasible instance of the transportation problem according to the given arguments

    Parameters
    ----------
    num_supply_node : number of supply nodes
    num_demand_node : number of demand nodes
    max_cost : maximum possible cost (Maximum value c_ij can take)
    max_amount : maximum demand/supply amount
    """
    # Create an array composed of random numbers representing the max amount of goods a supply node can provide
    supply = np.array([np.random.randint(0, max_amount) for i in range(num_supply_node)])

    # Create an array composed of random numbers representing the max amount of goods a demand node can request
    demand = np.array([np.random.randint(0, max_amount) for i in range(num_demand_node)])

    # Create a matrix composed of randomly chosen costs representing the cost of supplying a good from a supply node to a demand node
    cost_matrix = np.array([[np.random.randint(1, max_cost) for j in range(num_demand_node)] for i in range(num_supply_node)])

    sum_supply = supply.sum()
    sum_demand = demand.sum()

    # Supply and demand should be equal to each other in the given assignment, check for this condition
    if sum_supply > sum_demand:
        extra = sum_supply - sum_demand
        index = 0
        while extra != 0:   # Subtract excess supply
            if supply[index] != 0:
                supply[index] -= 1
                extra -= 1
            index = (index+1) % num_supply_node

    elif sum_demand > sum_supply:
        extra = sum_demand - sum_supply
        index = 0
        while extra != 0:   # Subtract excess demand
            if demand[index] != 0:
                demand[index] -= 1
                extra -= 1
            index = (index+1) % num_demand_node

    return supply, demand, cost_matrix


def solver(supply, demand, cost_matrix):
    '''
    Function that takes a transportation problem instance as input, formulates an LP model and solves it

    Parameters
    ----------
    supply : array storing max amount of goods each supply node can provide
    demand : array storing max amount of goods each demand node can request
    cost_matrix : matrix storing the cost of supplying a good from a supply node to a demand node
    '''

    # Create lists of all supply and demand nodes
    supply_nodes = list()
    demand_nodes = list()
    
    for i in range(len(supply)):
        supply_nodes.append(str(i))
    for i in range(len(demand)):
        demand_nodes.append(str(i))

    # Create dictionaries to store capacities for supply/demand amounts for each supply/demand node
    supply_amounts = dict()
    demand_amounts = dict()

    for i in supply_nodes:
        supply_amounts[i] = supply[int(i)]
    for i in demand_nodes:
        demand_amounts[i] = demand[int(i)]

    # Convert matrix into dictionary format
    costs = pulp.makeDict([supply_nodes, demand_nodes], cost_matrix, 0)

    # Create 'prob' variable to contain the problem data
    prob = pulp.LpProblem("transportation_problem", pulp.LpMinimize)

    # Create a list of tuples containing each route
    routes = [(supply_node, demand_node) for supply_node in supply_nodes for demand_node in demand_nodes]

    # Create dictionary to hold amount of goods send from supply nodes to demand nodes
    decision_vars = pulp.LpVariable.dicts("routes", (supply_nodes, demand_nodes), 0, None, pulp.LpContinuous)

    # Add objective function
    prob += (
        pulp.lpSum([costs[supply_node][demand_node] * decision_vars[supply_node][demand_node] for (supply_node, demand_node) in routes]),
        "total_transportation_cost",
    )

    # Add supply constraints
    for supply_node in supply_nodes:
        prob += (
            pulp.lpSum([decision_vars[supply_node][demand_node] for demand_node in demand_nodes]) == supply_amounts[supply_node],
            f"sum_of_goods_sent_from_supply_node_{supply_node}",
        )

    # Add demand constraints
    for demand_node in demand_nodes:
        prob += (
            pulp.lpSum([decision_vars[supply_node][demand_node] for supply_node in supply_nodes]) == demand_amounts[demand_node],
            f"sum_of_goods_demanded_from_demand_node_{demand_node}",
        )

    prob.writeLP("transportation_problem.lp")
    prob.solve()

    # Print results
    print("Status:", pulp.LpStatus[prob.status])

    # Print each decision variable's optimum value
    for v in prob.variables():
        print(v.name, "=", v.varValue)

    # Print minimized objective function value
    print("Total Transportation Cost = ", pulp.value(prob.objective))


def _revised_simplex_helper(A, b, c, c_original=[]):
    '''
    Helper function for the revised simplex function

    Parameters
    ----------
    A : matrix containing constraint equations
    b : right hand side of the simplex tableau
    c : matrix storing the cost of supplying a good from a supply node to a demand node (coefficients of decision variables in the objective function)
    c_original : a flattened (1D) version of the original cost matrix used for calculating objective function value when Big-M is used. If Big-M method will not be used
    leave this parameter empty, otherwise enter a 1D array containing coefficients of the objective function decision variables
    '''
    # Identify basic variables and non basic variables from A
    x_basic = list()    # Keep basic variable names
    c_basic = list()    # Keep objective function coefficients of basic variables
    c_basic_original = list()

    x_non_basic = list()
    c_non_basic = list()
    c_non_basic_original = list()
    for j in range(A.shape[1]): # Traverse columns of A
        col = A[:, j]
        if(np.sum((-10**(-10) <= col) & (col <= 10**(-10))) == len(col) - 1):
            x_basic.append(j)
            c_basic.append(c[j].item())
            if len(c_original) != 0:    # Big - M method is used
                c_basic_original.append(c_original[j].item())
        else:
            x_non_basic.append(j)
            c_non_basic.append(c[j].item())
            if len(c_original) != 0:
                c_non_basic_original.append(c_original[j].item())

    # Start iterations
    while True:
        # Check for infeasibility
        if not np.all(b > -10**(-10)):  # If a value in RHS is negative
            print('Infeasible')
            return
        
        # Check for optimality
        N = A[:, x_non_basic]
        B = A[:, x_basic]
        B_inverse = np.linalg.inv(B)

        tmp = c_basic @ B_inverse @ N - c_non_basic

        current_b = B_inverse @ b
        if np.all(tmp >= -10**(-10)):
            optimal_value = 0
            if len(c_original) == 0:  # Big-M is not used
                optimal_value = c_basic @ current_b
                optimal_value *= -1
            else:
                optimal_value = c_basic_original @ current_b

            result = list()
            for i in range(len(x_basic)):
                result.append((x_basic[i], current_b[i]))
            for i in range(len(x_non_basic)):
                result.append((x_non_basic[i], 0.0))
            result.append(-optimal_value)
            return result
        
        # Not optimal yet, find entering variable
        entering_var = np.argmin(tmp)
        pivot_col = N[:, entering_var]

        # Check for unboundedness
        if np.all(pivot_col <= 10**(-10)):
            print('Unbounded')
            return

        # Find leaving variable
        min_ratio_index = 0
        min_ratio_value = np.inf
        for ratio_index in range(len(pivot_col)):
            if pivot_col[ratio_index] <= 10**(-10): # Min ratio test not valid, skip this row
                continue

            ratio_value = current_b[ratio_index] / pivot_col[ratio_index]
            if ratio_value < min_ratio_value:
                min_ratio_index = ratio_index
                min_ratio_value = ratio_value

        leaving_var = min_ratio_index

        # Exchange entering and leaving variable names and their coefficients
        tmp_var_name = x_basic[leaving_var]
        tmp_var_coefficient = c_basic[leaving_var]
        if len(c_original) != 0:
            tmp_var_coefficient_original = c_basic_original[leaving_var]

        x_basic[leaving_var] = x_non_basic[entering_var]
        c_basic[leaving_var] = c_non_basic[entering_var]
        if len(c_original) != 0:
            c_basic_original[leaving_var] = c_non_basic_original[entering_var]

        x_non_basic[entering_var] = tmp_var_name
        c_non_basic[entering_var] = tmp_var_coefficient
        if len(c_original) != 0:
            c_non_basic_original[entering_var] = tmp_var_coefficient_original


def revised_simplex(A, b, c, c_original=[]):
    '''
    Function that takes any maximization linear programming instance as an input and solves it using the Revised Simplex Algorithm

    Parameters
    ----------
    A : matrix containing constraint equations
    b : right hand side of the simplex tableau
    c : matrix storing the cost of supplying a good from a supply node to a demand node (coefficients of decision variables in the objective function)
    c_original : a flattened (1D) version of the original cost matrix used for calculating objective function value when Big-M is used. If Big-M method will not be used
    leave this parameter empty, otherwise enter a 1D array containing coefficients of the objective function decision variables
    '''
    result = _revised_simplex_helper(A, b, c, c_original)

    if len(c_original) != 0:    # Big-M is used, do the printing in the big_m() function
        return result
    else:
        if result:  # Result is not empty, optimal solutions have been found
            print(f"Optimal solution found!")
            dec_vars = result[:-1]
            dec_vars.sort(key=lambda x: x[0])
            for dec_var in dec_vars:
                    print(f"x_{dec_var[0]} = {dec_var[1]:.2f}")
            print(f'Objective function value: {result[-1]}')


def big_m(supply, demand, cost_matrix):
    '''
    Function that reformats the given 'supply, demand, cost_matrix' arrays to create A, b, c arrays and solves the given LP using Big-M method by 
    first bringing simplex tableau into canonical form and then calling the revised_simplex function

    Parameters
    ----------
    supply : array storing max amount of goods each supply node can provide
    demand : array storing max amount of goods each demand node can request
    cost_matrix : matrix storing the cost of supplying a good from a supply node to a demand node
    '''
    big_m_value = np.max(cost_matrix) * (10**4)
    c = np.concatenate((cost_matrix.flatten() * -1, [-1 * big_m_value for i in range(len(supply) + len(demand))]), dtype=np.float64)
    b = np.concatenate((supply, demand), dtype=np.float64)
    A = np.zeros((len(b), len(supply) * len(demand)))

    # Fill supply constraint parts of the matrix A
    for i in range(len(supply)):
        for j in range(len(demand)):
            A[i][i*len(demand) + j] = 1

    # Fill demand constraint parts of A
    for i in range(len(supply), len(supply) + len(demand)):
        for j in range(len(supply)):
            A[i][i - len(supply) + j*len(demand)] = 1

    A = np.hstack((A, np.identity(len(b))))

    c_original = np.array(c)
    for i in range(len(b)):
        c += big_m_value * A[i, :]

    result = revised_simplex(A, b, c, c_original)

    if result:  # Result is not empty, optimal solutions have been found
        dec_vars = result[:-1]
        dec_vars.sort(key=lambda x: x[0])
        solution = list()   # Store strings stating the values for each decision variable
        for dec_var in dec_vars:
            if dec_var[0] < len(supply) * len(demand):  # Do not print artificial variables
                solution.append(f"x_{dec_var[0] // len(demand)}_{dec_var[0] % len(demand)} = {dec_var[1]}")
            else:
                if not -10**(-10) <= dec_var[1] <= 10**(-10):   # If an artificial variable is non-zero
                    print("Infeasible!")
                    return

        print(f"Optimal solution found!")
        for i in solution:
            print(i)
        print(f'Objective function value: {result[-1]}')
    

if __name__ == "__main__":
    # Generate a transportation problem
    supply, demand, cost_matrix = generate_instance(3, 2, 10, 10)

    # Use PuLP solver
    print("Solver's answer:")
    solver(supply, demand, cost_matrix)

    print('\n-----------------------------\n')

    # Use our solver
    print("Our answer:")
    big_m(supply, demand, cost_matrix)

    print('\n-----------------------------\n')
    print('Solving a random LP')

    # Solve a random LP apart from transportation problem
    A = np.array([[1,0,1,0,0],[0,2,0,1,0],[3,2,0,0,1]])
    b = np.array([4,12,18])
    c = np.array([3,5, 0, 0, 0])
    revised_simplex(A,b,c)
