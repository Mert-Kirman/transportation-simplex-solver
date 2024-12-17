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


def revised_simplex(A, b, c):
    '''
    Function that takes any linear programming instance as an input and solves it using the Revised Simplex Algorithm

    Parameters
    ----------
    A : matrix containing constraint equations
    b : right hand side of the simplex tableau
    c : matrix storing the cost of supplying a good from a supply node to a demand node (coefficients of decision variables in the objective function)
    '''
    # Identify basic variables and non basic variables from A
    x_basic = list()    # Keep basic variable names
    c_basic = list()    # Keep objective function coefficients of basic variables

    x_non_basic = list()
    c_non_basic = list()
    for j in range(A.shape[1]): # Traverse columns of A
        col = A[:, j]
        if(np.sum(col == 0) == len(col) - 1):
            x_basic.append(j)
            c_basic.append(c[j].item())
        else:
            x_non_basic.append(j)
            c_non_basic.append(c[j].item())

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
            optimal_value = c_basic @ current_b

            print(f'Optimal solution found! Optimal value is: {optimal_value}')
            for i in range(len(x_basic)):
                print(f'x_{x_basic[i]} = {current_b[i]:.2f}')
            for i in range(len(x_non_basic)):
                print(f'x_{x_non_basic[i]} = {0}')
            return
        
        # Not optimal yet, find entering variable
        entering_var = np.argmin(tmp)
        pivot_col = N[:, entering_var]

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

        x_basic[leaving_var] = x_non_basic[entering_var]
        c_basic[leaving_var] = c_non_basic[entering_var]

        x_non_basic[entering_var] = tmp_var_name
        c_non_basic[entering_var] = tmp_var_coefficient


A = np.array([[1, 0, 1, 0, 0],
              [0, 2, 0, 1, 0],
              [3, 2, 0, 0, 1]])

b = np.array([4, 12, 18])
c = np.array([3, 5, 0, 0, 0])

revised_simplex(A, b, c)
