---
layout: 'post'
title: 'Optimisation' 
categories: 'OR'
permalink: '/:categories/:title'
---

## Steps involved solving LP problem

1. import the required library
2. declar the solver
3. create the variables
4. define the constraints
5. define the objective functions
6. invoke the solver
7. display the result

```Python
from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit


def main():
    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return

    # Create the variables x and y.
    x = solver.NumVar(0, 1, 'x')
    y = solver.NumVar(0, 2, 'y')

    print('Number of variables =', solver.NumVariables())

    # Create a linear constraint, 0 <= x + y <= 2.
    ct = solver.Constraint(0, 2, 'ct')
    ct.SetCoefficient(x, 1)
    ct.SetCoefficient(y, 1)

    print('Number of constraints =', solver.NumConstraints())

    # Create the objective function, 3 * x + y.
    objective = solver.Objective()
    objective.SetCoefficient(x, 3)
    objective.SetCoefficient(y, 1)
    objective.SetMaximization()

    solver.Solve()

    print('Solution:')
    print('Objective value =', objective.Value())
    print('x =', x.solution_value())
    print('y =', y.solution_value())


if __name__ == '__main__':
    pywrapinit.CppBridge.InitLogging('basic_example.py')
    cpp_flags = pywrapinit.CppFlags()
    cpp_flags.logtostderr = True
    cpp_flags.log_prefix = False
    pywrapinit.CppBridge.SetFlags(cpp_flags)

    main()
```

### Linear Optimization

### TSP Topic

#### Distance Matrix

1. You can pre-compute the distance and define it explicitly
2. You can use google map api distance matrix to compute the distance [Google Map API](https://developers.google.com/optimization/routing/vrp#distance_matrix_api)

```Python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model():
    """Stores the data for the problem."""
    data = {}

    # distance encodes the distance from i to j
    data['distance_matrix'] = [
        [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
        [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
        [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
        [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
        [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
        [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
        [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
        [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
        [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
        [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
        [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
        [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
        [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
    ]  # yapf: disable
    data['num_vehicles'] = 1
    data['depot'] = 0

    '''
    data['starts'] = [1, 2, 15, 16]
    data['ends'] = [0, 0, 0, 0]
    you can specify starts and ends with multiple cities rather than a single depot
    '''
    return data

'''
0. New York - 1. Los Angeles - 2. Chicago - 3. Minneapolis - 4. Denver - 5. Dallas
- 6. Seattle - 7. Boston - 8. San Francisco - 9. St. Louis - 10. Houston - 11. Phoenix - 12. Salt Lake City
'''

# create distance callback function
def distance_callback(from_index, to_index):
    '''Returns the distance between the two nodes'''
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]


# solution printer
def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)
    plan_output += 'Route distance: {}miles\n'.format(route_distance)


# save routes to a list
def get_routes(solution, routing, manager):
    '''get vehicle routes from a solution and store them in an array'''
    '''i,j entry i is the jth location visited by vehicle i along its route'''
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes

# create the routing model

data = create_data_model()
manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), 
                                        data['num_vehicles'], 
                                        data['depot'])

routing = pywrapcp.RoutingModel(manager)


# call the distance callback function

transit_callback_index = routing.RegisterTransitCallback(distance_callback)

# set the cost of travel

routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

#set search parameters

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

solution = routing.SolveWithParameters(search_parameters)

if solution:
    print_solution(manager, routing, solution)


routes = get_routes(solution, routing, manager)
# Display the routes.
for i, route in enumerate(routes):
  print('Route', i, route)
```

### The procedures involed to solve the problem

1. Create the data
2. Create the routing model
3. Create the distance callback
4. Set the cost of travel
5. Set search parameters
6. Add solution printers
7. Solve and print the solution
8. run the program

[Link to the tutorial](https://developers.google.com/optimization/routing/tsp#python_3)




















