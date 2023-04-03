# Required Libraries
import pandas as pd
import numpy as np
import random
import copy
from matplotlib import pyplot as plt 

# Function: Tour Distance
def distance_calc(Xdata, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + Xdata[city_tour[0][k]-1, city_tour[0][m]-1]           
    return distance


# Function: Initial Seed
def seed_function(Xdata):
    seed = [[],float("inf")]
    sequence = list(range(0, 11))
    seed[0] = sequence
    seed[1] = distance_calc(Xdata, seed)
    return seed


# Function: Build Distance Matrix
def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   D = np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()
   np.savetxt("Distance_Matrix.txt", D)
   return D


# Function: Tour Plot
def plot_tour_coordinates (coordinates, city_tour):
    xy = np.zeros((len(city_tour[0]), 2))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy[:,0], xy[:,1], marker = 'o', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy[0,0], xy[0,1], marker = 'o', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy[1,0], xy[1,1], marker = 'o', alpha = 1, markersize = 7, color = 'orange')
    for i, txt in enumerate(city_tour[0]):
        plt.annotate(txt, (xy[i, 0], xy[i, 1]), fontsize=14, color='red')
    return

# Function: Stochastic 2_opt
def stochastic_2_opt(Xdata, city_tour):
    best_route = copy.deepcopy(city_tour)      
    i, j  = random.sample(range(1, len(city_tour[0])-1), 2) # range start changed from 0 to 1 to keep depot as starting point always
    if (i > j):
        i, j = j, i
    best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))  
    best_route[0][-1]  = best_route[0][0]              
    best_route[1] = distance_calc(Xdata, best_route)
    print(best_route)               
    return best_route


# Function: Local Search
def local_search(Xdata, city_tour, max_attempts = 50, neighbourhood_size = 15):
    count = 0
    Data = [[],[]]
    solution = copy.deepcopy(city_tour)
    while (count < max_attempts):
        Data[0].append(count)
        for i in range(0, neighbourhood_size):
            Data[1].append(i)
            candidate = stochastic_2_opt(Xdata, city_tour = solution)
        if candidate[1] < solution[1]:
            solution  = copy.deepcopy(candidate)
            count = 0
        else:
            count = count + 1                             
    return solution

# Function: Variable Neighborhood Search
def variable_neighborhood_search(Xdata, city_tour, max_attempts = 20, neighbourhood_size = 5, iterations = 50):
    count = 0
    solution = copy.deepcopy(city_tour)
    best_solution = copy.deepcopy(city_tour)
    Delta = 0
    iter = [[],[]]
    while (count < iterations):
        for i in range(0, neighbourhood_size):
            for j in range(0, neighbourhood_size):
                solution = stochastic_2_opt(Xdata, city_tour = best_solution)
            solution = local_search(Xdata, city_tour = solution, max_attempts = max_attempts, neighbourhood_size = neighbourhood_size)
            if (solution[1] < best_solution[1]):
                Delta = best_solution[1] - solution[1]
                best_solution = copy.deepcopy(solution)
                break
        count = count + 1
        iter[0].append(count)
        iter[1].append(best_solution[1])
        print("Iteration = ", count, "-> Distance ", best_solution)
    return best_solution, iter

#################################################################

# Load File - Coordinates
Y = pd.read_csv('training_dataset_10Cities_level1.txt', sep = '\t') 
Y = Y.values

# Build the Distance Matrix
X = build_distance_matrix(Y)

# Start a Random Seed
seed = seed_function(X)

# Call the Function
lsvns = variable_neighborhood_search(X, city_tour = seed, max_attempts = 15, neighbourhood_size = 5, iterations = 10)
iter = lsvns[1][0]
print(iter)
Distance = lsvns[1][1]
print(Distance)
# Plot Solution. Red Point = Initial city; Orange Point = Second City
plot_tour_coordinates(Y, lsvns[0])
plt.savefig("best_route_5_10.png")
plt.figure()
plt.grid()
plt.plot(iter, Distance)
plt.xlabel('Iter')
plt.ylabel('Distance')
plt.show()
# perf = lsvns[1]
# output = open('perf.txt', 'w')
# print(perf, file = output)