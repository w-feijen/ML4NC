
#import time
#from datetime import date, datetime
#import plotly.graph_objects as go
#import numpy as np
#import copy 
#import plotly.offline as py
import random
#import math
from itertools import permutations

import data_structure
import iteration_operators

def find_nearest_neighbors(instance, k, cur_customer, other_customers):
    '''Returns the k nearest neighbors from the current customer '''
    distances = {}
    #print("find nearest neighbors for cur_customer: " + repr(cur_customer) + "in other customers: " + repr(other_customers))
    for c in other_customers:
        distances[c] = instance.distance(cur_customer, c)
    result = [key for key,value in sorted(distances.items(), key=lambda item: item[1])][:k]
    return result



def find_nearest_soonest_neighbors(instance, k, cur_customer, cur_time, other_customers):
    '''Returns the k nearest (both in time and place) neighbors from the current customer 
    Also needs to check if you arrive on time at customer, if you leave at cur_time'''
    distances = {}
    #print("find nearest neighbors for cur_customer: " + repr(cur_customer) + "in other customers: " + repr(other_customers))
    for c in other_customers:
        distance_to_c = instance.distance(cur_customer, c)
        distance_c_to_depot = instance.distance(c, instance.depot)
        waiting_time = max(0, c.start - (cur_time + distance_to_c ))
        
        #check with tw of c and depot if c can be added to route:        
        if (cur_time + distance_to_c <= c.end and
            cur_time + distance_to_c + waiting_time + distance_c_to_depot <= instance.depot.end):
            distances[c] = distance_to_c + waiting_time
            #distances[c] = instance.temporal_distance(cur_customer, c)

    result = [[key,value] for key,value in sorted(distances.items(), key=lambda item: item[1])][:k]
    return result

   

def make_route(instance, vehicle_index, free_cap, cur_customer, customers_to_visit):
    '''Part of construction phase:
    Makes route from cur_customer, and randomly selects one of x nearest neighbors if they still fit.
    This iteratively repeats itself, until no neighbors fit anymore, and truck goes back to depot.'''
    route = [cur_customer]
    customers_to_visit.remove(cur_customer)
    free_cap = free_cap - cur_customer.demand
    x = 3 #nr of nearest neighbors to choose randomly from
    
    #Compute the x nearest neighbors
    distance_depot_to_cur = instance.distance(cur_customer, instance.depot)
        
    cur_time = max( distance_depot_to_cur, cur_customer.start ) + cur_customer.service
    nearest_fitting_neighbors = find_nearest_soonest_neighbors(instance, x, cur_customer, cur_time, [c for c in customers_to_visit if c.demand <= free_cap])
    
    #Add neighbors to route as long as it fits
    while ( len(nearest_fitting_neighbors) > 0 ):
        cur_customer, distance = random.choice(nearest_fitting_neighbors)
        customers_to_visit.remove(cur_customer)
        route += [cur_customer]
        free_cap = free_cap - cur_customer.demand
        #print( "route: " + repr(route) + "freecap: " + repr(free_cap))
        
        assert free_cap >= 0, "The capacity left is smaller than zero: " + str(free_cap)
        cur_time += distance + cur_customer.service
        nearest_fitting_neighbors = find_nearest_soonest_neighbors(instance, x, cur_customer, cur_time, [c for c in customers_to_visit if c.demand <= free_cap])
        
        #print( "Find_nearest_neighbors: " + repr(nearest_fitting_neighbors))
        if len(nearest_fitting_neighbors) == 0 :
            #print("break")
            break
    v = data_structure.vehicle(instance, vehicle_index, instance.capacity, route)
    return v



def cons_phase(instance):
    '''Picks a random customer, and randomly selects one of x nearest neighbors if they still fit.
    If none of the x nearest neighbors fit, then truck goes back to depot. '''
    customers_to_visit = [c for c in instance.customers]
    vehicle_index = 0
    vehicles = []
        
    while len(customers_to_visit) > 0 :
        vehicle_index += 1 #start with index 1
        if instance.restrict_nr_vehicles and vehicle_index > instance.nr_vehicles:
            break
        first_customer, _ = find_nearest_soonest_neighbors(instance, 1, instance.depot, instance.depot.start, customers_to_visit)[0]
        v = make_route(instance, vehicle_index, instance.capacity, first_customer, customers_to_visit)
        vehicles += [v]
         
    result = data_structure.solution(instance, vehicles, check_solution=False)
    
    
    if len(customers_to_visit) > 0:
        #possible if the maximum nr of vehicles was reached.
        #try to insert these customers in some place

        result, _= iteration_operators.x_opt(result, customers_to_visit, 100)
        result.update_cost()
    result.update_cost()
    result.check_solution()
    assert not instance.restrict_nr_vehicles or instance.nr_vehicles == instance.nr_vehicles, "Number of vehicles after construction phase is not equal to forced nr in instance"
    return result


def cons_phase_2(instance):
    '''Sort customers based on start of time window'''
    if not instance.restrict_nr_vehicles:
        assert False, 'Cannot use this construction phase if nr vehicles not restricted'
        
    vehicles = []
    for vehicle_index in range(1,instance.nr_vehicles+1):
        v = data_structure.vehicle(instance, vehicle_index, instance.capacity, [])
        vehicles += [v]
        
    result = data_structure.solution(instance, vehicles, check_solution=False)
    
    sorted_customers = sorted([c for c in instance.customers], key=lambda c: c.start)
    result.insert_multiple(sorted_customers)
    
    result.update_cost()
    return result


def cons_phase_3(instance, plot=False):
    if not instance.restrict_nr_vehicles:
        assert False, 'Cannot use this construction phase if nr vehicles not restricted'
    '''Same as cons_phase_2 but we sort on middle of tw instead of start
    This makes sense for instances where only some of the tw are constraining and others are not, you want to plan the constraining ones first'''
        
    vehicles = []
    for vehicle_index in range(1,instance.nr_vehicles+1):
        v = data_structure.vehicle(instance, vehicle_index, instance.capacity, [])
        vehicles += [v]
        
    result = data_structure.solution(instance, vehicles, check_solution=False)
    
    #create score: time at which you have to leave from depot to arrive in middle of tw
    #idea: either customers which are far away, or customers which need to be served early, are chosen sooenr
    customer_score =  { c.nr: (c.start+c.end) / 2 - instance.distance(c, instance.depot) for c in instance.customers}
    
    sorted_customers = sorted([c for c in instance.customers], key=lambda c: customer_score[c.nr])
    result.insert_multiple(sorted_customers, plot=plot)
    
    result.update_cost()
    return result

# def cons_phase_4(instance):
#     '''combination of cons_phase_1 and cons_phase_3'''
    
#     vehicles = []
#     for vehicle_index in range(1,instance.nr_vehicles+1):
#         v = data_structure.vehicle(instance, vehicle_index, instance.capacity, [])
#         vehicles += [v]
#     result = data_structure.solution(instance, vehicles, check_solution=False)
    
#     customers_to_visit = [c for c in instance.customers]
    
#     while len(customers_to_visit) > 0 :
        
#         #pick the customer which is most urgent as first customer
#         first_customer = min(customers_to_visit, key=lambda c: (c.start + c.end)/2 )
      