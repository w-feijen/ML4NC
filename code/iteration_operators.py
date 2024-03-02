import numpy as np
import data_structure
import copy
import random
import math
from itertools import permutations
from scipy import stats
from time import time, sleep
import vroom_operator
from datetime import datetime
import itertools
import NC_attention_model

from pathlib import Path
import joblib


# import grb_iteration_operator
# import grb_iteration_operator2

global model, indices_features_used_by_model, feature_header_line, update_feature_header_line, P_randomness, artificial_analysis_string
model = None
indices_features_used_by_model = None
artificial_analysis_string = None

def set_vroom_exploration_level(vroom_exploration_level):
    vroom_operator.vroom_exploration_level = vroom_exploration_level
       
def init_nc_attention():
    NC_attention_model.init()

def init_nc_attention_with_heuristic(instanceName, heuristicSamplesTestName, nrRoutes):
    NC_attention_model.init()
    
    NC_attention_model.model = joblib.load("..//models//initmodel-" + heuristicSamplesTestName + "-" + instanceName + ".mdl")
    


def set_test_and_runNumber(instanceName, testName, runNumber):
    NC_attention_model.instanceName = instanceName
    NC_attention_model.testName = testName
    NC_attention_model.currentRunNumber = runNumber

def get_random_set_of_permuations(customers, nr_of_permutations):
    if len(customers) <= 5:
        all_permutations = [[c for c in p] for p in permutations(customers)]
        if len(all_permutations) <= nr_of_permutations:
            return all_permutations
        return random.sample(all_permutations, nr_of_permutations)
    
    
    subset_permutations = []
    
    while len(subset_permutations) < nr_of_permutations:
        np.random.shuffle(customers)
        permutation = [c for c in customers]
        if not permutation in subset_permutations:
            subset_permutations += [permutation]
    return subset_permutations

def x_opt(temp_sol, customers, nr_of_permutations = 5):
    temp_sol_violation = temp_sol.total_tw_violation
    temp_sol_value = temp_sol.total_distance + temp_sol.total_tw_violation
    
    for customer in customers:
        temp_sol.remove(customer)
        
        
    subset = get_random_set_of_permuations(customers, nr_of_permutations)
    
    max_decrease = -np.infty
    for customers in subset:
        
        temp_temp_sol = temp_sol.get_copy(check_solution=False)
        temp_msg = temp_temp_sol.insert_multiple(customers)
        temp_temp_sol.update_cost()
        decrease = temp_sol_value - temp_temp_sol.total_distance - temp_temp_sol.total_tw_violation
        if decrease > max_decrease:
            max_decrease = decrease
            max_sol = temp_temp_sol
            max_msg = temp_msg
        if temp_sol_violation - temp_temp_sol.total_tw_violation > 0:
            break
    
    assert max_decrease == temp_sol_value - max_sol.total_distance - max_sol.total_tw_violation
        
    #msg = temp_sol.insert_multiple(customers)
    #temp_sol.update_cost()

    return max_sol, max_msg

def make_prediction(features, settings):
    '''Given a set of feature strings, 
    this method uses network to create prediction , 
    and returns index of the best prediction (if setting is 'highest')
    or samples a neighborhood based on probabilities (if setting is 'sample')
    '''
        
    global model, indices_features_used_by_model
    
    ml_mode = settings['ml_mode']
    if ml_mode == 'ksample_prop' or ml_mode == 'ksample_rnd':
        k = settings['k']
    
    #features =[ [feature[i] for i in range(len(feature)) if not i in columns_to_delete_from_features] for feature in features]
    
    #features =[ [(feature[i]-sample_means[i])/sample_stds[i] for i in range(len(feature))] for feature in features]
    
    if indices_features_used_by_model is None:    
        features = np.array(features)
    else:
        features = np.array(features)[:, indices_features_used_by_model]

    predictions = model.predict_proba(features)[:,1]
    
    if ml_mode == 'sample':
        predictions_norm = predictions/sum(predictions)
        return_index = np.random.choice(list(range(len(features))), p=predictions_norm)    
    elif ml_mode == 'highest':
        return_index = np.argmax(predictions)
    elif ml_mode == 'ksample_prop':
        indices_k_best = np.argsort(predictions)[:k]
        predictions_k_best = [predictions[i] for i in indices_k_best]
        predictions_k_best = predictions_k_best/sum(predictions_k_best)
        return_index = np.random.choice(indices_k_best, p=predictions_k_best)
    elif ml_mode == 'ksample_rnd':
        indices_k_best = np.argsort(predictions)[:k]
        return_index = np.random.choice(indices_k_best)
    else:
        assert False, f'error, setting {ml_mode} unknown'
    
    
    return return_index, predictions


def add_max_min_avg(features, values, name):
    global feature_header_line, update_feature_header_line
    features.extend([max(values), min(values), np.average(values), np.std(values), np.sum(values)])
    
    if update_feature_header_line:
        feature_header_line += f'{name}_max\t{name}_min\t{name}_avg\t{name}_std\t{name}_sum\t'
    return features


def record_all_features(sol, anchors_vehicles, it_nr):
    '''If we add features here, make sure to also adjust the header line, which is added before every run'''
    global feature_header_line, update_feature_header_line
        
    anchors = [c for c,_ in anchors_vehicles]
    vehicles_per_anchor = [v for _,v in anchors_vehicles]
    
    features = [it_nr, len(anchors)]
    
    if update_feature_header_line:       
        feature_header_line = 'date\tit_nr\tsize\t' 
        
    # vehicles_per_anchor = [sol.find_vehicle(c) for c in anchors]
    vehicles = [v for v in sol.vehicles if v in vehicles_per_anchor]
    deliveries = [v.get_delivery(c) for v, c in zip(vehicles_per_anchor, anchors)]
    
    
    
    #Features per customer
    lateness = tuple(d.lateness_value for d in deliveries)
    add_max_min_avg(features, lateness, 'lateness')
    waiting_time = tuple(d.waiting_time for d in deliveries)
    add_max_min_avg(features, waiting_time, 'wait_time')
    feature_collection_type = 3
    if feature_collection_type == 1: 
        #Data collection with quicker closeness. 
        #First improement of closeness
        closeness = tuple(sol.get_closeness(c, False,vehicles) for c in anchors)
        add_max_min_avg(features, closeness, 'closeness')
        temporal_closeness = tuple(0 for c in anchors)
        add_max_min_avg(features, temporal_closeness, 'temp_closeness')
    elif feature_collection_type == 2:
        closeness = tuple(sol.get_closeness_old(c, False,v) for v, c in zip(vehicles_per_anchor, anchors))
        add_max_min_avg(features, closeness, 'closeness')
        temporal_closeness = tuple(sol.get_closeness_old(c, True,v) for v, c in zip(vehicles_per_anchor, anchors))
        add_max_min_avg(features, temporal_closeness, 'temp_closeness')        
    elif feature_collection_type == 3:
        #Newest version of feature_collection with extra improved closeness measures
        
        other_customers = [[c for c in anchors if not c in v.route] for v in vehicles_per_anchor]
        other_vehicles = [[v2 for v2 in vehicles if not v is v2] for v in vehicles_per_anchor]
    
        closeness = tuple(sol.get_closeness(c, False, vehicles) for c in anchors)
        add_max_min_avg(features, closeness, 'closeness')
        closeness2 = tuple(sol.get_closeness_v2(c, v, oc, False)  for v,c,oc in zip(vehicles_per_anchor, anchors, other_customers))
        add_max_min_avg(features, closeness2, 'closeness2')
        closeness2_temp = tuple(sol.get_closeness_v2(c, v, oc, True)  for v,c,oc in zip(vehicles_per_anchor, anchors,other_customers))
        add_max_min_avg(features, closeness2_temp, 'closeness2_temp')
        
        
    distance_cont = tuple(v.get_distance_contribution(c) for v, c in zip(vehicles_per_anchor, anchors))
    add_max_min_avg(features, distance_cont, 'dist_cont')
    tw_length = tuple(c.end - c.start for c in anchors)
    add_max_min_avg(features, tw_length, 'tw_length')
    distance_to_depot = tuple(sol.instance.distance(sol.instance.depot, c) for c in anchors)
    add_max_min_avg(features, distance_to_depot, 'dist_depot')
    load = tuple(c.demand for c in anchors)
    add_max_min_avg(features, load, 'demand')
    # nr_shipments_fit_capacity = calculate_shipment_fit_capacity(load, free_capacity)
    # add_max_min_avg(features, nr_shipments_fit_capacity)
    
    if feature_collection_type == 3:
        min_greedy_addition_cost_both = tuple(sol.get_min_greedy_addition_cost(c,ov,dc) for c,ov,dc in zip(anchors, other_vehicles, distance_cont))
        min_greedy_addition_cost = tuple(x for x,_ in min_greedy_addition_cost_both)
        max_gain_if_possible = tuple(y for _,y in min_greedy_addition_cost_both)
        add_max_min_avg(features, min_greedy_addition_cost, 'min_greedy_addition_cost')
        add_max_min_avg(features, max_gain_if_possible, 'max_gain_if_possible')
        
        possible_delay = tuple(d.possible_delay for d in deliveries)
        add_max_min_avg(features, possible_delay, 'possible_delay')
        
    
    #Features per vehicle 
    route_dist = tuple(v.distance for v in vehicles)
    add_max_min_avg(features, route_dist, 'route_dist')
    average_route_dist = tuple(v.get_average_route_distance() for v in vehicles)
    add_max_min_avg(features, average_route_dist, 'avg_rout_dist')
    empty_distance = tuple(v.get_empty_distance() for v in vehicles)
    add_max_min_avg(features, empty_distance, 'empty_dist')
    distance_worst_case_fraction = tuple(v.get_distance_worst_case_fraction() for v in vehicles)
    add_max_min_avg(features, distance_worst_case_fraction, 'dist_worst_case_frac')
    route_duration = tuple(v.get_route_duration() for v in vehicles)
    add_max_min_avg(features, route_duration, 'route_duration')
    average_route_duration = tuple(v.get_average_route_duration() for v in vehicles)
    add_max_min_avg(features, average_route_duration, 'avg_route_duration')
    idle_time = tuple(v.get_idle_time() for v in vehicles)
    add_max_min_avg(features, idle_time, 'idle_time')
    free_capacity = tuple(v.capacity - v.load for v in vehicles)
    add_max_min_avg(features, free_capacity, 'free_capacity')
    fitting_candidates = tuple(v.get_fitting_candidates(deliveries) for v in vehicles)
    add_max_min_avg(features, fitting_candidates, 'fitting_candidates')
    expected_fit = tuple(v.expected_fit(deliveries) for v in vehicles)
    add_max_min_avg(features, expected_fit, 'expected_fit')
    
    if feature_collection_type == 3:
        distances_between_routes_tw = tuple(v1.distance_to_route_tw(v2) for v1,v2 in itertools.product(vehicles,vehicles) if v1.nr < v2.nr)
        add_max_min_avg(features, distances_between_routes_tw, 'distances_between_routes_tw')
    
    
    if update_feature_header_line:
        feature_header_line += 'total_delta\tdist_delta\tviolation_delta\tnr_violation_delta\tnr_vehicles_delta\ttime\n'
    
    update_feature_header_line = False
    
    return features



def select_customers_routes_random(temp_sol, nr_to_delete):
    '''
    Choose a random set of routes from the solution.
    And return all the customers in those routes.
    '''
    
    vehicles = np.random.choice([v for v in temp_sol.vehicles if v.nr_tw_violations==0], size=nr_to_delete, replace=False)
    
    customers = [c for v in vehicles for c in v.route]
    
    temp_sol.check_solution()
    
    return customers, ""

def select_customers_routes(temp_sol, nr_to_delete, tw_based_distances=False):
    '''
    Choose a set of routes from the solution.
    Choose a set of routes based on locality, more probability is given to routes which are closer.
    And return all the customers in those routes.
    
    ''' 
    global P_randomness
    used_vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0]
    
    #choose one route to build neighborhood around
    anchor_route = np.random.choice([v for v in used_vehicles if v.nr_tw_violations==0])
     
    vehicles_without_anchor = [v for v in used_vehicles if not v is anchor_route]
    
    recommendations = compute_recommendations(anchor_route, vehicles_without_anchor)
    
    vehicles_without_anchor = recommendations
    
    #calculate distance from this route to other routes
    if tw_based_distances:    
        distances = [anchor_route.distance_to_route_tw(v) for v in vehicles_without_anchor]
    else:
        distances = [anchor_route.distance_to_route(v) for v in vehicles_without_anchor]
    
    #Sort vehicles, and then create weight based on position in sorted vector
    sorted_vehicles = [x for x,_ in sorted(zip(vehicles_without_anchor, distances), key=lambda pair: pair[1])]
    p = P_randomness
    weights = [ math.pow(len(vehicles_without_anchor) - i, p) for i in range(len(vehicles_without_anchor))]
    cum_weight = sum(weights)
    probabilities = [weights[i]/cum_weight for i in range(len(vehicles_without_anchor))]
    
    vehicles = list(np.random.choice(sorted_vehicles, size=nr_to_delete, replace=False, p=probabilities))

    vehicles.append(anchor_route)
    
    customers_vehicles = [(c,v) for v in vehicles for c in v.route]
    
    #temp_sol.check_solution()
    
    return customers_vehicles, ""

def artificial_neighborhoods(temp_sol, nr_to_delete, pool_size, nr_to_create, nr_best_to_keep):
    

    #First we create an extra large set of neighborhoods
    candidates_vehicles = [select_customers_routes(temp_sol, nr_to_delete, True) for i in range(nr_to_create)]
    
    #Create neighborhoods     

    # temp_sols = []
    msgs = []
    solution_values = []
    
    sets_of_old_new_vehicles = []
    
    # Calculate all solution values
    
    for anchors_vehicles, anchor_msg in candidates_vehicles:
        
        anchors = [c for c,_ in anchors_vehicles]
        old_vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0 and v.route[0] in anchors]
        vehicles_distance = sum(v.distance for v in old_vehicles)
        
        new_routes, msg = vroom_insertion(temp_sol, old_vehicles, return_routes=True)
        msg = anchor_msg +"\t" + str(len(candidates_vehicles)) +":\t"+ msg
        
        msgs.append(msg)
        
        if new_routes is None:
            print("new routes is None")
            solution_values.append(-np.infty)
        else:
            new_solution_value = temp_sol.total_value - vehicles_distance + sum(temp_sol.instance.calculate_route_distance(r) for r in new_routes)
            solution_values.append(new_solution_value)
        
        old_vehicle_nrs = [v.nr for v in old_vehicles]
        sets_of_old_new_vehicles.append((old_vehicle_nrs,new_routes))
        
        # print(solution_values)
        # print(old_vehicle_nrs, new_routes[0])
        
    
        
    with open(f'..//testRuns//analysis//arti_it_{temp_sol.instance.name}_{artificial_analysis_string}.txt', 'a') as analysis_file:
      analysis_file.write("\t".join([f'{temp_sol.total_value-x:.2f}' for x in solution_values]))
      analysis_file.write("\n")
        
    # print('solution values: ', solution_values)
    # Find the nr_best_to_keep best ones
    best_ones = np.argsort(solution_values)[:nr_best_to_keep]
    # print('best ones', best_ones)
    
    # Add other neighborhoods to get pool_size in total
    worst_ones = np.argsort(solution_values)[nr_best_to_keep:]
    chosen_worst_ones = np.random.choice(worst_ones, pool_size-nr_best_to_keep, replace=False)
    
    indices = np.concatenate((best_ones, chosen_worst_ones))
    
    # print('solution_values:', [f"{x:.0f}" for x in solution_values])
    # print('chosen indices:', indices)
    
    # Return neighborhoods
    
    candidates_vehicles = [ candidates_vehicles[i] for i in indices ] 
    sets_of_old_new_vehicles = [sets_of_old_new_vehicles[i] for i in indices]
    solution_values = [solution_values[i] for i in indices]
    return candidates_vehicles, sets_of_old_new_vehicles, solution_values

def choose_ML(temp_sol, candidates, it_nr, settings):
    '''
    for pool_size many neighborhoods, we compute features and  make the prediction.
    and return index of the best prediction (if setting is 'highest')
    or sample a neighborhood based on probabilities (if setting is 'sample')
    ''' 
    
    features = [record_all_features(temp_sol, candidate_neighborhood, it_nr) for candidate_neighborhood,_ in candidates]
    
    index_best, predictions = make_prediction(features, settings)
    
    anchors_vehicles, msg_anchor = candidates[index_best]
    
    anchors = [c for c,_ in anchors_vehicles]
    vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0 and v.route[0] in anchors]
    
    return vehicles, msg_anchor

def choose_ML_arti(temp_sol, sets_of_old_new_vehicles, candidates, it_nr, settings):
    '''
    for pool_size many neighborhoods, we compute features and  make the prediction.
    and return index of the best prediction (if setting is 'highest')
    or sample a neighborhood based on probabilities (if setting is 'sample')
    ''' 
    
    features = [record_all_features(temp_sol, candidate_neighborhood, it_nr) for candidate_neighborhood,_ in candidates]
    
    index_best, predictions = make_prediction(features, settings)
    
    _, msg_anchor = candidates[index_best]
    
    # print('index best:', index_best)
    
    old_vehicle_nrs, new_routes = sets_of_old_new_vehicles[index_best]
    
    # print('index best old vehicle nrs:', old_vehicle_nrs, new_routes[0])
    
    insert_new_vehicles(temp_sol, old_vehicle_nrs, new_routes)
    
    
    return temp_sol, msg_anchor

def choose_ML_timing(temp_sol, candidates, it_nr):
    '''
    for pool_size many neighborhoods, we compute features and  make the prediction.
    We return the neighborhood with the highest prediction
    
    ''' 
    
    tic = time()
    features = [record_all_features_timing(temp_sol, candidate_neighborhood, it_nr) for candidate_neighborhood,_ in candidates]
    toc = time()
    time_record_features = toc-tic
    
    tic = time()
    index_best, predictions = make_prediction(features)
    toc = time()
    time_make_prediction = toc - tic
    
    anchors_vehicles, msg_anchor = candidates[index_best]
    anchors = [c for c,_ in anchors_vehicles]
    return anchors, msg_anchor, time_record_features, time_make_prediction

def choose_random(temp_sol, candidates):
    '''
    for the candidates, we choose one random one    
    ''' 
    
    choose_index = np.random.choice(len(candidates))
    
    candidates_vehicles,_ = candidates[choose_index]
    
    anchors = [c for c,_ in candidates_vehicles]
    vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0 and v.route[0] in anchors]
    
    return vehicles, ''

def choose_random_arti(temp_sol, sets_of_old_new_vehicles):
    '''
    for the candidates, we choose one random one    
    ''' 
    
    chosen_index = np.random.choice(len(sets_of_old_new_vehicles))
    
    old_vehicle_nrs, new_routes = sets_of_old_new_vehicles[chosen_index]
    insert_new_vehicles(temp_sol, old_vehicle_nrs, new_routes)

     
    return temp_sol, ""

def choose_best(temp_sol, candidates, k=1, setting='random'):
    ''' 
    if k == 1: We run all neighborhoods and choose the best solution and return it. 
    if k > 1: We run all neighborhoods, and sample from the best k solutions
    '''
    
    # temp_sols = []
    msgs = []
    solution_values = []
    
    sets_of_old_new_vehicles = []
    
    for anchors_vehicles, anchor_msg in candidates:
        
        anchors = [c for c,_ in anchors_vehicles]
        old_vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0 and v.route[0] in anchors]
        vehicles_distance = sum(v.distance for v in old_vehicles)
        
        new_routes, msg = vroom_insertion(temp_sol, old_vehicles, return_routes=True)
        msg = anchor_msg +"\t" + str(len(candidates)) +":\t"+ msg
        
        msgs.append(msg)
        
        if new_routes is None:
            solution_values.append(-np.infty)
        else:
            new_solution_value = temp_sol.total_value - vehicles_distance + sum(temp_sol.instance.calculate_route_distance(r) for r in new_routes)
            solution_values.append(new_solution_value)
            
        old_vehicle_nrs = [v.nr for v in old_vehicles]
        sets_of_old_new_vehicles.append((old_vehicle_nrs,new_routes))
        
    
    if k == 1:
        chosen_index = solution_values.index(min(solution_values))
    else:
        indices_k_best = np.argsort(solution_values)[:k]
        
        if setting == 'random':
            chosen_index = np.random.choice(indices_k_best)
        elif setting == 'fixed':
            chosen_index = indices_k_best[k-1]
        elif setting == 'proportional':
            solution_values_best_k = [solution_values[i] for i in indices_k_best]
            sum_values = sum(solution_values_best_k)
            if sum_values == 0:
                chosen_index = np.random.choice(indices_k_best)
            else:
                p = [solution_values[i]/sum_values for i in indices_k_best]
                chosen_index = np.random.choice(indices_k_best, p=p)
    
    old_vehicle_nrs, new_routes = sets_of_old_new_vehicles[chosen_index]
    insert_new_vehicles(temp_sol, old_vehicle_nrs, new_routes)
    
    return temp_sol, msgs[chosen_index]

def choose_best_arti(temp_sol, sets_of_old_new_vehicles, solution_values, k=1, setting='random'):
    ''' 
    if k == 1: We run all neighborhoods and choose the best solution and return it. 
    if k > 1: We run all neighborhoods, and sample from the best k solutions
    '''
    
    # # temp_sols = []
    # msgs = []
    
    # sets_of_old_new_vehicles = []
    
    # for anchors_vehicles, anchor_msg in candidates:
        
    #     anchors = [c for c,_ in anchors_vehicles]
    #     old_vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0 and v.route[0] in anchors]
    #     vehicles_distance = sum(v.distance for v in old_vehicles)
        
    #     new_routes, msg = vroom_insertion(temp_sol, old_vehicles, return_routes=True)
    #     msg = anchor_msg +"\t" + str(len(candidates)) +":\t"+ msg
        
    #     msgs.append(msg)
        
    #     if new_routes is None:
    #         solution_values.append(0)
    #     else:
    #         new_solution_value = temp_sol.total_value - vehicles_distance + sum(temp_sol.instance.calculate_route_distance(r) for r in new_routes)
    #         solution_values.append(new_solution_value)
            
    #     old_vehicle_nrs = [v.nr for v in old_vehicles]
    #     sets_of_old_new_vehicles.append((old_vehicle_nrs,new_routes))
        
    
    if k == 1:
        chosen_index = solution_values.index(min(solution_values))
    else:
        indices_k_best = np.argsort(solution_values)[:k]
        
        if setting == 'random':
            chosen_index = np.random.choice(indices_k_best)
        elif setting == 'fixed':
            chosen_index = indices_k_best[k-1]
        elif setting == 'proportional':
            solution_values_best_k = [solution_values[i] for i in indices_k_best]
            sum_values = sum(solution_values_best_k)
            if sum_values == 0:
                chosen_index = np.random.choice(indices_k_best)
            else:
                p = [solution_values[i]/sum_values for i in indices_k_best]
                chosen_index = np.random.choice(indices_k_best, p=p)
    
    old_vehicle_nrs, new_routes = sets_of_old_new_vehicles[chosen_index]
    insert_new_vehicles(temp_sol, old_vehicle_nrs, new_routes)
    
    return temp_sol, ""

def choose_best_timing(temp_sol, candidates):
    ''' 
    We run all neighborhoods and choose the best solution and return it
    '''
    
    temp_sols = []
    msgs = []
    solution_values = []
    
    tic = time()
    
    for anchors_vehicles, anchor_msg in candidates:
        
        anchors = [c for c,_ in anchors_vehicles]
    
        sol_copy = temp_sol.get_copy(False)
        sol_copy, msg = vroom_insertion(sol_copy, anchors)
        msg = anchor_msg +"\t" + str(len(candidates)) +":\t"+ msg
        sol_copy.update_cost()
        
        temp_sols.append(sol_copy)
        msgs.append(msg)
        solution_values.append(sol_copy.total_value)
    toc = time()
    time_insert = toc - tic
    
    tic = time()
    index_best = solution_values.index(min(solution_values))
    toc = time()
    time_make_prediction =  toc - tic
    return temp_sols[index_best], msgs[index_best], time_make_prediction, time_insert

def insert_new_vehicles(temp_sol, old_vehicle_nrs, new_routes):
    if new_routes is None:
        print('No new routes were returned by VROOM, we set improvement for this neighborhood to 0')
        return
    
    i = 0
    
    for j,v in enumerate(temp_sol.vehicles):
        if v.nr in old_vehicle_nrs:
            if i < len(new_routes):
                route = [temp_sol.instance.customers[c.nr-1] for c in new_routes[i]]
                temp_sol.vehicles[j].route = route
            else:
                temp_sol.vehicles[j].route = []
            temp_sol.vehicles[j]._needs_update=True
            i+=1
            
    temp_sol.update_cost()


def data_collection(temp_sol, candidates, it_nr, datacollection_strategy='random'):
    ''' 
    We record the features for all candidates, we run all neighborhoods in order to complete the y-values of the features
    Then we randomly choose one of the computed solutions and return it
    '''
    
    # temp_sols = []
    msgs = []
    feature_msg = ""
    all_features = []
    
    sets_of_old_new_vehicles = []
    
    
    for anchors_vehicles, anchor_msg in candidates:
        #record features
        features = record_all_features(temp_sol, anchors_vehicles, it_nr)
        if datacollection_strategy == "ML":
            all_features.append(features)
        feature_string = "\t".join([f"{feature:.3f}" for feature in features])
        
        anchors = [c for c,_ in anchors_vehicles]
        old_vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0 and v.route[0] in anchors]
        vehicles_distance = sum(v.distance for v in old_vehicles)
        num_old_vehicles = len(old_vehicles)

        tic = time()
        new_routes, msg = vroom_insertion(temp_sol, old_vehicles, return_routes=True)
        toc = time()
        insert_customers_time = toc-tic
        msg = anchor_msg +"\t" + str(len(candidates)) +":\t"+ msg

        if new_routes is None:
            #Error in calculating new routes in vroom. set to 0
            dist_delta = 0
            total_delta = 0
            nr_vehicles_delta = 0
        else:
            dist_delta = vehicles_distance - sum(temp_sol.instance.calculate_route_distance(r) for r in new_routes)
            total_delta = dist_delta
            nr_vehicles_delta = len(new_routes) - num_old_vehicles

        dt_string = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        feature_string = dt_string + "\t" + feature_string
        feature_string += f"\t{total_delta:.3f}\t{dist_delta:.3f}\t{nr_vehicles_delta}\t{insert_customers_time:.3f}\n"
        feature_msg += feature_string

        msgs.append(msg)
        
        old_vehicle_nrs = [v.nr for v in old_vehicles]
        sets_of_old_new_vehicles.append((old_vehicle_nrs,new_routes))
    
    if datacollection_strategy == 'random':
        choose_index = np.random.choice(len(candidates))
    elif datacollection_strategy == 'ML':
        index_best, predictions = make_prediction(all_features, {'ml_mode':'highest'})
        
        choose_index = index_best
        
    old_vehicle_nrs, new_routes = sets_of_old_new_vehicles[choose_index]
    insert_new_vehicles(temp_sol, old_vehicle_nrs, new_routes)
    
    return temp_sol, msgs[choose_index], feature_msg



def vroom_insertion(temp_sol, vehicles, debug=False, return_routes=False):
    
    '''Create a subproblem for some of the routes.
    '''
    
    # distance_sol_pre_vroom = temp_sol.total_distance

    # vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0 and v.route[0] in anchors]

    # distance_veh_pre_vroom = sum(v.distance for v in vehicles)

    new_routes, msg = vroom_operator.vroom_insert(vehicles, debug)
    
    # distance_sol_post_vroom = temp_sol.total_distance
    # distance_veh_post_vroom = sum(v.distance for v in new_vehicles)

    # assert distance_sol_pre_vroom - distance_sol_post_vroom == distance_veh_pre_vroom - distance_veh_post_vroom

    if return_routes:
        return new_routes, msg
    else:
        #I think here we need to put the new vehicles in the temp_sol?
        #Maybe create a method for it, such that we can also use this method in datacollection
        if not new_routes is None:
            insert_new_vehicles(temp_sol, [v.nr for v in vehicles], new_routes)
        return msg
    
    
def nc_random(sol, nr_to_delete):
    
    '''
    Choose a random set of routes from the solution.
    
    ''' 
    
    used_vehicles = [v for v in sol.vehicles if len(v.route) > 0]
    
    #choose one route to build neighborhood around
    anchor_route = np.random.choice([v for v in used_vehicles if v.nr_tw_violations==0])
    vehicles_without_anchor = [v for v in used_vehicles if not v is anchor_route]
    
    recommendations = compute_recommendations(anchor_route, vehicles_without_anchor)
    
    # vehicles = list(np.random.choice(used_vehicles, size=nr_to_delete, replace=False))
    vehicles = list(np.random.choice(recommendations, size=nr_to_delete, replace=False))

    vehicles.append(anchor_route)
    
    customers_vehicles = [(c,v) for v in vehicles for c in v.route]
    
    #temp_sol.check_solution()
    
    return customers_vehicles, ""

def compute_nc_features(anch_recs):
    
    features = []
    anchor_route = anch_recs[0]
    
    for v in anch_recs:
        v_features = []
        deliveries = [v.get_delivery(c) for c in v.route]
        
        #Features per customer
        lateness = tuple(d.lateness_value for d in deliveries)
        add_max_min_avg(v_features, lateness, 'lateness')
        waiting_time = tuple(d.waiting_time for d in deliveries)
        add_max_min_avg(v_features, waiting_time, 'wait_time')

        
        distance_cont = tuple(v.get_distance_contribution(c) for c in v.route)
        add_max_min_avg(v_features, distance_cont, 'dist_cont')
        tw_length = tuple(c.end - c.start for c in v.route)
        add_max_min_avg(v_features, tw_length, 'tw_length')
        distance_to_depot = tuple(v.instance.distance(v.instance.depot, c) for c in v.route)
        add_max_min_avg(v_features, distance_to_depot, 'dist_depot')
        load = tuple(c.demand for c in v.route)
        add_max_min_avg(v_features, load, 'demand')
        # nr_shipments_fit_capacity = calculate_shipment_fit_capacity(load, free_capacity)
        # add_max_min_avg(v_features, nr_shipments_fit_capacity)
        
        x = tuple(c.x for c in v.route)
        add_max_min_avg(v_features, x, 'x')
        y = tuple(c.y for c in v.route)
        add_max_min_avg(v_features, y, 'y')
        tw_start = tuple(c.start for c in v.route)
        add_max_min_avg(v_features, tw_start, 'tw_start')
        tw_end = tuple(c.end for c in v.route)
        add_max_min_avg(v_features, tw_end, 'tw_end')
        
        possible_delay = tuple(d.possible_delay for d in deliveries)
        add_max_min_avg(v_features, possible_delay, 'possible_delay')
                    
        
        #Features per vehicle 
        route_dist = v.distance
        average_route_dist = v.get_average_route_distance()
        empty_distance = v.get_empty_distance() 
        distance_worst_case_fraction = v.get_distance_worst_case_fraction() 
        route_duration = v.get_route_duration()
        average_route_duration = v.get_average_route_duration()
        idle_time = v.get_idle_time()
        free_capacity = v.capacity - v.load
        dist_to_anch = anchor_route.distance_to_route(v)
        dist_to_anch_tw = anchor_route.distance_to_route_tw(v)
        
        v_features.extend([route_dist, 
                      average_route_dist, 
                      empty_distance, 
                      distance_worst_case_fraction, 
                      route_duration, 
                      average_route_duration, 
                      idle_time, 
                      free_capacity,
                      dist_to_anch, 
                      dist_to_anch_tw])
        
        features.append(v_features)
    
    # print(len(features))
    return features

def compute_recommendations(anchor, other_vehicles):
    min_nr_recs = 50
    nr_shipments_per_route = 5
    tw_penalty_prob = 0
    instance = anchor.instance
    
    if len(anchor.route) < nr_shipments_per_route:
        S_a = anchor.route
    else:
        S_a = np.random.choice(anchor.route, size=nr_shipments_per_route, replace=False)
    
    w = anchor.distance / len(anchor.route)
    
    
    
    shellnumbers = {}
    for v in other_vehicles:
        
        if len(v.route) < nr_shipments_per_route:
            S_o = anchor.route
        else:
            S_o = np.random.choice(v.route, size=nr_shipments_per_route, replace=False)
        
        distances = [instance.distance(c_a, c_o) for c_a in S_a for c_o in S_o]
        

        shellnumber = np.ceil(min(distances)/w)
        shellnumbers[v] = shellnumber
    
    maxShellNumber = max(sn for r,sn in shellnumbers.items())
    for v in other_vehicles:
        if np.random.rand() < tw_penalty_prob and not anchor.has_tw_overlap(v): #Note that this function does not exist yet and needs to be written
            shellnumbers[v] += maxShellNumber
            
    sorted_routes = sorted(shellnumbers.items(), key=lambda x:x[1])
    
    # print("sorted_routes", [(r.nr, sn) for r,sn in sorted_routes])
    
    recs = []
    current_shell_to_add = sorted_routes[0][1]
    
    for r,sn in sorted_routes:
        if sn > current_shell_to_add:
            if len(recs) > min_nr_recs:
                break
            current_shell_to_add = sn
        
        recs.append(r)
        
    np.random.shuffle(recs)
        
    # print("len(recs): ",len(recs), ". recs: ", [r.nr for r in recs])
    return recs
       
def nc_attention(sol, nr_to_delete):
    '''
    Choose a set of routes using attention model.

    '''

    used_vehicles = [v for v in sol.vehicles if len(v.route) > 0]

    #choose one route to build neighborhood around
    anchor_route = np.random.choice([v for v in used_vehicles if v.nr_tw_violations==0])


    vehicles_without_anchor = [v for v in used_vehicles if not v is anchor_route]
    #possibly here: choose recommendations
    recommendations = compute_recommendations(anchor_route, vehicles_without_anchor)

    #Convention: set of vehicles starts with the anchor route:
    anch_recs = [anchor_route] + recommendations

    features = compute_nc_features(anch_recs)
    
    # print(features)

    # vehicles = list(np.random.choice(used_vehicles, size=nr_to_delete, replace=False))
    pi = NC_attention_model.sample_nhood_attention(features, nr_to_delete)

    vehicles = [v for i,v in enumerate(anch_recs) if i in pi]
    # print('num chosen vehicles:', len(vehicles))
    # Need to check if the anchor vehicle is returned in neighborhood creation, otherwise do the following:
    # vehicles.append(anchor_route)

    # print('anchor:', anchor_route.nr, [v.nr for v in vehicles])
    
    customers_vehicles = [(c,v) for v in vehicles for c in v.route]

    #temp_sol.check_solution()

    return customers_vehicles, "", features, pi



# def grb_iteration(temp_sol, it_nr, record_features, record_nhoods, nr_to_delete, use_ML_iterations, pool_size):
def iteration_solver(temp_sol,nr_to_delete, it_nr, strategy, pool_size, record_nhoods, settings):
    '''
    If the pool size is 1, only 1 neighborhood is created and this one will be destroyed and removed
    If the pool size > 1, more neighborhoods are created, and one is chosen based on the strategy:
        if strategy = 'random', a random neighborhood is chosen
        if strategy = 'best', all the neighborhoods are sent to the optimizer and the best one is taken
        if strategy = 'ML', the neighborhood with the best prediction is sent to the solver
        if strategy = 'data_collection', all the neighborhoods are sent to the optimizer, and a random one is taken
    
    '''
    
    feature_msg = None
    nhood_string = None
    
    
    if settings['nc'] == 'random':
        candidates_vehicles = [nc_random(temp_sol, nr_to_delete) for i in range(pool_size)]
    elif settings['nc'] == 'attention':
        candidates_vehicles_features_pi = [nc_attention(temp_sol, nr_to_delete) for i in range(pool_size)]
        
        candidates_vehicles = [(cv,msg) for cv,msg,_,_ in candidates_vehicles_features_pi]
        features_pi = [(ft,pi) for _,_,ft,pi in candidates_vehicles_features_pi]
        
    elif settings['nc'] == 'heuristic':
        if not 'arti' in strategy:
            candidates_vehicles = [select_customers_routes(temp_sol, nr_to_delete, True) for i in range(pool_size)]
        else:
            candidates_vehicles, sets_of_old_new_vehicles, solution_values = artificial_neighborhoods(temp_sol, nr_to_delete, pool_size, settings['nr_to_create'], settings['nr_best_to_keep'])
    else:
        assert False, "Don't recognize neighborhood creation setting:" + settings['nc']
    
    if strategy == 'ML':
        vehicles, msg_anchor = choose_ML(temp_sol, candidates_vehicles, it_nr, settings=settings)
    elif strategy == 'random':
        vehicles, msg_anchor = choose_random(temp_sol, candidates_vehicles)
    elif strategy == 'best':
        #In case of choosing best, the solution (instead of the neighborhood) is returned, because it is calculated already anyway
        temp_sol, msg = choose_best(temp_sol, candidates_vehicles, settings['k'], settings['oracle_mode'])
    elif strategy == 'data_collection':
        temp_sol, msg, feature_msg = data_collection(temp_sol, candidates_vehicles, it_nr, settings['datacollection_strategy'])
    # elif strategy == 'test':
    #     # temp_sol, msg = test_iteration3(temp_sol, candidates_vehicles, it_nr)
    #     temp_sol, msg = testIterationFullInfoChooseBy(temp_sol, candidates_vehicles, it_nr, settings)
    elif strategy == 'ML_arti':
        temp_sol, msg = choose_ML_arti(temp_sol, sets_of_old_new_vehicles, candidates_vehicles, it_nr, settings=settings)
    elif strategy == 'random_arti':
        temp_sol, msg = choose_random_arti(temp_sol, sets_of_old_new_vehicles)
    elif strategy == 'best_arti':
        temp_sol, msg = choose_best_arti(temp_sol, sets_of_old_new_vehicles, solution_values, settings['k'], settings['oracle_mode'])
    
    if strategy == 'ML' or strategy == 'random':
        cost_before = temp_sol.total_value
        msg = vroom_insertion(temp_sol, vehicles)
        msg = msg_anchor +"\t" + str(nr_to_delete) +":\t"+ msg
    
        cost_after = temp_sol.total_value
        improvement = cost_before - cost_after
        
        # print('improvement:', improvement)
        # if record_nhoods:
        #     nhood_string = "\t".join(repr(c.nr) for c in anchors) 
        
    if settings['nc'] == 'attention':
        assert strategy in ['random', 'ML'], "Not yet possible to have different strategy than random or ML because we cannot calculate improvement yet. Easy fix probably to create this"
        #Perform update
        assert pool_size == 1, "Not possible yet to have neighborhood selection + neighborhood creation. Need chosen index to choose correct features and pi"
        index_chosen = 0
        features,pi = features_pi[index_chosen]
        
        
        NC_attention_model.train(features, pi, improvement, nr_to_delete, settings["contlearn"], it_nr)

    return temp_sol, msg, feature_msg, nhood_string


def iteration_solver_timing(temp_sol,nr_to_delete, it_nr, strategy, pool_size, record_nhoods):
    '''
    If the pool size is 1, only 1 neighborhood is created and this one will be destroyed and removed
    If the pool size > 1, more neighborhoods are created, and one is chosen based on the strategy:
        if strategy = 'random', a random neighborhood is chosen
        if strategy = 'best', all the neighborhoods are sent to the optimizer and the best one is taken
        if strategy = 'ML', the neighborhood with the best prediction is sent to the solver
        if strategy = 'data_collection', all the neighborhoods are sent to the optimizer, and a random one is taken
    
    '''
    
    feature_msg = None
    nhood_string = None
    
    tic1 = time()
    candidates_vehicles = [select_customers_routes_location_based(temp_sol, nr_to_delete) for i in range(pool_size)]
    toc = time()
    time_create_nhoods = toc - tic1
    
    if strategy == 'ML':
        anchors, msg_anchor, time_record_features, time_make_prediction = choose_ML_timing(temp_sol, candidates_vehicles, it_nr)
    elif strategy == 'random':
        time_record_features = 0
        time_make_prediction = 0
        anchors = choose_random(temp_sol, candidates)
    elif strategy == 'best':
        #In case of choosing best, the solution (instead of the neighborhood) is returned, because it is calculated already anyway
        time_record_features = 0
        temp_sol, msg, time_make_prediction, time_insert = choose_best_timing(temp_sol, candidates)
    elif strategy == 'data_collection':
        temp_sol, msg, feature_msg = data_collection_choose_random(temp_sol, candidates, it_nr)
    elif strategy == 'test':
        temp_sol, msg = test_iteration(temp_sol, candidates, it_nr)
        
    
    if strategy == 'ML' or strategy == 'random':
        tic = time()
        temp_sol, msg = vroom_insertion(temp_sol, anchors)
        msg = msg_anchor +"\t" + str(nr_to_delete) +":\t"+ msg
        temp_sol.update_cost()
    
        toc = time()
        time_insert = toc - tic
        if record_nhoods:
            nhood_string = "\t".join(repr(c.nr) for c in anchors)
    
    toc1 = time()
    time_iteration = toc1 - tic1
    times = [time_create_nhoods, time_record_features, time_make_prediction, time_insert, time_iteration]
    return temp_sol, msg, feature_msg, nhood_string, times


def completerunupdate(runNumber, instanceName, testName, nrRoutes):
    
    saveDir = "../models/NCattentionmodels/"
    
    init_nc_attention()
    if runNumber > 1:
        #Read model as it was at start of run
        NC_attention_model.load_model(runNumber, saveDir, testName)
        
    #Read all samples
    sampleDir = "../temp - sample dump"
    samples = []
    costs = []
    for path in Path(sampleDir).glob(f"contlearn_samples_{instanceName}_{testName}_{runNumber}_*"):
        # print(path)
        
        samples_temp, costs_temp = joblib.load(path)
        
        samples += samples_temp
        costs += costs_temp
    
    print(f"Update model with {len(samples)} from previous run")
    
    
    samples_costs = list(zip(samples, costs))
    
    #Shuffle samples
    np.random.shuffle(samples_costs)
    
    counter = 0
    #Re update model
    for sample, cost in samples_costs:
        features,selected,_ = sample
        
        if cost < 0:
            NC_attention_model.update_with_sample(features, selected, nrRoutes, cost)
        else:
            counter += 1
    
    print( f'Did not use {counter} many samples because they did not give an improvement')
            
            
            
            
    #Save model for next run
    NC_attention_model.save_model(runNumber+1, saveDir, testName)
