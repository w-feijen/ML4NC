__copyright__ = """Copyright Dassault Systèmes. All rights reserved."""

import numpy as np
import math
import vroom_operator
import NC_attention_model
from pathlib import Path
import joblib

global indices_features_used_by_model, feature_header_line, P_randomness, artificial_analysis_string
global useRecommendations
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

def add_max_min_avg(features, values, name):
    global feature_header_line, update_feature_header_line
    features.extend([max(values), min(values), np.average(values), np.std(values), np.sum(values)])
    
    return features

def nc_heuristic(temp_sol, nr_to_delete, phase, tw_based_distances=False):
    '''
    Choose a set of routes from the solution.
    Choose a set of routes based on locality, more probability is given to routes which are closer.
    And return all the customers in those routes.
    
    ''' 
    global P_randomness
    used_vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0]
    
    #choose one route to build neighborhood around
    if phase == 1:
        anchor_route = used_vehicles[0]
    else:
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


def vroom_insertion(temp_sol, vehicles, debug=False, return_routes=False):
    
    '''Create a subproblem for some of the routes.
    '''
    new_routes, msg = vroom_operator.vroom_insert(vehicles, debug)
    
    if return_routes:
        return new_routes, msg
    else:
        if not new_routes is None:
            insert_new_vehicles(temp_sol, [v.nr for v in vehicles], new_routes)
        return msg
    
    
def nc_random(sol, nr_to_delete, phase):
    
    '''
    Choose a random set of routes from the solution.
    
    ''' 
    global useRecommendations
    
    used_vehicles = [v for v in sol.vehicles if len(v.route) > 0]
    
    #choose one route to build neighborhood around
    if phase == 1:
        anchor_route = used_vehicles[0]
    else:
        anchor_route = np.random.choice([v for v in used_vehicles if v.nr_tw_violations==0])
    vehicles_without_anchor = [v for v in used_vehicles if not v is anchor_route]
    
    
    if phase >= 1 or (phase == 0 and useRecommendations):
        recommendations = compute_recommendations(anchor_route, vehicles_without_anchor)
        vehicles = list(np.random.choice(recommendations, size=nr_to_delete, replace=False))
    else:
        vehicles = list(np.random.choice(used_vehicles, size=nr_to_delete, replace=False))
    
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
       
def nc_attention(sol, nr_to_delete, phase):
    '''
    Choose a set of routes using attention model.

    '''

    used_vehicles = [v for v in sol.vehicles if len(v.route) > 0]

    #choose one route to build neighborhood around
    if phase == 1:
        anchor_route = used_vehicles[0]
    else:
        anchor_route = np.random.choice([v for v in used_vehicles if v.nr_tw_violations==0])


    vehicles_without_anchor = [v for v in used_vehicles if not v is anchor_route]
    
    #possibly here: choose recommendations
    recommendations = compute_recommendations(anchor_route, vehicles_without_anchor)

    #Convention: set of vehicles starts with the anchor route:
    anch_recs = [anchor_route] + recommendations

    features = compute_nc_features(anch_recs)
    
    # vehicles = list(np.random.choice(used_vehicles, size=nr_to_delete, replace=False))
    pi = NC_attention_model.sample_nhood_attention(features, nr_to_delete)

    vehicles = [v for i,v in enumerate(anch_recs) if i in pi]
    
    # Need to check if the anchor vehicle is returned in neighborhood creation, otherwise do the following:
    # vehicles.append(anchor_route)

    customers_vehicles = [(c,v) for v in vehicles for c in v.route]

    return customers_vehicles, "", features, pi



def iteration_solver(temp_sol, nr_to_delete, it_nr, nc_option, cont_learn_option, phase):
    '''
    
    '''
    
    #We do not do neighborhood selection, and therefore have a pool size of one neighborhood per iteration
    pool_size = 1
    
    #Neighborhood Creation
    if nc_option == 'random':
        candidates_vehicles = [nc_random(temp_sol, nr_to_delete, phase) for i in range(pool_size)]
    elif nc_option == 'attention':
        candidates_vehicles_features_pi = [nc_attention(temp_sol, nr_to_delete, phase) for i in range(pool_size)]
        
        candidates_vehicles = [(cv,msg) for cv,msg,_,_ in candidates_vehicles_features_pi]
        features_pi = [(ft,pi) for _,_,ft,pi in candidates_vehicles_features_pi]
        
    elif nc_option == 'heuristic':
        candidates_vehicles = [nc_heuristic(temp_sol, nr_to_delete, phase, True) for i in range(pool_size)]
        
    else:
        assert False, "Don't recognize neighborhood creation setting:" + nc_option
    
    #Possibly: Neighborhood Selection
    #There is only one neighborhood created so it is selected
    assert len(candidates_vehicles)==1
    candidates_vehicles,_ = candidates_vehicles[0]
    anchors = [c for c,_ in candidates_vehicles]
    vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0 and v.route[0] in anchors]
    
    #Destroy+repair solution and calculate size of improvement
    cost_before = temp_sol.total_value
    msg = vroom_insertion(temp_sol, vehicles)
    msg = "\t" + str(nr_to_delete) +":\t"+ msg
    cost_after = temp_sol.total_value
    improvement = cost_before - cost_after
        
    if nc_option == 'attention':
        #Perform update
        index_chosen = 0
        features,pi = features_pi[index_chosen]
        NC_attention_model.train(features, pi, improvement, nr_to_delete, cont_learn_option, it_nr)

    return temp_sol, msg

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
                
    #Save model for next run
    NC_attention_model.save_model(runNumber+1, saveDir, testName)
