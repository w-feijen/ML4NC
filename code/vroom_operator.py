# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:11:42 2021

@author: WFN2
"""

import data_structure
import data_reader
import numpy as np
from subprocess import check_output, PIPE, Popen
import math
import json
import os

import vroom

CUSTOM_PRECISION = 1000

global vroom_exploration_level

vroom_exploration_level = 5

def nint(x):
    return int(x + 0.5)

def euc_2D(c1, c2, PRECISION=1):
    xd = c1[0] - c2[0]
    yd = c1[1] - c2[1]
    return nint(PRECISION * math.sqrt(xd * xd + yd * yd))


# Compute matrix based on ordered list of coordinates.
def get_matrix(coords, PRECISION=1):
    N = len(coords)
    matrix = [[0 for i in range(N)] for j in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            value = euc_2D(coords[i], coords[j], PRECISION)
            matrix[i][j] = value
            matrix[j][i] = value

    return matrix


def get_subproblem_json(vehicles, use_warm_start = True):
    '''Creates a dictionary, based on VROOM style, on which we can apply VROOM solver.
    Only the given vehicles are in this dictionary'''
    
    instance = vehicles[0].instance
    capacity = instance.capacity
    start = instance.depot.start * CUSTOM_PRECISION
    end = instance.depot.end * CUSTOM_PRECISION
    
    meta = {}
    meta["NAME"] = instance.name
    meta["VEHICLES"] = len(vehicles)
    meta["CAPACITY"] = capacity
    meta["JOBS"] = sum(len(v.route) for v in vehicles)
    meta["TIME WINDOW"] = [start, end]
    
    
    location_index = 0
    coords = [[instance.depot.x, instance.depot.y]]
    jobs = []
    
    for v in vehicles:
        for c in v.route:
            location_index += 1
            jobs.append(
                {
                    "id": int(c.nr),
                    "location": [c.x, c.y],
                    "location_index": location_index,
                    "amount": [c.demand],
                    "time_windows": [
                        [
                            CUSTOM_PRECISION * c.start,
                            CUSTOM_PRECISION * c.end,
                        ]
                    ],
                    "service": CUSTOM_PRECISION * c.service,
                })
            coords.append([c.x, c.y])
   
    vehicles_list = []
    for v in vehicles:
        
        v_dict = {
            "id": v.nr,
            "start": coords[0],
            "start_index": 0,
            "end": coords[0],
            "end_index": 0,
            "capacity": [capacity],
            "time_window": [start, end] 
            }
        if use_warm_start:
            steps = []
            for c in v.route:
                steps.append({
                    "type": "job",
                    "id" : c.nr,
                    "service_at" : int(CUSTOM_PRECISION * v.get_delivery(c).service_start)
                    })
            
            v_dict['steps'] = steps
        
        vehicles_list.append(v_dict)


    matrix = get_matrix(coords, CUSTOM_PRECISION)
    
    subproblem_dict = {"meta": meta, "vehicles": vehicles_list, "jobs": jobs, "matrix": matrix}
    return json.dumps(subproblem_dict)


def vroom_subproblem( subproblem_json ):
    '''Calls vroom to solve the subproblem_json and returns the dictionary of the output'''
    
    command_length_limit = 120000
    
    if len(subproblem_json) > command_length_limit:
        
        subproblem_json_temp_file = f'subproblem_json_temp_file_{np.random.randint(1000000)}.txt'
        with open(subproblem_json_temp_file, 'w') as f: f.write(subproblem_json)
        
        p = Popen(['vroom', '-x', '5', '-i', subproblem_json_temp_file], stdout=PIPE)    
        result = p.communicate()[0]
        try:    
            os.remove(subproblem_json_temp_file)
        except:
            print('unable to remove', subproblem_json_temp_file)
            pass
    else:
        p = Popen(['vroom', '-x', '5', subproblem_json], stdout=PIPE)
        result = p.communicate()[0]
    
    
    
    return json.loads(result)

def vroom_subproblem2( subproblem_json ):
    instance = vroom.Input()
    instance._from_json(subproblem_json, False)
    
    solution = instance.solve(exploration_level=vroom_exploration_level, nb_threads=4)
    
    # print(solution.summary.cost)
    
    return solution.to_dict()
    # print(solution.routes[["vehicle_id", "type", "arrival", "location_index", "id"]])
    
def add_subsolution_to_solution(old_vehicles, subsolution_result):
    
          
    route_dicts = subsolution_result['routes']
    
    instance = old_vehicles[0].instance
    
    assert len(old_vehicles) >= len(route_dicts)
    
    new_routes = []
        
    for v, veh_dic in zip(old_vehicles, route_dicts):
        route = [instance.customers[step['job']-1] for step in veh_dic['steps'] if step['type']=='job']
        new_routes += [route]

    if len(old_vehicles) > len(route_dicts):
        print(f'solution found by Vroom with less vehicles {len(route_dicts)} than available {len(old_vehicles)}')
        for v_index in range(len(route_dicts), len(old_vehicles)):
            print(f'Route {old_vehicles[v_index].nr} becomes an empty route')
            # vehicles[v_index].route = []
            # vehicles[v_index]._needs_update=True
            new_routes += [[]]
    
    return new_routes

def vroom_insert(vehicles, debug):
    
    new_routes = None
    
    msg = '-'.join([f'{v.nr:2}' for v in vehicles])
    
    
    use_warm_start = True
    subproblem_json = get_subproblem_json(vehicles, use_warm_start)
    
    # print(subproblem_json)
    
    # print('current solution\n', temp_sol)
    # print('vehicle chosen:', vehicles)
    
    subsolution_result = vroom_subproblem2( subproblem_json )
    # subsolution_result = vroom_subproblem( subproblem_json )
    
    # print(subsolution_result)
    
    if subsolution_result['code'] == 0:
        if subsolution_result['summary']['unassigned']==0:
            new_routes = add_subsolution_to_solution(vehicles, subsolution_result)
    else:
        msg += 'no feasible solution found: error: ' + subsolution_result['error']
    return new_routes, msg