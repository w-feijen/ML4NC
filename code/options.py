import random
import numpy as np
from datetime import datetime
import data_reader
import iteration_operators


def set_options(phase, nc_option):
    global test_name, nroftimes, nr_iterations, running_time, show_all_iterations, keep_track_of_data, seed
    global nhood_size, heuristicSamplesTestName, contlearn, init_file_name

    #General test options
    test_name       = 'default'
    nroftimes       = 10                    #Nr of times an instance should be run
    if phase <= 1:                  
        nr_iterations = 500                 #Minimum number of iterations
    else:
        nr_iterations = 1000                #Minimum number of iterations
    running_time    = 1                     #Minimum seconds run should take
    
    #Logging and tracking options
    show_all_iterations = False             #If false, only improving iterations are shown
    keep_track_of_data  = True              #Tracks and saves some data each iteration
    
    #Technical options
    seed        = 42
    random.seed(seed)
    np.random.seed(random.randint(0,2**32-1))

    #Neighborhood creation settings
    nhood_size  = 10                 #Nr routes in a nhood
    if phase == 0:
        contlearn = 'none'
    elif phase == 1:
        contlearn = 'none'
    elif phase == 2:
        heuristicSamplesTestName = "phase2-saveheurist-v3"
        contlearn = 'none'
    elif phase == 3:
        heuristicSamplesTestName = "phase3-saveheurist"
        contlearn = 'completerun'

    #Set the exploration settings
    iteration_operators.P_randomness = 40
    vroom_exploration_level = 5
    iteration_operators.set_vroom_exploration_level(vroom_exploration_level)

    #Filenames and locations
    filenames = data_reader.test_filenames
    init_file_name = "..//data//Solutions//init//Vroom solutions//INSTANCENAME_sol.json"
     
    return filenames
    
def find_phase(args):
    if len(args) < 1:
        print("No phase is given so we follow phase 3")
        phase = 3
    elif args[0] == "0":
        phase = 0
    elif args[0] == "1":
        phase = 1
    elif args[0] == "2":
        phase = 2
    elif args[0] == "3":
        phase = 3
    else:
        raise Exception("Error, given phase: " + args[0] + " unknown. Should be one of [0,1,2,3]")
        
    if phase == 0:
        # global recommendations_options
        nc_option = "random"
        if len(args) < 2:
            print("No recommendations specification is given so we run model with recommendations") 
            recommendations_options = True
        elif args[1] == "0":
            recommendations_options = False
            print("Run model without recommendations")
        elif args[1] == "1":
            recommendations_options = True
            print("Run model with recommendations")
        else:
            raise Exception("Error, given recommendations specification: " + args[1] + " unknown. Should be one of [0,1] for recommendations test")
        iteration_operators.useRecommendations = recommendations_options
        
    if phase >= 1:
        if len(args) < 2:
            print("No neighborhood creation model specification is given so we use random neighborhood creation")
            nc_option = "random"
        elif args[1] == "0":
            nc_option = "random"
        elif args[1] == "1":
            nc_option = "heuristic"
        elif args[1] == "2":
            nc_option = "attention"
        else:
            raise Exception("Error, given neighborhood creation model specification: " + args[1] + " unknown. Should be one of [0,1,2] for phase " + phase)
        
    return phase, nc_option

def print_options(filenames, phase, nc_option):
    
    info_string = 'start ' + datetime.today().strftime('%Y_%m_%d_%H%M') + '\n'
    info_string += f'Do test called "{test_name}", {nroftimes} time(s) per instance\n'
    info_string += f'One run takes either {nr_iterations} iterations or {running_time} seconds \n'
    info_string += 'instances: ' + str(filenames) + '\n'
    info_string += 'tracking? ' + str(keep_track_of_data) + '\n'
    info_string += 'phase: ' + str(phase) + ("(recommendations test)" if phase ==0 else "") + '\n'
    info_string += 'nc_option: ' + nc_option + '\n'
    if phase >= 2:
        info_string += 'heuristcsamples: ' + heuristicSamplesTestName + '\n'
    info_string += 'seed: ' + str(seed) + "\n"
    info_string += 'nhood_size: ' + str(nhood_size) + '\n'
    info_string += 'use_init_from_file: '+init_file_name + '\n'
    
    print(info_string)

def parse_options(args):
    phase, nc_option = find_phase(args)
    filenames = set_options(phase, nc_option)
    print_options(filenames, phase, nc_option)
    
    return filenames, phase, nc_option