# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:23:45 2019

@author: WFN2


#in order to use keras: open anaconda prompt, activate kerasgpu, type spyder (or if it's not installed, conda install spyder)
"""

import time
from datetime import date, datetime
import numpy as np
import copy 
import random

import data_structure
import data_reader
import cons_phase
import iteration_operators
import joblib

    
import sklearn
test_name = 'phase3-attention'

create_route_picture = False    #If you want to see a picture of the created routes

nroftimes = 4                     #Nr of times an instance should be run
nr_it = 50                     #Nr of iterations in a run. (Is redundant when track_sample=True and track_every_iteration = False)

running_time = 1 #0*60 #10*60 #10*60 #60*10 #60*30

strategy = 'random'

keep_track_of_data = True             #Tracks some data each iteration
record_nhoods = False           #Does not work strategy best or data_collection
show_all_iterations = False

use_init_from_file = True

timing = False

vroom_exploration_level = 5  #Before 6 Sept 2023, this was always 5
iteration_operators.set_vroom_exploration_level(vroom_exploration_level)

settings = {'ml_mode':'highest',#'ksample_rnd',#'ksample_prop', #'ksample_rnd','highest', #'highest', 'sample'
            'k':1,
            'oracle_mode': 'best', #'random', 'best', 'proportional'
            'testExecute' : 'random',
            'datacollection_strategy': "ML", # "random", #"random" #"ML"
             'nr_to_create': 20,
            'nr_best_to_keep': 5, #changed from 5 to 3
            'nc' : 'attention',
            'contlearn': 'completerun'#'completerun', #completerun', # 'none, 'continuous', 'completerun'
            }

#seed_string = input('enter seed: ' )
seed = 117 #int(seed_string) #11 #110
random.seed(seed) #changed from 3 to 2
np.random.seed(random.randint(0,2**32-1))

# instance_string = input('enter instance number ( between 1 and 100) ')
# instance_index = int(instance_string)-1

info_string = 'start ' + datetime.today().strftime('%Y_%m_%d_%H%M') + '\n'
info_string += test_name + '\n'
info_string += 'tracking? ' + str(keep_track_of_data) + '\n'
info_string += 'strategy:' + strategy + '\n'
info_string += 'settings: ' + str(settings) + "\n"
info_string += 'seed: ' + str(seed) + "\n"

heuristicSamplesTestName = "phase3-saveheurist"

info_string += "heuristcsamples: " + str(heuristicSamplesTestName) + "\n"

if strategy in ['best', 'random', 'ML', 'test', 'ML_arti', 'random_arti', "best_arti"]:
    filenames = data_reader.test_filenames
    instance_string = '' # '10_1' #'10_10' #'10_10' #This is the name of the actual test instance
    instance_string_2 = '' #'100_10' #'100_norm120-30' #'100_norm60-20' # #'25_30' #'100_norm120-30' #This is used to find the corresponding ML model
    filenames = [f for f in filenames if instance_string in f]
    # filenames = filenames[2:]
    if timing:
        filenames = data_reader.test_filenames[:1]
        
    iteration_operators.P_randomness = 40
elif strategy in ['data_collection']:
    #filenames = ['R1_10_1_100_10_5.txt',
                 #'R1_10_1_100_10_6.txt',
                 #'R1_10_1_100_10_7.txt',
                 #'R1_10_1_100_10_9.txt',
                 #'R1_10_1_100_10_10.txt',]
    # data_collection_split = int(input('data collection split (between 1 and 4):'))
    # info_string += 'data_collection_split: ' + str(data_collection_split) + "\n"
    filenames = data_reader.all_train_instance_names()
    # filenames = [filenames[i] for i in range(len(filenames)) if i % 10 == 0]
    # filenames = filenames[7:]
    data_reader.datafolder = data_reader.train_datafolder
    # filenames = data_reader.train_filenames
    # filenames = ['R1_10_1_100_10_7.txt']
    skip = [] #, 'r1_10_1_100_10_3.txt'] #'R1_10_1_75_10_1.txt']#'R1_10_2.txt',
                    #  'R1_10_4.txt',
                    # 'R1_10_6.txt',
                    # 'R1_10_8.txt',] #['R1_10_1_100_10_6.txt']#, 'R1_10_1_75_10_1.txt',  #skip because done
    #         'R1_10_1_50_10_1.txt', 'R1_10_1_25_10_1.txt', 'R1_10_1_100_30_1.txt' #skip because invalid solutions
    #         ]
    filenames = [f for f in filenames if not f in skip]
    instance_string = '' #This is the name of the actual test instance
    instance_string_2 = "" #"100_10" #'100_norm120-30' #This is used to find the corresponding ML model
    filenames = [f for f in filenames if instance_string_2 in f]
    print('filenames', filenames)
    # filenames = [f for f in filenames if ('20_7' in f)]
    iteration_operators.P_randomness = 10
    
else:
    assert False
    
# if settings['nc'] == 'attention':
    
    
info_string += 'P_randomness: ' + str(iteration_operators.P_randomness) + "\n"

import warnings
#from sklearn.exceptions import UserWarning
warnings.filterwarnings(action='ignore', category=UserWarning)


make_curve_picture = False
if keep_track_of_data:
    make_curve_picture = False      #probably broken, needs to be repaired


#initial value
nhood_size = 10         #used to be nr_to_delete_init

num_nhoods = 1           #how many neighborhoods must be computed in each iteration. The strategy decides which is chosen.


info_string += 'num_nhoods' + str(num_nhoods) + '\n'

init_file_name = None
if use_init_from_file:

    init_file_name = "..//data//Solutions//init//Vroom solutions//INSTANCENAME_sol.json"

    if strategy == 'data_collection' and data_reader.datafolder == data_reader.train_datafolder:
        init_file_name = "..//data//feijen_1000//vroom_solutions//INSTANCENAME_sol.json"

    info_string += 'use_init_from_file: '+init_file_name + '\n'

if strategy in ['ML', 'test', 'ML_arti'] or (strategy == 'data_collection' 
                                  and 'datacollection_strategy' in settings 
                                  and settings['datacollection_strategy'] == "ML"):
    
    model_name = "..//models//clf-2022-08-25-2022-08-26-instanceslikeR1-standard_rf2_balanced.sav"
    model_name = "..//models//clf-2022-10-26-2022-10-28-instanceslikeR1-MLexecdata-standard_rf2_balanced.sav"
    model_name = "..//models//clf-2022-11-08-2022-11-11-instanceslikeR1-MLexecdata2-standard_rf2_balanced.sav"
    model_name = "..//models//clf-2022-10-26-2022-10-28-instanceslikeR1-onlyMLexecdata-standard_rf2_balanced.sav"
    model_name = f"..//models//clf-2022-11-25-2022-12-05-instanceslikeR1_{instance_string_2}-standard_rf2_balanced.sav"
    model_name = f"..//models//clf-2022-12-30-2023-01-20-instanceslikeR1_{instance_string_2}_ML2-standard_rf2_balanced.sav"


    model_name = f"..//models//clf-2022-08-25-2022-08-26-instanceslikeR1-standard_rf2_balanced.sav"
    
    model_name = f"..//models//clf-2023-08-22-2023-08-24-instanceslikeR1_{instance_string_2}-standard_rf2_balanced.sav"


    model_name_prefix = "..//models//clf-2023-09-06-2023-09-12-instanceslikeR1_"
    model_name_suffix = "_ML4explore10-standard_rf2_balanced.sav"

    #Indicate which features are used by the saved model
    iteration_operators.update_feature_header_line = False
    
    info_string += 'model_name: ' + model_name_prefix + "{INSTANCENAME}" + model_name_suffix + '\n'

if strategy in ['data_collection']:
    #At start of run we need to update the header line
    iteration_operators.update_feature_header_line = True
else:
    iteration_operators.update_feature_header_line = False

    
if keep_track_of_data and make_curve_picture:
    import plotly.graph_objects as go
    
if 'arti' in strategy:
    iteration_operators.artificial_analysis_string = test_name + '-' + datetime.now().strftime("%Y%m%d-%H%M%S")

print(info_string)

def print_result(result, filenames):
    
    with open(f'../testRuns/test_{date.today()}.txt', 'a') as resultFile:
        
        for filename in filenames:
            msg = filename + "\t"
            
            times, best_sols, nr_iterations = result[filename]
            avg_score = np.mean([sol.total_value for sol in best_sols])
            avg_dist = np.mean([sol.total_distance for sol in best_sols])
            avg_tw_viol = np.mean([sol.total_tw_violation for sol in best_sols])
            avg_tw_viol_nr = np.mean([sol.nr_tw_violations for sol in best_sols])
            avg_nr_vehicles = np.mean([sol.non_empty_vehicles for sol in best_sols])
            msg += f"{avg_score:.2f}\t"
            msg += f"{avg_dist:.2f}\t"
            msg += f"{avg_tw_viol:.2f}\t"
            msg += f"{avg_tw_viol_nr:.2f}\t"
            msg += f"{np.mean(times):.2f}=("
            for runtime in times:
                msg += f"{runtime:.2f}-"
            msg += ")\t"
            msg += f"{np.mean(nr_iterations):.2f}=("
            for nr_iteration in nr_iterations:
                msg += f"{nr_iteration}-"
            msg += ")\t"
            msg += f"{avg_nr_vehicles:.2f}\t"
            
            for sol in best_sols:
                msg += f'{sol.total_value:.2f}\t'
                msg += f'(\t{sol.total_distance:.2f}\t' 
                msg += f'{sol.total_tw_violation:.2f}\t'
                msg += f' {sol.nr_tw_violations:>3}\t)\t'
                msg += f' {sol.non_empty_vehicles:>3}\t)\t'
                
            print(msg)
            resultFile.write(msg + "\n")
            

def write_data_to_file(track_data, instance_filename, info_string):
    '''We save the track_data in a .sav file and we store the info string in the info file.
    '''
    
    datetime_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    #Save info in infofile
    with open( f'../testRuns/info.txt', 'a') as info_file:
        info_file.write(info_string)
        info_file.write('\n')
        info_file.write(datetime_string) 
        info_file.write('\n\n')
    
    track_filename = f'../testRuns/track_{test_name}_' + instance_filename.lower().replace('.txt',f'_{datetime_string}.sav')
    joblib.dump(track_data, track_filename)
    
            
def write_features_to_file(instance_name, feature_msg):
    today = date.today()
    feature_file = open(f'..//features//features-{instance_name}-{today}-{test_name}.txt', 'a')
    header_line = iteration_operators.feature_header_line
    feature_file.write(header_line)
    feature_file.write(feature_msg)
    feature_file.close()
    
def write_nhoods_to_file(instance_name, test_name, nhoods_msg):
    today = date.today()
    nhoods_file = open(f'..\\features\\nhoods-{instance_name}-{test_name}-{today}.txt', 'a')
    nhoods_file.write(nhoods_msg)
    nhoods_file.close()                

def make_curve_from_run(track_data):
    '''Todo: get curve_data from track_data'''
    x = []
    y = []
    fig = go.Figure() 
    curve = go.Scatter(x=x, y=y, mode='lines')
    fig.add_trace(curve)    
    
    


def main(filenames, datafolder, num_nhoods, init_file_name, clf=None):
    
    result = {}
    
    for instance_filename in filenames:
        
        
        times = []
        best_sols = []
        nr_iterations = []
        
        instance = data_reader.read_data(datafolder + instance_filename, True)
            
        if strategy in ['ML', 'test', 'ML_arti'] or (strategy == 'data_collection' 
                                          and 'datacollection_strategy' in settings 
                                          and settings['datacollection_strategy'] == "ML"):
            model_name = model_name_prefix + data_reader.ML_model_string(instance.name) + model_name_suffix
            print("reading model:" , model_name)
            iteration_operators.model = joblib.load(model_name)
            iteration_operators.indices_features_used_by_model = list(range(1,1 + iteration_operators.model.steps[1][1].n_features_in_))  #In this model, the iteration number is not a feature

        if settings['nc'] == 'attention':
            if settings['contlearn'] in ['continuous', 'completerun']:
                #At first iteration of a new test instance we initialize the attention model
                # iteration_operators.init_nc_attention()
                iteration_operators.init_nc_attention_with_heuristic(instance_filename[:-4], heuristicSamplesTestName, nhood_size)

            else:
                assert settings['contlearn'] == 'none'
                #If there is no continous learning, we initialize at start of every time, so do that in lower for loop

        for i in range(nroftimes): 
            
            if settings['nc'] == 'attention' and settings['contlearn'] == 'completerun':
                iteration_operators.set_test_and_runNumber(instance_filename[:-4], test_name, i+1)
            
            if settings['nc'] == 'attention' and settings['contlearn'] == 'none':
                # iteration_operators.init_nc_attention()
                iteration_operators.init_nc_attention_with_heuristic(instance_filename[:-4], heuristicSamplesTestName, nhood_size)
                
            if 'arti' in strategy:
                with open(f'..//testRuns//analysis//arti_it_{instance.name}_{iteration_operators.artificial_analysis_string}.txt', 'a') as analysis_file:
                  analysis_file.write("\n")
              
            tic = time.time()
            print('Start with iteration', i+1,'/',nroftimes, 'at time', datetime.now().strftime("%H:%M:%S"), instance_filename)
            best_sol, track_data, feature_msg, nhoods_msg = solve(instance, num_nhoods, init_file_name, strategy, clf)
            toc = time.time()
            print("Time taken for %s: %.4f" % ( instance_filename, toc-tic ) )
            
            times += [toc-tic]
            best_sols += [best_sol]
            nr_iterations_in_run = track_data[0][-1]
            nr_iterations += [nr_iterations_in_run]
            if keep_track_of_data:
                write_data_to_file(track_data, instance_filename, info_string)

            if strategy == 'data_collection':
                write_features_to_file(instance.name, feature_msg)
            
            if record_nhoods:
                write_nhoods_to_file(instance.name, test_name, nhoods_msg)

            if keep_track_of_data and make_curve_picture:
                make_curve_from_run(track_data)
                
            if settings['nc'] == 'attention' and settings['contlearn'] == 'completerun':
                if i == nroftimes - 1: #No need to do update if you just did the last run
                    continue
                iteration_operators.completerunupdate(i+1, instance_filename[:-4], test_name, nhood_size)
        
        result[instance_filename] = [times, best_sols, nr_iterations]
        
    print_result(result, filenames)
    return result
                
                
def accept(temp_sol, best_sol):
    return temp_sol.is_better_than(best_sol)

    
def print_iteration_info(it_end_time, it, cur_sol, sign, dist_delta, violation_delta, msg, nr_it_no_improv):
    print("%7.2fs" %it_end_time +  " it " +  f'{it:>4}:\t' +
      f'{cur_sol.total_value:9.2f}\t' +
      f'{cur_sol.total_distance:9.2f}{sign}\t' +  
      f'{cur_sol.total_tw_violation:9.2f}' +  
      f' {cur_sol.nr_tw_violations:>3}'+
      f'\t{dist_delta:6.2f}' + f'\t{violation_delta:9.2f}'  + 
      f" ({cur_sol.non_empty_vehicles}/{cur_sol.instance.nr_vehicles})",
      "\t", msg,
      nr_it_no_improv)
    
def initialization(use_init_from_file, instance, init_file_name, create_route_picture):
    if use_init_from_file:
        
        init_sol = data_structure.solution.from_vroom_json_file(instance, init_file_name.replace("INSTANCENAME", instance.name))
        
    else:
        init_sol_1 = cons_phase.cons_phase(instance)
        print("init sol 1:", f"{init_sol_1.total_value:.2f}")

        
        init_sol_2 = cons_phase.cons_phase_2(instance)
        print("init sol 2:", f"{init_sol_2.total_value:.2f}")

        
        init_sol_3 = cons_phase.cons_phase_3(instance)
        print("init sol 3:", f"{init_sol_3.total_value:.2f}")
        
        init_sols = [init_sol_1, init_sol_2, init_sol_3]
        init_sol = min(init_sols, key=lambda init_sol: init_sol.total_value)
        
        filename = f'init//init{instance.name}_{datetime.today().strftime("%Y_%m_%d_%H%M") }_{instance.name}.txt'
        init_sol.write(None, filename=filename)
        print(filename)
    
    #Print info about solution
    msg = "init dist: %.2f" %init_sol.total_distance
    msg += " viol: %.2f" %init_sol.total_tw_violation
    msg += f" nr viol: {init_sol.nr_tw_violations}"
    msg += " sum: %.2f" %init_sol.total_value
        
    print(msg)
    
    if create_route_picture:
        init_sol.plot("init")
        init_sol.print_extensive()
        
    return init_sol

def print_timing_results(track_times, time_totals, nr_iterations):
    
    total_time = track_times[-1]
    print('time taken:', total_time)
    print(f'average iteration length            : {total_time/nr_iterations:10.3f}')
    print(f'average iteration length in function: {time_totals[4]/nr_iterations:10.3f}  {100*time_totals[4]/time_totals[4]:8.3f}%')
    print(f'average neighborhood creation       : {time_totals[0]/nr_iterations:10.3f}  {100*time_totals[0]/time_totals[4]:8.3f}%')
    print(f'average recording features          : {time_totals[1]/nr_iterations:10.3f}  {100*time_totals[1]/time_totals[4]:8.3f}%')
    print(f'average making prediction           : {time_totals[2]/nr_iterations:10.3f}  {100*time_totals[2]/time_totals[4]:8.3f}%')
    print(f'average inserting                   : {time_totals[3]/nr_iterations:10.3f}  {100*time_totals[3]/time_totals[4]:8.3f}%')
    
    
def solve(instance, num_nhoods, init_file_name, strategy, clf=None):
    global nr_to_delete, a
    
    init_sol = initialization(use_init_from_file, instance, init_file_name, create_route_picture)
    best_sol = init_sol.get_copy(False)
    cur_sol = init_sol.get_copy(False)
    
    tic = time.time()
    
    it_end_time = 0
    
    nr_it_no_improv = 0
    
    track_iterations = [0]
    track_solution_values = [init_sol.total_value]
    track_times = [0]
    track_nr_vehicles = [len(init_sol.vehicles)]
    track_dist_delta = []
    track_improvements = []
    track_predictions = []
    track_indices = []
    
    
    if strategy == 'data_collection':
        feature_msg = ""
    else:
        feature_msg = None
    if record_nhoods:
        nhoods_msg = ""
    else:
        nhoods_msg = None

    if timing:
        time_totals = [0,0,0,0,0]
        
    it = 0
    
    while it < nr_it or it_end_time < running_time:
        
        
        temp_sol = cur_sol.get_copy(False)
        if timing:
            temp_sol, msg, feature_string, nhoods_string, times = iteration_operators.iteration_solver_timing(temp_sol,nhood_size, it, strategy, num_nhoods, record_nhoods) 
            time_totals = [time_totals[t] + times[t] for t in range(5)]   
        else:
            temp_sol, msg, feature_string, nhoods_string = iteration_operators.iteration_solver(temp_sol,nhood_size, it, strategy, num_nhoods, record_nhoods, settings)
        
        if strategy == 'test':
            msg, predictions, improvements, best_index, random_index, ML_index = msg
        
        if strategy == 'data_collection':
            feature_msg += feature_string
        if record_nhoods:
            nhoods_msg += nhoods_string
                
        dist_delta = 0
        sign = " "
        if(temp_sol.is_better_than(best_sol)):
            best_sol = temp_sol.get_copy(False)
            sign = "*"
        accept_temp = accept(temp_sol, cur_sol )
        dist_delta = cur_sol.total_distance - temp_sol.total_distance
        violation_delta = 0
        violation_delta = cur_sol.total_tw_violation - temp_sol.total_tw_violation
            
            
        if accept_temp:
            #Question: why do we need to make a copy here??
            
            cur_sol = temp_sol.get_copy(False)
            if dist_delta <=0:
                nr_it_no_improv +=1
            else:
                nr_it_no_improv = 0
            sign += "^" if dist_delta < 0 else ( "-" if dist_delta == 0 else "v")
            #cur_sol.plot("it"+repr(it))
        else:
            nr_it_no_improv += 1
            

        toc = time.time()
        it_end_time = toc - tic
        
        
        track_iterations.append(it+1)
        track_solution_values.append(cur_sol.total_value)
        track_times.append(it_end_time)
        track_nr_vehicles.append(cur_sol.non_empty_vehicles)
        track_dist_delta.append(dist_delta)
        if strategy == 'test':
            track_improvements.append(improvements)
            track_predictions.append(predictions)
            track_indices.append([best_index, random_index, ML_index])
    
        if show_all_iterations or not dist_delta == 0 or not violation_delta == 0:
            if cur_sol.total_value - temp_sol.total_value >= 0:
                print_iteration_info(it_end_time, it, cur_sol, sign, dist_delta, violation_delta, msg, nr_it_no_improv)
        
            
        it += 1
    
    # best_sol.write('best')                
    if create_route_picture:
        best_sol.plot("best")
        best_sol.write('best')
        print(best_sol)
        
    msg = "best dist: %.2f" %best_sol.total_distance
    msg += " viol: %.2f" %best_sol.total_tw_violation
    print(msg)
     
    track_data = [track_iterations, track_times, track_solution_values, track_nr_vehicles, track_improvements, track_predictions, track_indices, track_dist_delta]
    
    if timing:
        print_timing_results(track_times, time_totals, track_iterations[-1])
    
    return best_sol, track_data, feature_msg, nhoods_msg

result = main(filenames, data_reader.datafolder, num_nhoods, init_file_name)
