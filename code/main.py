import sys
import time
from datetime import datetime
import data_structure
import data_reader
import iteration_operators
import options
import auxiliaries
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

def main(filenames, phase, nc_option):
    
    result = {}
    
    for instance_filename in filenames:
        
        times = []
        best_sols = []
        nr_iterations = []
        
        instance = data_reader.read_data(data_reader.datafolder + instance_filename, True)
            
        if nc_option == 'attention':
            if options.contlearn in ['continuous', 'completerun']:
                #At first iteration of a new test instance we initialize the attention model
                # iteration_operators.init_nc_attention()
                iteration_operators.init_nc_attention_with_heuristic(instance_filename[:-4], options.heuristicSamplesTestName, options.nhood_size)

            else:
                assert options.contlearn == 'none'
                #If there is no continous learning, we initialize at start of every time, so do that in lower for loop

        for i in range(options.nroftimes): 
            
            if nc_option == 'attention' and options.contlearn == 'completerun':
                iteration_operators.set_test_and_runNumber(instance_filename[:-4], options.test_name, i+1)
            
            if nc_option == 'attention' and options.contlearn == 'none':
                if phase == 1:
                    iteration_operators.init_nc_attention()
                if phase >= 2:
                    iteration_operators.init_nc_attention_with_heuristic(instance_filename[:-4], options.heuristicSamplesTestName, options.nhood_size)
                  
            tic = time.time()
            print('Start with iteration', i+1,'/',options.nroftimes, 'at time', datetime.now().strftime("%H:%M:%S"), instance_filename)
            best_sol, track_data = solve(instance, options.init_file_name, phase)
            toc = time.time()
            print("Time taken for %s: %.4f" % ( instance_filename, toc-tic ) )
            
            times += [toc-tic]
            best_sols += [best_sol]
            nr_iterations_in_run = track_data[0][-1]
            nr_iterations += [nr_iterations_in_run]
            if options.keep_track_of_data:
                auxiliaries.write_data_to_file(track_data, instance_filename, options.test_name)

            if nc_option == 'attention' and options.contlearn == 'completerun':
                if i == options.nroftimes - 1: #No need to do update if you just did the last run
                    continue
                iteration_operators.completerunupdate(i+1, instance_filename[:-4], options.test_name, options.nhood_size)
        
        result[instance_filename] = [times, best_sols, nr_iterations]
        
    auxiliaries.print_result(result, filenames)
    return result
                
                
def accept(phase, temp_sol, best_sol):
    
    if phase == 3:
        return temp_sol.is_better_than(best_sol)
    else:
        return False
    
def initialization(instance, init_file_name):
    init_sol = data_structure.solution.from_vroom_json_file(instance, init_file_name.replace("INSTANCENAME", instance.name))
    
    #Print info about solution
    msg = "init dist: %.2f" %init_sol.total_distance
    msg += " viol: %.2f" %init_sol.total_tw_violation
    msg += f" nr viol: {init_sol.nr_tw_violations}"
    msg += " sum: %.2f" %init_sol.total_value
        
    print(msg)
    
    return init_sol
    
def solve(instance, init_file_name, phase):
    init_sol = initialization(instance, init_file_name)
    best_sol = init_sol.get_copy(False)
    cur_sol = init_sol.get_copy(False)
    
    tic = time.time()
    
    it_end_time = 0
    
    track_iterations = [0]
    track_solution_values = [init_sol.total_value]
    track_times = [0]
    track_nr_vehicles = [len(init_sol.vehicles)]
    track_dist_delta = []
    track_improvements = []
    track_predictions = []
    track_indices = []
    
    it = 0
    
    while it < options.nr_iterations or it_end_time < options.running_time:
        
        
        temp_sol = cur_sol.get_copy(False)
        temp_sol, msg = iteration_operators.iteration_solver(temp_sol, options.nhood_size, it, nc_option, options.contlearn, phase)
        
        dist_delta = 0
        sign = " "
        if(temp_sol.is_better_than(best_sol)):
            best_sol = temp_sol.get_copy(False)
            sign = "*"
        accept_temp = accept(phase, temp_sol, cur_sol )
        dist_delta = cur_sol.total_distance - temp_sol.total_distance
        violation_delta = 0
        violation_delta = cur_sol.total_tw_violation - temp_sol.total_tw_violation
            
            
        if accept_temp:
            cur_sol = temp_sol.get_copy(False)
            sign += "^" if dist_delta < 0 else ( "-" if dist_delta == 0 else "v")
            

        toc = time.time()
        it_end_time = toc - tic
        
        track_iterations.append(it+1)
        track_solution_values.append(cur_sol.total_value)
        track_times.append(it_end_time)
        track_nr_vehicles.append(cur_sol.non_empty_vehicles)
        track_dist_delta.append(dist_delta)

        if options.show_all_iterations or not dist_delta == 0 or not violation_delta == 0:
            if cur_sol.total_value - temp_sol.total_value >= 0:
                auxiliaries.print_iteration_info(it_end_time, it, cur_sol, sign, dist_delta, violation_delta, msg)
        
        it += 1
        
    msg = "best dist: %.2f" %best_sol.total_distance
    msg += " viol: %.2f" %best_sol.total_tw_violation
    print(msg)
     
    track_data = [track_iterations, track_times, track_solution_values, track_nr_vehicles, track_improvements, track_predictions, track_indices, track_dist_delta]
    
    return best_sol, track_data

if __name__ == "__main__":
    args = sys.argv[1:]
    filenames, phase, nc_option = options.parse_options(args)
    result = main(filenames, phase, nc_option) #filenames, data_reader.datafolder, num_nhoods, init_file_name)
