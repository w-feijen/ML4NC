__copyright__ = """Copyright Dassault Systèmes. All rights reserved."""

import numpy as np
from datetime import date, datetime
import joblib

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
            

def write_data_to_file(track_data, instance_filename, test_name):
    '''We save the track_data in a .sav file and we store the info string in the info file.
    '''
    
    datetime_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    #Save info in infofile
    with open( '../testRuns/info.txt', 'a') as info_file:
        info_file.write('\n')
        info_file.write(datetime_string) 
        info_file.write('\n\n')
    
    track_filename = f'../testRuns/track_{test_name}_' + instance_filename.lower().replace('.txt',f'_{datetime_string}.sav')
    joblib.dump(track_data, track_filename)               


def print_iteration_info(it_end_time, it, cur_sol, sign, dist_delta, violation_delta, msg):
    print("%7.2fs" %it_end_time +  " it " +  f'{it:>4}:\t' +
      f'{cur_sol.total_value:9.2f}\t' +
      f'{cur_sol.total_distance:9.2f}{sign}\t' +  
      f'{cur_sol.total_tw_violation:9.2f}' +  
      f' {cur_sol.nr_tw_violations:>3}'+
      f'\t{dist_delta:6.2f}' + f'\t{violation_delta:9.2f}'  + 
      f" ({cur_sol.non_empty_vehicles}/{cur_sol.instance.nr_vehicles})",
      "\t", msg,)