# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:35:39 2019

@author: WFN2
"""

small_datafolder = '..//data//solomon_25//'
datafolder = '..//data//solomon_100//'
#datafolder = 'data//homberger_1000_customer_instances//'

train_datafolder = '..//data//feijen_1000//'

#train_filenames = [
#                   "R101.txt",
#                   "R103.txt",
#                   "R105.txt",
#                   "R108.txt",
#                   "R110.txt",
#                   "R112.txt"
#                   ]

# test_filenames = [
#                     # "R102.txt",
#                     # "R104.txt",
#                     # "R106.txt",
#                     # "R107.txt",
#                    # "R109.txt",
#                     # "R111.txt"
#                     # "R102L.txt",
#                     "R104L.txt",
#                     "R106L.txt",
#                     "R107L.txt",
#                     "R109L.txt",
#                     "R111L.txt"
#                     ]

test_filenames = [ 'R1_10_1.TXT',
                    'R1_10_2.TXT',
                    'R1_10_3.TXT',
                    'R1_10_4.TXT',
                    'R1_10_5.TXT',
                    'R1_10_6.TXT',
                    'R1_10_7.TXT',
                    'R1_10_8.TXT',
                    'R1_10_9.TXT',
                    'R1_10_10.TXT',
                  ]

test_filenames_ordered = ['R1_10_1.txt',
                    'R1_10_2.txt',
                    'R1_10_3.txt',
                    'R1_10_4.txt',
                    'R1_10_5.txt',
                    'R1_10_6.txt',
                    'R1_10_7.txt',
                    'R1_10_8.txt',
                    'R1_10_9.txt',
                    'R1_10_10.txt',
                  ]

conversion_dict = { 'r1_10_1':'100_10',
                    'r1_10_2':'75_10',
                    'r1_10_3':'50_10',
                    'r1_10_4':'25_10',
                    'r1_10_5':'100_30',
                    'r1_10_6':'75_30',
                    'r1_10_7':'50_30',
                    'r1_10_8':'25_30',
                    'r1_10_9':'100_norm60-20',
                    'r1_10_10':'100_norm120-30',
                  }


# test_filenames = ['R1_10_2.txt']

train_filenames = ['R1_10_2.txt',
                     'R1_10_4.txt',
                    'R1_10_6.txt',
                    'R1_10_8.txt',
                   'R1_10_10.txt',
                  ]

filenames = [
             # "easy.txt"
             # "C101_10.txt",
             # "C101.txt",
             # "C102.txt",
             # "C103.txt",
             # "C104.txt",
             # "C105.txt",
             # "C106.txt",
             # "C107.txt",
             # "C108.txt",
             # "C109.txt",
             # "C201.txt",
              # "R101.txt",
              # "R102.txt",
              # "R103.txt",
              # "R104.txt",
              # "R105.txt",
              # "R106.txt",
              # "R107.txt",
              # "R108.txt",
              # "R109.txt",
              # "R110.txt",
              # "R111.txt",
              # "R112.txt",
              # "R201.txt",
             # "R202.txt",
             # "R204.txt",
             # "RC101.txt",
             # "RC102.txt",
             # "RC201.txt",
             # "RC202.txt",
             # "RC1_10_10.txt",
             # "R1_10_1.txt"
             # 'R105_v0.txt',
             # 'R105_v1.txt',
             # 'R105_v2.txt',
             # 'R105_v3.txt',
             # 'R105_v4.txt',
             # 'R105_v5.txt',
             # 'R105_v6.txt',
             # 'R105_v7.txt',
             # 'R105_v8.txt',
             # 'R105_v9.txt',
             # 'R105_w1.txt',
             # 'R105_w2.txt',
             # 'R105_w3.txt',
             # 'R105_w4.txt',
             # 'R105_w5.txt',
             # 'R105_w6.txt',
             # 'R105_w7.txt',
             # 'R105_w8.txt',
             # 'R105_w9.txt',
               # 'R105_w10.txt',
             ]


import data_structure
import numpy as np

accept_missing__nr_vehicles_optimal = False

use_only_25 = False
use_only_50 = False
use_1000 = True
assert not (use_only_25 and use_only_50)
if use_only_25:
    datafolder = '..//data//solomon_25//'
if use_only_50:
    datafolder = '..//data//solomon_50//'
if use_1000:
    datafolder = '..//data//homberger_1000_customer_instances//'
    accept_missing__nr_vehicles_optimal = True


log = False
def printif( string ):
    if (log):
        print(string)

def read_data(filename, restrict_nr_vehicles):
    
    fw = open( filename, 'r' )
    lines = [line.rstrip('\n') for line in fw]
    

    name = lines[0]
    customers = []
    l = 1
    while l < len(lines):
        line = lines[l].split()
        if(len(line) < 1):
            l+=1
            continue
        printif(line)
        if line[0] == "VEHICLE":
            printif(lines[l+1].split())
            l+=2
            line = lines[l].split()
            printif(line)
            nr_vehicles = int(line[0])
            capacity = int(line[1])
            if not accept_missing__nr_vehicles_optimal and restrict_nr_vehicles:
                assert len(line) > 2, "Nr of vehicles in world record not given. Change instance."
                nr_vehicles = int(line[2])
        if line[0] == "CUSTOMER":
            printif(lines[l+1].split())
            l+=3
            break
        l += 1
    while l < len(lines):
        line = lines[l].split()
        printif(line)
        cust = data_structure.customer( int(line[0]),
                                        int(line[1]),
                                        int(line[2]),
                                        int(line[3]),
                                        int(line[4]),
                                        int(line[5]),
                                        int(line[6]) )
        if cust.nr == 0:
            depot = cust
        else:
            customers += [cust]
        l += 1
    fw.close()
    inst = data_structure.instance(name, depot, customers, nr_vehicles, capacity, restrict_nr_vehicles)
    #inst.plot()
    return inst


def get_all_instance_names(basic_instance_name, nr_of_instances, tw_lengths, fraction_tws, normal_tw_lengths_parameters):
    result = []
    
    for i in range(nr_of_instances):
        for tw_length in tw_lengths:
            for fraction_tw in fraction_tws:
                instance_name = basic_instance_name + f"_{int(fraction_tw*100)}_{tw_length}_{i+1}.txt"  
                result.append(instance_name)
        for avg,std in normal_tw_lengths_parameters:
            instance_name = basic_instance_name + f"_{100}_norm{avg}-{std}_{i+1}.txt"
            result.append(instance_name)
    return result

def all_train_instance_names(split = None):
    
    basic_instance_filename = "r1_10_1"
    nr_of_instances = 10
    tw_lengths = [10,30]
    # tw_lengths = [10]
    fraction_tws = [1.0, 0.75, 0.5, 0.25]
    # fraction_tws = [1.0]
    normal_tw_lengths_parameters = [(60,20), (120,30)]
    # normal_tw_lengths_parameters = []
    instance_names = get_all_instance_names(basic_instance_filename, nr_of_instances, tw_lengths, fraction_tws, normal_tw_lengths_parameters)
    
    ##Take out all the ones that we have done already
    #instance_names = [instance_names[i] for i in range(len(instance_names)) if not i % 10 == 0]
    
    
    
    
    # if split == 1:
    #     halfway = int(len(instance_names)/2)
    #     instance_names = instance_names[:halfway]
    # elif split == 2:
    #     halfway = int(len(instance_names)/2)
    #     instance_names = instance_names[halfway:]
    if not split is None:
        assert split in [1,2,3,4], "Error, split should be between 1 and 4"
        return list(np.array_split(instance_names, 4))[split-1]
    return instance_names

def ML_model_string(instance_name):
    '''Checks the name of the instance, and returns the string necessary for reading the ML model that corresponds to this instance'''
    
    name_split = instance_name.split("_")
    if len(name_split) == 3: #Standard instance, e.g. r1_10_1, or r1_10_10
        return conversion_dict[instance_name]
    elif len(name_split) == 6 and 'r1_10_1' in instance_name: #data colleciton instance, eg. r1_10_1_100_norm60-20_1
        return name_split[3] + "_" + name_split[4]
    else:
        print("ERROR: DON'T KNOW WHICH ML MODEL TO READ FOR INSTANCE", instance_name)
        assert False
        
