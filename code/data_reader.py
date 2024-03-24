__copyright__ = """Copyright Dassault Systèmes. All rights reserved."""

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


import data_structure

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