import numpy as np
import math
from os import path
import copy
import json

time_window_violation_cost = 10
cost_per_violation = 100

class vehicle(object):
    def __init__(self, instance, nr, capacity, route, distance=None, load=None, first_deliveries=None):
        self.nr = nr
        self.capacity = capacity
        self.route = route
        self.instance = instance
        
        self.centroid = None
        
        
        if distance is None:
            self._needs_update = True
            self.update()
        else:

            self.distance = distance
            self.load = load 
            
            self.create_deliveries(first_deliveries)
            if self.instance.restrict_nr_vehicles:
                self.calculate_tw_violation()
            
            
            self.calc_vehicle_centroid()
            
            self._needs_update = False
        
       
    def __repr__(self):
        return f"v{self.nr}\t({self.load:3}/{self.capacity})" 
    
    def get_route_segs(self):
        '''Returns a list of tuples [c1,c2] which represent the edges of the route'''
        result = []
        depot = self.instance.depot
        u = depot
        for c in self.route:
            v = c
            result += [(u,v)]
            u = v
        result += [(u,depot)]
        return result
    
    def calculate_distance(self):
        result = 0
        for c1,c2 in self.get_route_segs():
            result += self.instance.distance(c1,c2)
        self.distance = result
    
    def calculate_load(self):
        self.load = sum([c.demand for c in self.route])
        
    def create_deliveries(self, first_deliveries = []):
        '''Possible to input some deliveries, then we only need to compute missing ones'''
        cur_time = self.instance.depot.start
        #self.departures = {}
        #self.arrivals = {}
        #self.service_start = {}
        
        nr_new_deliveries = len(self.route) + 1 - len(first_deliveries)
        if nr_new_deliveries == 0:
            self.deliveries = first_deliveries
            return
        else:
            self.deliveries = first_deliveries + [None]*(nr_new_deliveries)
            
            
        service_start = -1
        arrival = -1
        waiting_time = -1
        lateness_value = -1
        
        for i, seg in enumerate(self.get_route_segs()):
            c1,c2 = seg
            
            if i < len(first_deliveries):
                #Delivery already exists, don't need to compute it
                
                continue
            if i > 0 and i == len(first_deliveries):
                cur_time = first_deliveries[i-1].departure
                previous_customer = first_deliveries[i-1].customer
                cur_time += self.instance.distance(previous_customer,c1)
                arrival = cur_time
                if arrival < c1.start:
                    waiting_time = c1.start - arrival
                    lateness_value = np.float32(0)
                    cur_time = c1.start
                elif arrival > c1.end:
                    waiting_time = np.float16(0)
                    lateness_value = arrival - c1.end
                else:
                    lateness_value = np.float16(0)
                    waiting_time = np.float16(0)
                    
                    
                #cur_time = max(cur_time, c1.start)
                service_start = cur_time
                cur_time += c1.service                
            
            departure = cur_time
            
            self.deliveries[i] = delivery(c1, arrival, departure, service_start, waiting_time, lateness_value)
            
            cur_time += self.instance.distance(c1,c2)
            #assert cur_time <= c2.end, repr(self) + 'arrives at ' + repr(c2) + ' at ' + repr(cur_time) + 'but time window ends at ' + repr(c2.end)
            #if cur_time > c2.end:
            #    print('error:',  repr(self) + 'arrives at ' + repr(c2) + ' at ' + repr(cur_time) + 'but time window ends at ' + repr(c2.end))
                
            arrival = cur_time
            
            
            if arrival < c2.start:
                waiting_time = c2.start - arrival
                lateness_value = np.float32(0)
                cur_time = c2.start
            elif arrival > c2.end:
                waiting_time = np.float16(0)
                lateness_value = arrival - c2.end
            else:
                waiting_time = np.float16(0)
                lateness_value = np.float16(0)
            #cur_time = max(cur_time, c2.start)
            service_start = cur_time
            cur_time += c2.service
        #self.service_start[self.instance.depot] = 0
        #self.deliveries[0].arrival = cur_time
        self.deliveries[0].update_arrival(cur_time)
        
    def get_delivery(self, customer):
        for delivery in self.deliveries:
            if delivery.customer.nr == customer.nr:
                return delivery
            
        assert False, f"Delivery of customer {customer.nr} not found in {repr(self)}"
        
        
    def calculate_tw_violation(self):
        #there is a cost for coming too late at each customer (and depot) where it is late
        self.tw_violation = sum(delivery.lateness_cost for delivery in self.deliveries)
        self.nr_tw_violations = sum( not delivery.is_on_time for delivery in self.deliveries)
        self.nr_tw_violation_costs = self.nr_tw_violations * self.instance.cost_per_violation
        
    def get_idle_time(self):
        '''Adds up the waiting time for all deliveries. 
        (Important to also add delivery for the depot, since there can be waiting time, in the beginning of the horizon, as well).'''
        return sum(d.waiting_time for d in self.deliveries)
        
    def check_time_windows(self):
        #returns if all the arrivals fall in the time window
        msg = ""
        is_correct = True
        for i,seg in enumerate(self.get_route_segs()):
            c1,c2=seg
            if self.deliveries[c2].arrival > c2.end: 
                is_correct = False
                msg += repr(self) + 'arrives at ' + repr(c2) + ' at ' + repr(self.deliveries[c2].arrival) + ' but time window ends at ' + repr(c2.end) + "\n"
        return is_correct, msg
    
    #def calculate_waiting_time(self):
    #    self.waiting_time =  sum([self.deliveries[c].waiting_time for c in self.route])
            
    def show_time_routes(self):

            
            #string += "%s[%.f]((%.f))[%.f]--%.f-->[%.f]\n" %(repr(c1), self.deliveries[c1].service_start, c1.service, self.deliveries[c1].departure, self.instance.distance(c1,c2), self.deliveries[c2].arrival)
        print(self.schedule_string())
        
    def schedule_string(self):
        ''' for each customer we print the following:
            c1 [service_start]---s:(service_duration)---[departure]---d:distance c1 to c2--- and in case waiting time: [arrival c2]---w:(waiting time)---'''
        string = "Route for " + repr(self) + ":\n"
        
        for i,seg in enumerate(self.get_route_segs()):
            c1,c2=seg
            delivery_c1 = self.get_delivery(c1)
            delivery_c2 = self.get_delivery(c2)
            
            string += repr(c1)
            string += f"[{delivery_c1.service_start:.2f}]"
            string += f"---s:({c1.service:.2f})---"
            string += f"[{delivery_c1.departure:.2f}]"
            string += f"---d:{self.instance.distance(c1,c2):.2f}---"
            if delivery_c2.waiting_time > 0 :
                string += f"[{delivery_c2.arrival:.2f}]"
                string += f"---w:{delivery_c2.waiting_time:.2f}---"    
            string += "\n"
        return string
        
           

        
    def update(self):
        if self._needs_update:
            self.calculate_distance()
            self.calculate_load()
            self.create_deliveries()
            if self.instance.restrict_nr_vehicles:
                self.calculate_tw_violation()
            self.calc_vehicle_centroid()
        self._needs_update = False
        
        
    def get_distance_contribution(self, customer):
        '''computes the contribution that this customer has to the distance of the vehicle, 
        calculated by previous and next customer: dist(prev,customer) + dist(customer,next) - dist(prev,next)'''
        assert customer.nr in [c.nr for c in self.route], f'cannot compute distance contribution for client {customer.nr} that is not in route of vehicle {self}: {self.route}'
        
        prev_cus = None
        next_cus = None
        for c1,c2 in self.get_route_segs():
            if customer.nr == c2.nr:
                prev_cus = c1
            elif customer.nr == c1.nr:
                next_cus = c2
            if prev_cus and next_cus:
                break
        assert prev_cus
        assert next_cus
        return self.instance.distance(prev_cus, customer) + self.instance.distance(customer, next_cus) - self.instance.distance(prev_cus, next_cus)
        
        
    def get_average_route_distance(self):
        '''computes the average route distance of the vehicle, 
        by dividing the distance over the number of customers'''
        
        if len(self.route) == 0:
            return 0
            #assert False, 'When developing, unclear when this situation would show up. Todo: check when you have an empty route and see if it suffices to set average route distance to 0'
        return self.distance / len(self.route)
    
    def get_empty_distance(self):
        '''returns the distance which the vehicle drives after the last delivery'''
        
        if len(self.route) == 0:
            return 0
        return self.instance.distance(self.instance.depot, self.route[-1])
        
    def get_distance_worst_case_fraction(self):
        '''Computes the worst case for deliveries in the vehicle. 
        Worst case is driving from depot to delivery back to depot, and this repeated for all customers. 
        Return the vehicle distance divided by the worst case.'''
        
        if len(self.route) == 0:
            return 1
        
        worst_case = sum([(2*self.instance.distance(self.instance.depot, c))  for c in self.route])
        return self.distance/worst_case
    
    def get_route_duration(self):
        '''Returns the time between the start and the return at the depot'''
        
        if len(self.route) == 0:
            return 0
        
        self.duration = self.deliveries[0].arrival - self.deliveries[0].departure
        return self.duration
    
    def get_average_route_duration(self):
        '''Returns the route duration divided by the number of clients'''
        
        if len(self.route) == 0:
            return 0
            #, 'When developing, unclear when this situation would show up. Todo: check when you have an empty route and see if it suffices to set average route distance to 0'
        return self.duration / len(self.route)
     
    
    def get_fitting_candidates(self, deliveries):
        '''Checks for all the deliveries which are not in this vehicle how many of them fit'''
        
        other_deliveries = [d for d in deliveries if not d.customer in self.route]
        count = 0
        for d in other_deliveries:
            if d.customer.demand + self.load <= self.capacity:
                count += 1
        return count
    
    def expected_fit(self, deliveries):
        '''Checks for all the deliveries which are not in this vehicle what their average load is. 
        Then we calculate how many items with this expected load we can add'''
        
        other_deliveries = [d for d in deliveries if not d.customer in self.route]
        
        average_load = np.average([d.customer.demand for d in other_deliveries])
        
        return (self.capacity - self.load) / average_load 
    
    
    def get_copy(self):
        '''return copy'''
        
        # deliveries can be a shallow copy
        deliveries_copy = [copy.copy(delivery) for delivery in self.deliveries]
        
        result = vehicle(self.instance, self.nr, self.capacity, [c for c in self.route], distance=self.distance, load=self.load, first_deliveries=deliveries_copy)
        
        return result
    
    def distance_increase_by_adding_at(self, customer, i):
        '''Returns the distance_increase by adding customer at location i in the route'''
        
        cus_before_i = self.route[i] if i < len(self.route) else self.instance.depot
        cus_after_i = self.route[i-1] if i > 0 else self.instance.depot
        
        distance_increase = - self.instance.distance(cus_before_i, cus_after_i) 
        distance_increase += self.instance.distance(cus_before_i, customer) 
        distance_increase += self.instance.distance(customer, cus_after_i)
        
        #return np.round(distance_increase,4)
        return distance_increase
        
        
    def check_add_at(self, customer,i, allow_tw_violations):
        '''Check if we can add customer at location i in the route,
        returns boolean if the time windows allow this, and returns addition in distance'''
        
        distance_increase = self.distance_increase_by_adding_at(customer, i)
        cum_tw_violation = 0
        num_new_tw_violations = 0
        
        new_cus_arrival_time = self.deliveries[i].departure + self.instance.distance(self.deliveries[i].customer,customer)
        new_cus_waiting_time = max(0, customer.start - new_cus_arrival_time)
        
        #distance_increase += new_cus_waiting_time
    
        
        customers_are_delayed_by = distance_increase + new_cus_waiting_time + customer.service
        
        #also keep track how much the customer after the new customer is delayed
        next_delivery = self.deliveries[0] if (i+1) == len(self.deliveries) else self.deliveries[i+1]
        next_customer_delay = max( 0, customers_are_delayed_by - next_delivery.waiting_time)
        
        
        #Only check deliveries after i and the depot
        deliveries_to_check = self.deliveries[i+1:] + [self.deliveries[0]]
        
        for j,d in enumerate(deliveries_to_check):
            # if j <= i:
            #     continue
            
            if d.arrival + customers_are_delayed_by > d.customer.end:
                if not allow_tw_violations:
                    return False, distance_increase, j
                else:
                    #If customer is too late now, but was on time, then count here
                    num_new_tw_violations += d.is_on_time
                    cum_tw_violation += (d.arrival + customers_are_delayed_by - d.customer.end) - d.lateness_value
                    
                    
                    
            customers_are_delayed_by = max( 0, customers_are_delayed_by - d.waiting_time)
            
        # depot_delivery = self.deliveries[0]
        # if depot_delivery.arrival +  customers_are_delayed_by > depot_delivery.customer.end:
        #     return False, distance_increase, 0
        
        #Check if you arrive on time for customer
        if i > 0:
            if new_cus_arrival_time > customer.end:
                if not allow_tw_violations:
                    return False, distance_increase, -2
                else:
                    num_new_tw_violations += 1
                    cum_tw_violation += (new_cus_arrival_time - customer.end)
                    
        if not allow_tw_violations:
            return True, distance_increase, -1
        else:
            return num_new_tw_violations * self.instance.cost_per_violation, cum_tw_violation * time_window_violation_cost,distance_increase, next_customer_delay, -1
        
        
    def calc_vehicle_centroid(self):
        
        if len(self.route) == 0:
            self.centroid = self.instance.depot.x, self.instance.depot.y
            return
        
        sum_x = 0
        sum_y = 0
        for c in self.route:
            sum_x += c.x
            sum_y += c.y
            
        length = len(self.route)
        
        
        self.centroid = (sum_x/length, sum_y/length)
    
    def distance_to_route(self, other_route):
        '''return the distance between the centroids of the two routes'''
        
        return np.float32(np.sqrt(  math.pow(self.centroid[0]-other_route.centroid[0], 2) 
                                  + math.pow(self.centroid[1]-other_route.centroid[1],2)))
    
    def distance_to_route_tw(self, other_route):
        '''
        
        Create a distance for each shipment in the self route. Either the shipment:
            - has a 'long' tw, then take the distance to the centroid of the other route
            - has a 'short' tw, then determine by tw starts before which shipment from other route to put it, take distance from that shipment
        Return minimum over distances for shipmetns.'''
        
        assert len(other_route.route) > 0
        
        long_tw_border = 50
        
        distances = []
        # place_after_id = -1  #indicates depot
        place_before_id = 0  #indicates the index of the customer in other_route.route before which the shipment will be placed
        place_last = False
        for c in self.route:
            if c.tw_length() > long_tw_border:
                # dist_c_to_centroid = np.float32(np.sqrt(  math.pow(self.centroid[0]-c.x, 2) 
                #                                         + math.pow(self.centroid[1]-c.y,2)))
                
                dist_c_to_any_shipment = min( self.instance.distance(c, other_c) for other_c in other_route.route)
                distances.append(dist_c_to_any_shipment)
            else:
                
                while not place_last and c.tw_middle() > other_route.route[place_before_id].tw_middle():
                    place_before_id += 1
                    
                    if place_before_id == len(other_route.route):
                        place_last = True
                
                if place_last:
                    place_before = self.instance.depot
                else:
                    place_before = other_route.route[place_before_id]
                    
                
                dist_c_to_place_before = self.instance.distance(c, place_before)
                distances.append(dist_c_to_place_before)
        # return np.average(sorted(distances)[:3])
        return min(distances)
                
    
    def distance_to_customer(self, customer):
        '''return the distance between the centroid of the route and the location of the customer'''
        
        return np.float32(np.sqrt(  math.pow(self.centroid[0]-customer.x, 2) 
                                  + math.pow(self.centroid[1]-customer.y,2)))
    
    def distance_to_customer2(self, customer):
        
        
        return min(self.instance.distance(c,customer) for c in self.route)
    
    def greedy_addition_cost(self, customer):
        '''Decide position to place the customer based on tw only.
        Return the increase in solution value if customer is placed in that position'''
        
        # if self.load + customer.demand > self.capacity:
        #     return np.infty, np.infty, None, None
        
        depot = self.instance.depot
        place_after = depot
        for c in self.route:
            if customer.start < c.start:
                place_before = c
                break
            place_before = depot
            place_after = c
            
        #place_after_delivery = self.get_delivery(place_after)
        distance_place_after_to_customer = self.instance.distance(place_after, customer) 
        
        #arrival_at_customer = place_after_delivery.departure + distance_place_after_to_customer
        #will_customer_be_in_time = arrival_at_customer < customer.end
        # if not will_customer_be_in_time:
        #     return np.infty, np.infty, None, None
        
        distance_gain = - self.instance.distance(place_after, place_before) 
        distance_increase = distance_place_after_to_customer
        distance_increase += self.instance.distance(customer, place_before)
        
        distance_delta = distance_gain + distance_increase
        
        waiting_time = 0 #max(customer.start - arrival_at_customer, 0 )
        is_delay_possible = self.get_delivery(place_before).possible_delay > distance_delta + c.service + waiting_time
        
        solution_value_increase = distance_increase + distance_gain
        
        solution_value_increase_possible = solution_value_increase if is_delay_possible else self.instance.depot.end
        
        return solution_value_increase, solution_value_increase_possible, place_after, place_before
        
class delivery(object):
    '''The object which links a customer to a vehicle in a solution. 
    In this object, the arrival and departure are stored, and the lateness and waiting time are calculated'''
    def __init__(self, customer, arrival, departure, service_start, waiting_time, lateness_value):
        #self.vehicle = vehicle
        self.customer = customer
        self.arrival = arrival
        self.departure = departure
        self.service_start = service_start
        # self.lateness_value = max(0, arrival - customer.end
        self.lateness_value = lateness_value
            
        #assert customer.nr == 0 or lateness_value == max(0, arrival - customer.end), f'{lateness_value}, {arrival}, {customer.end}'
        self.lateness_cost = self.lateness_value * time_window_violation_cost
        # self.waiting_time = max(0, customer.start - arrival)
        self.waiting_time = waiting_time
        # if self.customer.nr > 0 and not self.waiting_time ==  max(0, customer.start - arrival):
        #     print(self.customer)
        # assert self.customer.nr == 0 or self.waiting_time == max(0, customer.start - arrival), f'{waiting_time}, {arrival}, {customer.start}'
        self.is_on_time = self.lateness_cost == 0
        
        self.possible_delay = self.customer.end - self.arrival
        
    def update_arrival(self, arrival):
        self.arrival = arrival
        self.lateness_value = max(0, arrival - self.customer.end)
        self.lateness_cost = self.lateness_value * time_window_violation_cost
        self.waiting_time = max(0, self.customer.start - arrival)
        self.is_on_time = self.lateness_cost == 0
        self.possible_delay = self.customer.end - self.arrival
        

class customer(object):
    def __init__(self, nr, x, y, demand, start, end, service):
        self.nr = nr
        self.x = x
        self.y = y
        self.demand = demand
        self.start = start
        self.end = end
        self.service = service
    def __repr__(self):
        return "c%d" %(self.nr)
        #return "c%d (%d)" %(self.nr, self.demand)
        
    def tw_length(self):
        return self.end - self.start
    
    def tw_middle(self):
        return self.start + self.tw_length()/2
        
class instance(object):
    def __init__(self, name, depot, customers, nr_vehicles, cap_vehicles, restrict_nr_vehicles):
        self.name = name
        self.depot = depot
        self.customers = customers
        self.nr_customers = len(customers)
        self.nr_vehicles = nr_vehicles
        self.non_empty_vehicles = nr_vehicles
        self.capacity = cap_vehicles
        self.calculate_distances()
        self.restrict_nr_vehicles = restrict_nr_vehicles
        self.time_window_violation_cost = time_window_violation_cost
        self.cost_per_violation = cost_per_violation
        
    def __repr__(self):
        result = "Instance: " + self.name + "\n"
        result += "Vehicles: " + repr(self.nr_vehicles) + ", capacity: " + repr(self.capacity) + "\n"
        for c in self.customers + [self.depot]:
            result += repr(c) + "\n"
        return result
    
    def calculate_distances(self):
        result = np.zeros((len(self.customers)+1, len(self.customers)+1))
        for i,c1 in enumerate([self.depot] + self.customers):
            for j,c2 in enumerate([self.depot] + self.customers):
                if j > i:
                    continue
                #result[i,j] = result[j,i] = np.round(np.sqrt(math.pow(c1.x-c2.x, 2) + math.pow(c1.y-c2.y,2)),4)
                result[i,j] = result[j,i] = np.float32(np.sqrt(math.pow(c1.x-c2.x, 2) + math.pow(c1.y-c2.y,2)))
        max_distance = np.max(result)
        self.__distances = tuple(tuple(c) for c in result)
        self.__normalized_distances = tuple(tuple(c/max_distance) for c in result)
        self.calculate_temporal_distances()
        self.calculate_sorted_distances()
        
    def calculate_temporal_distances(self):
        '''Defined as the normal distance + the time window difference'''
        result = np.zeros((len(self.customers)+1, len(self.customers)+1))
        for i,c1 in enumerate([self.depot] + self.customers):
            for j,c2 in enumerate([self.depot] + self.customers):
                if j > i:
                    continue
                distance = self.distance(c1,c2)
                result[i,j] = result[j,i] = distance + self.tw_difference(c1,c2, distance)
                
        max_temporal_distance = np.max(result)
        self.__temporal_distances = tuple(tuple(c) for c in result)
        self.__normalized_temporal_distances = tuple(tuple(c/max_temporal_distance) for c in result)
    
    def calculate_sorted_distances(self):
        result = {}
        result_temporal = {}
        locations = ([self.depot] + self.customers)
        for c in locations:
            
            distances = [(n, self.distance(c,n)) for n in locations]
            distances.sort(key = lambda x : x[1])
            result[c] = distances
            
            distances_temporal = [(n, self.temporal_distance(c,n)) for n in locations]
            distances_temporal.sort(key = lambda x : x[1])
            result_temporal[c] = distances_temporal
            
        result = tuple(result[c] for c in locations)
        result_temporal = tuple(result_temporal[c] for c in locations)
        
        self.__sorted_distances = result
        self.__sorted_temporal_distances = result_temporal
        
    def distance(self, c1, c2):
        return self.__distances[c1.nr][c2.nr]
    
    def normalized_distance(self, c1,c2):
        return self.__normalized_distances[c1.nr][c2.nr]
    
    def sorted_distances(self, c, temporal=False):
        if temporal:
            return self.__sorted_temporal_distances[c.nr]    
        else:
            return self.__sorted_distances[c.nr]
    
    def tw_difference(self, c1, c2, distance = None):
        '''We check the two orders of the clients, visiting c1 first and visiting c2 first. 
        We decide for both combinations how much the other client is too late if you would travel from earliest client at its earliest end to other.
        Also check what the minimum waiting time would be if you would travel from earliest client at its latest end to other
        If the distance between the clients is known, it can be given'''
        
        c1_is_earlier = c1.start + c1.service <= c2.start + c2.service
        early_client, late_client = (c1,c2) if c1_is_earlier else (c2,c1)
        
        if not distance:
            distance = self.distance(early_client, late_client)
        
        earliest_possible_arrival_at_late = early_client.start + early_client.service +  distance
        latest_possible_arrival_at_late =   early_client.end   + early_client.service + distance
        minimum_lateness_1 =     max(0, earliest_possible_arrival_at_late - late_client.end)
        minimum_waiting_time_1 = max(0, late_client.start - latest_possible_arrival_at_late)
        
        early_client, late_client = (c1,c2) if not c1_is_earlier else (c2,c1)
        
        
        earliest_possible_arrival_at_late = early_client.start + early_client.service +  distance
        latest_possible_arrival_at_late =   early_client.end   + early_client.service + distance
        minimum_lateness_2 =     max(0, earliest_possible_arrival_at_late - late_client.end)
        minimum_waiting_time_2 = max(0, late_client.start - latest_possible_arrival_at_late)
        
        allow_lateness_in_puzzle = False
        if allow_lateness_in_puzzle == False:
            # Here, we don't allow lateness in puzzle,
            # So if there is a case in which the lateness is positive, we increase the tw difference a lot.
            if minimum_lateness_1 > 0: 
                minimum_lateness_1 = self.depot.end
            if minimum_lateness_2 > 0:
                minimum_lateness_2 = self.depot.end
        
        return min(minimum_lateness_1 + minimum_waiting_time_1, minimum_lateness_2 + minimum_waiting_time_2)
    
    def temporal_distance(self, c1, c2):
        '''Both the actual distance as the difference in time windows are taken in consideration.'''
        #distance = self.distance(c1,c2)
        return self.__temporal_distances[c1.nr][c2.nr]
    
    def normalized_temporal_distance(self, c1, c2):
        '''Returns the temporal_distance divided by the greatest temporal distance'''
        return self.__normalized_temporal_distances[c1.nr][c2.nr]
        
    def customer_by_nr(self, customer_nr):
        for customer in self.customers:
            if customer.nr == customer_nr:
                return customer
        if customer_nr == 0:
            return self.depot
        assert False, f'customer {customer_nr} not found in instance'
        return None    


    def write(self, filename):
        '''Write current instance to a new file'''
        assert not path.exists(filename), f'filename for instance {filename} already exists'
        file = open(filename, 'w')
        
        file.write(self.name + "\n")
        file.write("\n")
        file.write("VEHICLE\n")
        file.write("NUMBER\tCAPACITY\tWORLD_RECORD\n")
        file.write(f'{self.nr_vehicles}\t{self.capacity}\t{self.nr_vehicles}\n')
        file.write("\n")
        file.write("\n")
        file.write("CUSTOMER\n")
        file.write("CUST NO.   XCOORD.   YCOORD.    DEMAND   READY TIME   DUE DATE   SERVICE TIME\n")
        file.write("\n")
        file.write(f"{self.depot.nr}\t{self.depot.x}\t{self.depot.y}\t{self.depot.demand}\t{self.depot.start}\t{self.depot.end}\t{self.depot.service}\n")
        for c in self.customers:
            file.write(f"{c.nr}\t{c.x}\t{c.y}\t{c.demand}\t{c.start}\t{c.end}\t{c.service}\n")
        file.close()   
        
    def calculate_route_distance(self, route):
        '''Instead of creating a whole vehicle object, sometimes convenient to just calculate the distance of a route
        That is what this method returns'''
        
        result = 0
        depot = self.depot
        u = depot
        for c in route:
            v = c
            result += self.distance(u,v)
            u = v
        result += self.distance(u,depot)
        return result
        
class solution(object):
    def __init__(self, instance, vehicles, check_solution = True):
        self.instance = instance
        self.vehicles = vehicles
        #self.calculate_distance()
        #self.calculate_trucks()
        #self.total_value = self.total_distance + self.total_tw_violation
        self.update_cost(check_solution)
        # if check_solution:
        #     self.check_solution()
        self.non_empty_vehicles = sum([len(v.route)>0 for v in self.vehicles])
        
    @classmethod
    def from_file(cls, instance, filename, change_instance=False):
        
        file = open(filename, 'r')
        lines = file.readlines()
        v_index = 0
        vehicles = []
        for line in lines:
            if not line[0]=='v':
                continue
            v_index += 1 #start with index 1
            customers_string = line[line.find("[")+1:line.find("]")]
            customers = [instance.customer_by_nr(int(cus_string.strip(",")[1:])) for cus_string in customers_string.split()]
            v = vehicle(instance, v_index, instance.capacity, customers)
            vehicles += [v]
        if change_instance:
            print(f'There are less vehicles in the initial solution ({len(vehicles)}) than in the instance ({instance.nr_vehicles}). The instance will be changed to match the initial solution')
            instance.nr_vehicles = len(vehicles)
        file.close()
        return cls(instance, vehicles)
    
    @classmethod
    def from_sintef_file(cls, instance, filename):
        
        file = open(filename, 'r')
        lines = file.readlines()
        v_index = 0
        vehicles = []
        for line in lines:
            line = line.split()
            if not line[0]=='Route':
                continue
            v_index += 1 #start with index 1
            customer_nrs_string = line[3:]
            customers = [instance.customer_by_nr(int(cus_string)) for cus_string in customer_nrs_string]
            v = vehicle(instance, v_index, instance.capacity, customers)
            vehicles += [v]
        file.close()
        return cls(instance, vehicles)
    
    @classmethod
    def from_vroom_json_file(cls, instance, filename):
        with open(filename, 'r') as json_file:
            
            sol_dic = json.load(json_file)
        
        v_index = 0
        vehicles = []
        
        # print(sol_dic['routes'])
        for veh_dic in sol_dic['routes']:
            v_index += 1 #start with index 1
            
            # print(veh_dic['steps'])
            # print(veh_dic['steps'])sin
            route = [instance.customers[step['job']-1] for step in veh_dic['steps'] if step['type']=='job']
            # if v_index == 1:
            #     route += [instance.customers[7], instance.customers[64]]
            
            
            v = vehicle(instance, v_index, instance.capacity, route)
            vehicles += [v]
            
        #If there are less vehicles in the VROOM solution, then we will change the instance to match this
        if instance.nr_vehicles > len(vehicles):
            print(f'There are less vehicles in the vroom initial solution ({len(vehicles)}) than in the instance ({instance.nr_vehicles}). The instance will be changed to match the vroom solution')
            instance.nr_vehicles = len(vehicles)
        return cls(instance, vehicles, True)
        

        
    def update_cost(self, check=True):
        
        
        for v in self.vehicles:
            if not self.instance.restrict_nr_vehicles and len(v.route) < 1:
                self.vehicles.remove(v)
            v.update()
            
        
        if self.instance.restrict_nr_vehicles:
            self.total_tw_violation = sum([v.tw_violation for v in self.vehicles])    
            self.nr_tw_violations = sum([v.nr_tw_violations for v in self.vehicles])
            self.total_tw_violation_nr = self.nr_tw_violations * self.instance.cost_per_violation
    
        self.total_distance = sum([v.distance for v in self.vehicles])
        
        self.non_empty_vehicles = sum([len(v.route)>0 for v in self.vehicles])
        
        #self.calculate_distance()
        #self.calculate_trucks()
        self.total_value = self.total_distance + self.total_tw_violation + self.total_tw_violation_nr
        if check:
            self.check_solution()
    
    def check_solution(self):
        is_correct = True
        msg = "\n"
        
        #Check if all customers only once in solution
        for c in [c for c in self.instance.customers]:
            vehicles_containing_c = [v for v in self.vehicles if c in v.route]
            
            if len(vehicles_containing_c) == 0:
                is_correct = False
                msg += repr(c) + " is not in the solution\n"
            if len(vehicles_containing_c) > 1:
                is_correct = False
                msg += repr(c) + " is " + repr(len(vehicles_containing_c)) + " times in the solution: ("
                msg += ", ".join([repr(v) for v in vehicles_containing_c]) + ")\n"
                
            
        #Check if no truck exceeds capacity
        for v in self.vehicles:
            if v.load > v.capacity:
                is_correct = False
                msg += repr(v) + " has load " + repr(v.load) + " but capacity " + repr(v.capacity) + "\n"
        
        #Check if each truck different vehicle number:
        vehicle_numbers = [v.nr for v in self.vehicles]
        if not len(vehicle_numbers) == len(set(vehicle_numbers)):
            is_correct = False
            msg += "Vehicle numbers contain duplicate: " + repr(vehicle_numbers) + "\n"
            
        #Check time windows
        if not self.instance.restrict_nr_vehicles: 
            #if nr of vehicles are restricted, then time windows are penalized but allowed
            for v in self.vehicles:
                is_correct, msg_tw = v.check_time_windows()
                msg += msg_tw
        
        #Check nr of vehicles
        if self.instance.restrict_nr_vehicles and not len(self.vehicles) == self.instance.nr_vehicles:
            is_correct = False
            msg += "There are only " + repr(len(self.vehicles)) + " vehicles in the solution, while there should be " + repr(self.instance.nr_vehicles)
        
        #Check if nr of customers in solution equals number of customers in instance
        if not len(self.instance.customers) == sum([len(v.route) for v in self.vehicles]):
            is_correct = False
            msg += f"Number of customers in instance and solution are not equal. There are {len(self.instance.customers)} customers in the instance, but there are {sum([len(v.route) for v in self.vehicles])} customers in the solution"
            
        #Check if all customers in solution are customers from instance (and not some copy)
        #You need this because otherwise methods won't work anymore
        for v in self.vehicles:
            for c in v.route:
                if not c in self.instance.customers:
                    is_correct = False
                    msg += f"The customer {c} is in the solution but is not in the instance. This might be because of copying the solution or instance" 
        
        assert is_correct, msg + "\n" + repr(self)
    
    def __repr__(self, more_info=False):
        result = f"Value: {self.total_value:.3f} Distance: {self.total_distance:.3f}\tViolation cost: {self.total_tw_violation:.3f}\tnr violations: {self.nr_tw_violations:} violation cost:\t{time_window_violation_cost}\n"
        for v in self.vehicles:
            #result += repr(v) + ":\t" f"{v.distance:6.2f}\t{v.tw_violation:8.2f}\t" + repr(v.route) + repr([v.get_delivery(c).lateness_value for c in v.route]) + "\n"
            #late_customers = [(c, v.get_delivery(c).lateness_value) for c in v.route if not v.get_delivery(c).is_on_time]
            late_customers = [(d.customer, d.lateness_value) for d in v.deliveries if not d.is_on_time]
            
            result += repr(v) + ":\t" f"{v.distance:6.2f}\t{v.tw_violation:8.2f}\t" + repr(v.route) 
            if len(late_customers) > 0:
                result +=  "\t\t\t late(ness) customers: " + repr(late_customers)
            result += "\n"
        return result
    
    def get_copy(self, check_solution=True):
        '''Creates a copy somewhere in between deep and shallow copy.
        Vehicles are made copy of, but customers and instance are taken the same.'''
        
        vehicles = []
        for v in self.vehicles:
            vehicles += [v.get_copy()]
        
        sol_copy = solution(self.instance, vehicles, check_solution)
        #todo: later, after testing, we don't need to check solution everytime here
        return sol_copy
        
    def print_extensive(self):
        
        result = repr(self)
        result += "\n\nformat: c1 [service_start]---s:(service_duration)---[departure]---d:distance c1 to c2--- and in case waiting time: [arrival c2]---w:(waiting time)---"
        for v in self.vehicles:
            result += repr(v) + ":\t" f"{v.distance:6.2f}\t{v.tw_violation:8.2f}\t" + repr(v.route) + "\n"
            result += v.schedule_string() + "\n\n"
        print(result)   
        
    def is_better_than(self, solution2):
        '''Changed into: only check difference in sum of violation + distance'''
        '''Also accept if is equal to'''
        if len(self.vehicles) < len(solution2.vehicles) :
            return True
        if self.instance.restrict_nr_vehicles:
            return self.total_value <= solution2.total_value
        #if self.instance.restrict_nr_vehicles and self.total_tw_violation < solution2.total_tw_violation:
        #    return True
        #elif self.instance.restrict_nr_vehicles and self.total_tw_violation > solution2.total_tw_violation:
        #    return False
        return self.total_distance < solution2.total_distance
        
    def remove(self, customer):
        for v in self.vehicles:
            if customer.nr in [c.nr for c in v.route]:
                c = next(c for c in v.route if c.nr == customer.nr)
                v.route.remove(c)
                v._needs_update = True
                if not self.instance.restrict_nr_vehicles and len(v.route) == 0:
                    self.vehicles.remove(v)
                self.update_cost(False)
                return
            
    def find_first_free_veh_nr(self):
        for i, v in enumerate(self.vehicles):
            if v.nr == i + 1: #customer with index i should have nr i+1
                continue
            return i
        return len(self.vehicles) +1
    
    def find_vehicle(self, cus):
        if cus == self.instance.depot:
            assert False, 'Cannot find vehicle for depot, since the depot is in every vehicle'
            return -1
        for v in self.vehicles:
            if cus.nr in [c.nr for c in v.route]:
                return v
        assert False, f"customer {cus} not found in solution"
        return -1
    
    def find_delivery(self, customer):
        if customer == self.instance.depot:
            assert False, 'Cannot find delivery for depot, since the depot has many deliveries'
            return -1
        return self.find_vehicle(customer).get_delivery(customer)
    
    def get_min_greedy_addition_cost(self, c, other_vehicles, dist_contribution, return_vehicle=False):
        if return_vehicle:
            values_after_before = [v.greedy_addition_cost(c) for v in other_vehicles]
            values = [x for x,_,_,_ in values_after_before]
            gain_if_possible = [max(dist_contribution - x,0) for _,x,_,_ in values_after_before]
            place_between = [(y,z) for _,_,y,z in values_after_before]
            
            index_min = values.index(min(values))
            index_max_if_possible = gain_if_possible.index(max(gain_if_possible))
            
            return values[index_min], other_vehicles[index_min], place_between[index_min], gain_if_possible[index_max_if_possible], other_vehicles[index_max_if_possible], place_between[index_max_if_possible]
        else:
            return min(v.greedy_addition_cost(c)[0] for v in other_vehicles), max(max(dist_contribution - v.greedy_addition_cost(c)[1],0) for v in other_vehicles)
                
    
    def get_distance_contribution(self, customer):
        if customer == self.instance.depot:
            assert False, 'Cannot calculate distance contribution for depot'
        vehicle = self.find_vehicle(customer)
        return vehicle.get_distance_contribution(customer)
    
    def get_total_cost_contribution(self, customer):
        if customer == self.instance.depot:
            assert False, 'Cannot calculate total cost contribution for depot, since different per vehicle'
        vehicle = self.find_vehicle(customer)
        delivery = vehicle.get_delivery(customer)
        distance_contribution = vehicle.get_distance_contribution(customer) 
        tw_contribution = delivery.is_on_time * cost_per_violation + delivery.lateness_cost
            
        return distance_contribution + tw_contribution
        
    def get_average_route_distance(self, customer):
        if customer == self.instance.depot:
            assert False, 'Cannot calculate average route distance for depot'
        vehicle = self.find_vehicle(customer)
        return vehicle.get_average_route_distance()
        
    def write(self, name, filename=None):
        #Write solution to file. Give a name for the solution and a filename will be generated, or giva a spcific filename
        filefolder = "..//Data//Solutions//"
        if not filename:
            filename = f"{self.instance.name}-{len(self.instance.customers)}_{name}_{self.total_distance + self.total_tw_violation:.0f}.txt"
        file = open(filefolder+filename, "w")
        file.write(repr(self))
        file.close()