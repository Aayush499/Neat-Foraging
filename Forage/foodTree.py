import random 
import math

def foodGenerator(root_coords, num_of_food, sensor_length ):
    num_sensors = 8

    agent_pos = root_coords
    first_food = random.randint(0,7)
    
    #coordinate of first food 

    angle = 2 * math.pi * first_food / num_sensors
    x1 = agent_pos[0] + sensor_length * math.cos(angle)
    y1 = agent_pos[1] + sensor_length * math.sin(angle)

    food_coords = [(x1,y1)]

    #each food item is a node and each node can have children. track each node's children and what sensor was used to generate each child
    food_children = {}
    #keep track of viable parents. these are nodes with less than 3 children that can have more children. Also keep track of what sensor this was made on
    viable_parents = [(x1,y1,first_food)]
    edges = []

    while len(food_coords) < num_of_food:
        #pick a random viable parent and update agent position as this coordinate and store the sensor that was used to generate this food
        _ = viable_parents[random.randint(0,len(viable_parents)-1)]
        agent_pos = (_[0], _[1])
        sensor = _[2]
        #if this parent has children, check if either sensors sensor, sensor+1 and sensor-1 are available and store them in a list
        available_sensors = []
        for i in range(-1,2):
            #check is sensor + i is occupied by any of the food object's children
            #iterate through food_children[agent_pos] and check if sensor + i is occupied
            occupied = False
            if agent_pos in food_children:
                for child in food_children[agent_pos]:
                    if (child[2] + i) % num_sensors == sensor:
                        occupied = True
                        break
            if not occupied:
                available_sensors.append((sensor + i) % num_sensors)

        #if there are available sensors, pick one at random and generate a new food object
        if len(available_sensors) > 0:
            new_sensor = available_sensors[random.randint(0,len(available_sensors)-1)]
            angle = 2 * math.pi * new_sensor / num_sensors
            x1 = agent_pos[0] + sensor_length * math.cos(angle)
            y1 = agent_pos[1] + sensor_length * math.sin(angle)
            food_coords.append((x1,y1))
            edges.append((agent_pos,(x1,y1)))
            viable_parents.append((x1,y1,new_sensor))
            if agent_pos in food_children:
                food_children[agent_pos].append((x1,y1,new_sensor))
            else:
                food_children[agent_pos] = [(x1,y1,new_sensor)]
            edges.append((agent_pos,(x1,y1)))
            if len(food_children[agent_pos]) == 3:
                viable_parents.remove((agent_pos[0],agent_pos[1],sensor))
        
        
    edges.append((root_coords, food_coords[0]))
    return food_coords, edges