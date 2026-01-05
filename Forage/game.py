from .agent import Agent
from .food import Food
from .obstacle import Obstacle
from .pheromone import Pheromone
from .nest import Nest
from .foodTree import foodGenerator
from .optimalPath import bestPath
from .optimalPath import bestPathPerNode
import pygame
import random
import math
import numpy
pygame.init()


class GameInformation:
    def __init__(self, score, food_collected, total_food, food_list):
        self.score = score
        self.food_collected = food_collected
        self.total_food = total_food
        self.food_list = food_list
        


class Game:
    """
    To use this class simply initialize and instance and call the .loop() method
    inside of a pygame event loop (i.e while loop). Inside of your event loop
    you can call the .draw() and .move_agent() methods according to your use case.
    Use the information returned from .loop() to determine when to end the game by calling
    .reset().
    """
    SCORE_FONT = pygame.font.SysFont("comicsans", 50)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    def __init__(self, window, window_width, window_height, decay_factor, arrangement_idx, obstacles , particles , ricochet, obstacle_type, seeded, o_switch, discount_factor = 0.99, pheromone_receptor=True, collision_threshold=7.0, time_constant=200.0, time_bonus_multiplier=1.0,  teleport=False, num_sensors=8, food_calibration=True, sparse = False, movement_type="holonomic", carrying_food_receptor=True, nest_receptor=True):
         
      
        
        
        self.teleport = teleport
        self.window_width = window_width
        self.window_height = window_height

        self.agent = Agent(
            self.window_height // 2 , self.window_width // 2, pheromone_receptor=pheromone_receptor, num_sensors=num_sensors, carrying_food_receptor=carrying_food_receptor, nest_receptor=nest_receptor)
        self.total_food = particles
        # self.total_food = 2
        food_pos = []
        buffer_distance = self.agent.sensor_length  # minimum Euclidean distance from agent (prevents spawn overlap)
        
        if seeded:
            random.seed(arrangement_idx)
            numpy.random.seed(arrangement_idx)

        # while len(food_pos) < self.total_food:
        #     x = random.randint(400, 600)
        #     y = random.randint(100, 250)
        #     # Prevent food from spawning on top of the agent
        #     dist = ((self.agent.x - x)**2 + (self.agent.y - y)**2)**0.5
        #     if dist > buffer_distance:
        #         food_pos.append((x, y))

        food_pos = []
        # while len(food_pos) < self.total_food:
        #     angle = random.uniform(0, 2 * math.pi)
        #     x = self.agent.x + self.agent.sensor_length * math.cos(angle)
        #     y = self.agent.y + self.agent.sensor_length * math.sin(angle)
        #     dist = ((self.agent.x - x) ** 2 + (self.agent.y - y) ** 2) ** 0.5
        #     if dist > buffer_distance:
        #         food_pos.append((x, y))
        
        prev_node = (self.agent.x, self.agent.y)
        angle = random.uniform(0, 2 * math.pi) if o_switch else math.pi/2
        #let the dist of food after the first food be sensor lenght - collision threshold if calibrated food check is true
        food_dist = self.agent.sensor_length - (collision_threshold+2) if food_calibration else self.agent.sensor_length
        first_chk = True
        food_placed = False
        dist = self.agent.sensor_length
        # dist = 30

        radial_arrangement = False
        for f in range(self.total_food):
            #generate an arrangement of food in a line
            if not first_chk:
                dist = food_dist
            x = prev_node[0] + dist * math.cos(angle)
            y = prev_node[1] + dist * math.sin(angle)
            food_pos.append((x, y))
            
            angle_variation = random.uniform(-math.pi/2, math.pi/2) if o_switch else (40/180)*math.pi
            if not radial_arrangement:
                prev_node = (x, y)
            
            first_chk = False
            food_placed = True
            angle += angle_variation if  food_placed else 0

        # self.food_list = [Food(x, y) for x, y in food_pos]
        # food_pos = arrangements[arrangement_idx]
        self.food_list = [Food(x, y) for x, y in food_pos]
        

        self.collision_threshold = collision_threshold
        self.time_constant = time_constant
        # self.time_bonus_multiplier = time_bonus_multiplier

        # self.optimalTime = 250
        # self.TIME_BONUS =  2*self.agent.sensor_length/self.agent.vel + 2 + 20
        # self.time_constant = (self.agent.sensor_length/self.agent.vel)*3 + 20
        # self.optimalTime = self.time_constant 
        self.optimalTime = self.time_constant 
        #calculate optimal distance to collect one food particle and return to nest
        self.distance_constant = self.agent.sensor_length* 2.5 # buffer of 1.6
        self.optimal_distance =  2*self.distance_constant# approx distance for one food particle collection and return
        
        
        self.score = 0
        self.food_collected = 0
        self.pheromones = []
        self.nest = Nest(self.agent.x, self.agent.y)
        self.window = window
        self.current_direction = ""

        self.discount_factor = discount_factor
        self.carry_time = 0
        self.searching_time = 0
        
        self.obstacles = [
            # Obstacle(300, 450, 300, 300)
            # Obstacle(200, 300, 700, 300)
            # Obstacle(600, 300, 600, 450)
            
        ]

        #choose a random angle from 0 to 2pi for the agent
        # if o_switch:
        #     self.agent.theta = random.uniform(0, 2*math.pi)
        # self.agent.theta = (arrangement_idx % 4) * (math.pi/2)
        if obstacles:
            if obstacle_type == "angular":
                if arrangement_idx % 3 == 0:
                    self.obstacles += [
                        # Obstacle(300, 450, 300, 300),
                        Obstacle(400, 400, 500, 300),
                        Obstacle(500, 300, 600, 400)
                    ]
                elif arrangement_idx % 3 == 1:
                    self.obstacles += [
                        Obstacle(450,475, 550,375),
                        Obstacle(550,375, 650,475)
                    ]
                elif arrangement_idx % 3 == 2:
                    self.obstacles += [
                        Obstacle(350,475, 450,375),
                        Obstacle(450,375, 550,475)
                    ]
            elif obstacle_type == "line":
                self.obstacles += [
                    Obstacle(400, 400, 600, 400)
                ]

        self.collision_occurred = False

        self.ricochet = ricochet

        self.simple_guidance = False

        self.decay_factor = decay_factor
        self.decay_limit = 0.0001
        pheromone_lifespan = math.log(self.decay_limit)/math.log(self.decay_factor)
        # self.sum_limit = (1 - self.decay_factor**pheromone_lifespan)/(1 - self.decay_factor)

        self.sparse = sparse

        self.milestone  =1

        self.food_seek_reward = 1
        self.nest_seek_reward = 1
        self.general_movement_reward_after_food_collection = .01
        self.persistent_movement_reward = 1

        

        self.movement_type = movement_type

        self.clock =0
        self.pulse_interval = 1  # frames till food removal

        self.total_distance_travelled = 0.0

        self.prev_pos = (self.agent.x, self.agent.y)

        self.path_milestones = []
        for idx, food in enumerate(self.food_list):
            self.path_milestones.append(('food', idx))
            self.path_milestones.append(('nest', None))

        self.current_milestone_idx = 0
        self.path_reward = 20.0      # tune as needed
        self.path_radius = self.collision_threshold * 1.5

         
    def _current_path_target(self):
        if self.current_milestone_idx >= len(self.path_milestones):
            return None  # path complete

        kind, idx = self.path_milestones[self.current_milestone_idx]
        if kind == 'nest':
            return (self.nest.x, self.nest.y)
        else:  # 'food'
            # food may be removed; if gone, just skip this milestone
            if idx >= len(self.food_list):
                return None
            f = self.food_list[idx]
            return (f.x, f.y)
        

    def _draw_score(self):
        score_text = self.SCORE_FONT.render(
            f"{self.score}", 1, self.WHITE)
        
        direction_text = self.SCORE_FONT.render(
            f"{self.current_direction}", 1, self.WHITE)
         
        self.window.blit(score_text, (self.window_width //
                                           4 - score_text.get_width()//2, 20))
        
        self.window.blit(direction_text, (self.window_width //
                                             4 - direction_text.get_width()//2, 60))
        
    
        

        
        #draw time remaining
        # time_remaining = max(0, int(self.optimalTime - (self.searching_time + self.carry_time)))
        # time_text = self.SCORE_FONT.render(
        #     f"Time: {time_remaining}", 1, self.WHITE)
        # self.window.blit(time_text, (self.window_width //
        #                                4 - time_text.get_width()//2, 100))  



    def _handle_collision(self):
       
        agent = self.agent
        nest = self.nest
        milestone = True
        # collision_threshold = self.agent.vel*1.5
        collision_threshold = self.collision_threshold
        if not agent.carrying_food:
            self.searching_time += 1
            for food in self.food_list:
                dist = ((agent.x - food.x) ** 2 + (agent.y - food.y) ** 2) ** 0.5
                # if dist < agent.radius + food.radius:
                if dist <= collision_threshold :
                    # small reward for picking up food
                    # calculate the score based on how quickly the agent picks up the food
                    if self.movement_type != "1d":
                        if not self.sparse :
                            
                            # self.score += 100 * (self.discount_factor ** self.searching_time) + self.food_collected*30*(self.discount_factor ** self.searching_time)
                            # self.score += 10**(self.milestone) * (self.discount_factor ** self.searching_time) 
                            self.score += 10 * (self.discount_factor ** self.searching_time) 
                        else:
                            self.score += 10**(self.milestone)
                        if milestone:
                            self.milestone += 1
                    
                    if self.teleport:
                        self.agent.x = food.x
                        self.agent.y = food.y
                    agent.carrying_food = True
                    self.carry_time = 0
                    self.food_list.remove(food)
                    
                    break   
        else:
          #heavy reward for bringing food to the nest, scaled by how quickly the agent does it
            self.carry_time += 1
            dist = ((agent.x - nest.x) ** 2 + (agent.y - nest.y) ** 2) ** 0.5
            # if dist < agent.radius + nest.radius:
            if dist <= collision_threshold :
                if not self.sparse:
                    # self.score += 150 * (self.discount_factor ** self.carry_time) + self.food_collected*200*(self.discount_factor ** self.carry_time)
                    # self.score += 10**(self.milestone) * (self.discount_factor ** self.carry_time)
                    self.score += 50 * (self.discount_factor ** self.carry_time)
                else:
                    self.score +=  10**(self.milestone)
                if milestone:
                    self.milestone += 1
                self.food_collected += 1   
                agent.carrying_food = False
                if self.food_collected == self.total_food:
                    self.score += 1  # bonus for collecting all food
                #remove all pheromones when food is delivered to nest
                # self.pheromones = []
                self.searching_time = 0
                # self.optimalTime += self.time_constant*2
                self.optimalTime += self.time_constant*(self.food_collected+1)*(self.food_collected+2)
                self.optimal_distance = self.distance_constant*(self.food_collected+1)*(self.food_collected+2)
        
        
            
        

    def update_sensor_data(self):

        agent = self.agent
        nest = self.nest
        
        for i in range(agent.sensor_count):
            receptor_cnt = 0
            pheromone_signal = 0
            if self.agent.pheromone_receptor:
                for pheromone in self.pheromones:
                    dist = ((agent.x - pheromone.x) ** 2 + (agent.y - pheromone.y) ** 2) ** 0.5
                    # if dist < agent.sensor_length, add the strength of the pheromone to the sensor signal
                    if dist <= agent.sensor_length:
                        angle_to_pheromone = math.atan2(pheromone.y - agent.y, pheromone.x - agent.x)
                        sensor_angle = 2 * math.pi * i / agent.sensor_count + agent.theta
                        angle_diff = abs((angle_to_pheromone - sensor_angle + math.pi) % (2 * math.pi) - math.pi)
                        if angle_diff < (math.pi / agent.sensor_count):
                            # pheromone_signal += pheromone.strength * (1 - (dist / (agent.sensor_length + 0.1)))
                            pheromone_signal += pheromone.strength / (dist + 0.1)
                            # print(f"Pheromone detected at Sensor {i}, {dist} distance away")
                # agent.sensors[i][0] = 2*pheromone_signal/self.sum_limit - 1
                agent.sensors[i][receptor_cnt] = pheromone_signal
                receptor_cnt += 1

            # for food in self.food_list:
            #     dist = ((agent.x - food.x) ** 2 + (agent.y - food.y) ** 2) ** 0.5
            #     if dist < agent.sensor_length:
            #         angle_to_food = math.atan2(food.y - agent.y, food.x - agent.x)
            #         sensor_angle = 2 * math.pi * i / agent.sensor_count + agent.theta
            #         angle_diff = abs((angle_to_food - sensor_angle + math.pi) % (2 * math.pi) - math.pi)
            #         if angle_diff < (math.pi / agent.sensor_count):
            #             agent.sensors[i][1] = 1 - (dist / (agent.sensor_length + 0.1))
            #             agent.sensors[i][1] =  agent.sensors[i][1] * 2 - 1
            #             break

            if self.agent.food_receptor:
                #initialize as -1
                agent.sensors[i][receptor_cnt] =  0
                for food in self.food_list:
                    dist = ((agent.x - food.x) ** 2 + (agent.y - food.y) ** 2) ** 0.5
                    #round off to 6 decimal places
                    dist = round(dist, 6)
                    if dist <= agent.sensor_length:
                        angle_to_food = math.atan2(food.y - agent.y, food.x - agent.x)
                        sensor_angle = 2 * math.pi * i / agent.sensor_count + agent.theta
                        angle_diff = abs((angle_to_food - sensor_angle + math.pi) % (2 * math.pi) - math.pi)
                        if angle_diff < (math.pi / agent.sensor_count):
                            # agent.sensors[i][receptor_cnt] = max(2* (1 - (dist / (agent.sensor_length + 0.1))) - 1 , agent.sensors[i][receptor_cnt])
                            # agent.sensors[i][receptor_cnt] =  agent.sensors[i][receptor_cnt] * 2 - 1
                            # print(f"Food detected at Sensor {i}, {dist} distance away")
                            agent.sensors[i][receptor_cnt] = max(1 - (dist / (agent.sensor_length + 0.1)), agent.sensors[i][receptor_cnt])
                            break
                receptor_cnt += 1
            
            if self.agent.nest_receptor:
                dist = ((agent.x - nest.x) ** 2 + (agent.y - nest.y) ** 2) ** 0.5
                if dist <= agent.sensor_length:
                    angle_to_nest = math.atan2(nest.y - agent.y, nest.x - agent.x)
                    sensor_angle = 2 * math.pi * i / agent.sensor_count + agent.theta
                    angle_diff = abs((angle_to_nest - sensor_angle + math.pi) % (2 * math.pi) - math.pi)
                    if angle_diff < (math.pi / agent.sensor_count):
                        agent.sensors[i][receptor_cnt] = 1 - (dist / (agent.sensor_length + 0.1))
                        #normalize to -1, 1
                        agent.sensors[i][receptor_cnt] = agent.sensors[i][receptor_cnt] * 2 - 1
                    else:
                        agent.sensors[i][receptor_cnt] = 0
                else:
                    agent.sensors[i][receptor_cnt] = 0
                receptor_cnt += 1
        
                        
        # if not self.agent.food_receptor:
        #     agent.sensors[i+1][0] = 0
        #     agent.sensors[i+1][1] = 0
           
        #     for food in self.food_list:
        #         dist = ((agent.x - food.x) ** 2 + (agent.y - food.y) ** 2) ** 0.5
        #         #round off to 6 decimal places
        #         dist = round(dist, 6)
        #         if dist <= agent.sensor_length:
        #             #find difference between agent theta and angle to food
        #             angle_to_food = math.atan2(food.y - agent.y, food.x - agent.x)
        #             angle_diff = abs((angle_to_food - agent.theta + math.pi) % (2 * math.pi) - math.pi)
        #             #pass these values to self.agnent.sensors 
        #             agent.sensors[i+1][0] = 1 - (dist / (agent.sensor_length + 0.1))
        #             #normalizing to -1, 1
        #             agent.sensors[i+1][0] = agent.sensors[i+1][0] * 2 - 1
        #             agent.sensors[i+1][1] = angle_diff / math.pi  # normalize to [0, 1]

        

                    
    def update_pheromones(self):
        for i in range(len(self.pheromones)):
            strength =   self.pheromones[i].strength
            # strength -= 1.0/self.optimalTime
            # strength *= .99
            strength *= self.decay_factor
            if strength < self.decay_limit:
                self.pheromones[i] = None
            else:
                self.pheromones[i].strength = strength
        self.pheromones[:] = [p for p in self.pheromones if p is not None]

    def draw_agent(self):
         
        self.agent.draw(self.window)

    def draw_pheromones(self):
        for pheromone in self.pheromones:
             
            pheromone.draw(self.window)

    def draw_food(self):
         
        for food in self.food_list:
            
            food.draw(self.window)

    def draw_nest(self):
        self.nest.draw(self.window)

    def draw_obstacles(self):
        for obstacle in self.obstacles:
            pygame.draw.line(self.window, obstacle.color, (obstacle.startx, obstacle.starty), (obstacle.endx, obstacle.endy), 5)


    def draw_sensors_collisions(self):
        agent = self.agent
        messages = []  # Store messages to display
        font = pygame.font.Font(None, 24)  # Create a font object
        #check agent's sensors for collisions with food, pheromones and nest and display a message in the window if there is a collision
        #food sensors idx
        food_sensor_idx = self.agent.pheromone_receptor + self.agent.food_receptor -1
        for i in range(agent.sensor_count):
            
            if agent.food_receptor:
                if agent.sensors[i][food_sensor_idx] >0 :
                    messages.append(f"Food detected at Sensor {i}, {agent.sensors[i][food_sensor_idx]} distance away")

        padding = 10
        line_height = 20
        start_y = self.window_height - (len(messages) * line_height) - padding  # Start from bottom

        for i, msg in enumerate(messages):
            text_surface = font.render(msg, True, (255, 255, 255))  # White text
            self.window.blit(text_surface, (padding, start_y + i * line_height))  # Align left with padding



    def draw(self, draw_score=True, draw_hits=False):
        self.window.fill(self.BLACK)
        self.draw_obstacles()

        self.draw_agent()
        self.draw_food()
        self.draw_pheromones()
        self.draw_nest()
        self._draw_score()
        self.draw_sensors_collisions()

        if getattr(self, 'collision_occurred', False):
            font = pygame.font.SysFont("comicsans", 60)
            text = font.render("Collision", True, self.RED)
            self.window.blit(text, (
                self.window_width // 2 - text.get_width() // 2,
                self.window_height // 2 - text.get_height() // 2
            ))
        

    

    def move_agent(self, O1, O2, O3): 
        self.collision_occurred = False

        def segment_intersection(p1, p2, p3, p4):
                """
                Returns the intersection point of two line segments p1-p2 and p3-p4,
                or None if they do not intersect inside the segments.
                """
                x1, y1 = p1
                x2, y2 = p2
                x3, y3 = p3
                x4, y4 = p4

                r = ((x2-x1), (y2-y1))
                s = ((x4-x3), (y4-y3))

                #store r cross s
                r_cross_s = r[0]*s[1] - r[1]*s[0]
                #store (q-p) cross r
                q = (x3, y3)
                p = (x1, y1)
                q_minus_p = (q[0]-p[0], q[1]-p[1])
                q_minus_p_cross_r = q_minus_p[0]*r[1] - q_minus_p[1]*r[0]
                q_minus_p_cross_s = q_minus_p[0]*s[1] - q_minus_p[1]*s[0]

                if r_cross_s == 0:
                    return None  # Parallel lines
                u = q_minus_p_cross_r / r_cross_s
                t = q_minus_p_cross_s / r_cross_s
                

                if r_cross_s != 0 and 0 <= t <= 1 and 0 <= u <= 1:
                    intersection_x = p[0] + t * r[0]
                    intersection_y = p[1] + t * r[1]
                    
                    return (intersection_x, intersection_y)
                else:
                    return None  # No intersection within the segments

                
        old_X = self.agent.x
        old_Y = self.agent.y

        # self.agent.theta = self.agent.theta + math.pi * O2
        # X = self.agent.x + self.agent.vel * O1* math.cos(self.agent.theta) 
        # Y = self.agent.y + self.agent.vel * O1* math.sin(self.agent.theta) 
        self.agent.theta = self.agent.theta + math.pi * O3
        self.agent.theta = self.agent.theta % (2 * math.pi)
        #make sure self.agent.theta is between 0 and 2pi

        if self.movement_type == "holonomic": 
        
            X = self.agent.x + self.agent.vel * O1* math.cos(self.agent.theta) + self.agent.vel * O2* math.cos(self.agent.theta + math.pi/2)
            Y = self.agent.y + self.agent.vel * O1* math.sin(self.agent.theta) + self.agent.vel * O2* math.sin(self.agent.theta + math.pi/2)
        elif self.movement_type == "tank":

            on_status  = 0 if O1 > 0 else 1
            X = self.agent.x + self.agent.vel * O1* math.cos(self.agent.theta) 
            Y = self.agent.y - self.agent.vel * O1* math.sin(self.agent.theta)
        elif self.movement_type == "1d":
            X = self.agent.x + self.agent.vel * O1* math.cos(self.agent.theta) 
            Y = self.agent.y + self.agent.vel * O1* math.sin(self.agent.theta)
        distance_to_travel = ((X - old_X) ** 2 + (Y - old_Y) ** 2) ** 0.5
        self.total_distance_travelled += distance_to_travel

        #check if agent's path crosses the coordinates of any food particles
        #if it does, move agent to that food particle directly, call _handle_collision, and then move to the final position
        #if movement is greater than 0
        eps=1e-9
        if distance_to_travel > eps:
            
            for food in self.food_list:
                cross = (food.x - old_X) * (Y - old_Y) - (food.y - old_Y) * (X - old_X)
                if abs(cross) > eps:
                    continue
                dot = (food.x - old_X) * (X - old_X) + (food.y - old_Y) * (Y - old_Y)
                if dot < -eps:
                    continue
                squared_length = (X - old_X) ** 2 + (Y - old_Y) ** 2
                if dot - squared_length > eps:
                    continue
                #move agent to food position
                self.agent.move(food.x, food.y)
                self._handle_collision()
            


        if X < 0 or X > self.window_width or Y < 0 or Y > self.window_height:
            return False
        #check if agent collides with any obstacles
        

        def ricochet_simulation(start_point, target_point, obstacles, remaining_distance, max_bounces=10):
            if max_bounces <= 0 or remaining_distance < 1e-6:
                #check if start point is inside any obstacle
                for obstacle in obstacles:
                    if segment_intersection(start_point, start_point, (obstacle.startx, obstacle.starty), (obstacle.endx, obstacle.endy)) is not None:
                        self.collision_occurred = True 
                        # print("Collision with obstacle")
                        return self.agent.x, self.agent.y
                return start_point
            for obstacle in obstacles:
                intersect = segment_intersection(
                    start_point, target_point, (obstacle.startx, obstacle.starty), (obstacle.endx, obstacle.endy))
                if intersect is not None:
                    self.collision_occurred = True 
                    # print("Collision with obstacle")
                    if not self.ricochet:
                        return start_point

                    travel_used = ((intersect[0] - start_point[0])**2 +
                                   (intersect[1] - start_point[1])**2)**0.5
                    new_remaining = remaining_distance - travel_used
                    # Calculate the normal vector of the obstacle
                    obs_vec = (obstacle.endx - obstacle.startx,
                               obstacle.endy - obstacle.starty)
                    obs_length = (obs_vec[0]**2 + obs_vec[1]**2)**0.5
                    if obs_length == 0:
                        continue  # Avoid division by zero
                    obs_unit = (obs_vec[0]/obs_length, obs_vec[1]/obs_length)
                    normal = (-obs_unit[1], obs_unit[0])  # Perpendicular vector

                    # Calculate incoming vector
                    incoming = (target_point[0] - start_point[0],
                                target_point[1] - start_point[1])
                    incoming_length = (
                        incoming[0]**2 + incoming[1]**2)**0.5
                    if incoming_length == 0:
                        continue  # Avoid division by zero
                    incoming_unit = (
                        incoming[0]/incoming_length, incoming[1]/incoming_length)

                    # Reflect the incoming vector around the normal
                    dot_product = (incoming_unit[0]*normal[0] +
                                   incoming_unit[1]*normal[1])
                    reflected = (incoming_unit[0] - 2*dot_product*normal[0],
                                 incoming_unit[1] - 2*dot_product*normal[1])

                    # Calculate new target point based on remaining distance
                    new_target = (intersect[0] + reflected[0]*new_remaining,
                                  intersect[1] + reflected[1]*new_remaining)
                    
                    # new_start = (intersect[0] + reflected[0]*1e-6,
                    #               intersect[1] + reflected[1]*1e-6)  # Small step to avoid immediate re-collision
                    #lets make new start as a point slightly before the intersection point using the incident vector
                    step_back = 1e-6
                    incident_unit = (incoming_unit[0], incoming_unit[1])
                    new_start = (intersect[0] - incident_unit[0]*step_back,
                                 intersect[1] - incident_unit[1]*step_back)
                    


                    # Recursively check for further collisions
                    return ricochet_simulation(new_start, new_target, obstacles, new_remaining, max_bounces - 1)
            return target_point  # No collision, return original target

           

        
        
        new_position = ricochet_simulation(
            (old_X, old_Y), (X, Y), self.obstacles, distance_to_travel)
        if new_position is None:
            return False  # Collision with obstacle, movement not valid

        self.agent.move(new_position[0], new_position[1])

        # target = self._current_path_target()
        # if target is not None:
        #     tx, ty = target
        #     dx = self.agent.x - tx
        #     dy = self.agent.y - ty
        #     dist = (dx*dx + dy*dy) ** 0.5

        #     # Reward when agent comes close enough to the current milestone, in order
        #     if dist <= self.path_radius:
        #         self.score += self.path_reward
        #         self.current_milestone_idx += 1


        if self.simple_guidance:
            #give a small score if agent is moving towards food when self.carrying_food is False
            if self.movement_type != "1d":
                if not self.agent.carrying_food:


                    if len(self.food_list) >0:

                        if self.searching_time%10 ==0:
                            dist1 = min([((self.agent.x - food.x) ** 2 + (self.agent.y - food.y) ** 2) ** 0.5 for food in self.food_list])
                            dist2 = min([((self.prev_pos[0] - food.x) ** 2 + (self.prev_pos[1] - food.y) ** 2) ** 0.5 for food in self.food_list])
                            if dist1 < dist2:
                                self.score += self.food_seek_reward * (abs(dist2 - dist1))
                                self.current_direction = "Towards Food"
                            # #punish if it instead moves away from food
                            # elif dist1 > dist2:
                            #     self.score -= self.food_seek_reward * (abs(dist2 - dist1))/2

                            # else:
                            #     #punish if it stays in the same place
                            #     self.score -= self.food_seek_reward * 5
                            #update previous position if current position was closer to food
                                self.prev_pos = (self.agent.x, self.agent.y)
                            

                else:
                    if self.carry_time %10 ==0:
                        dist1 = ((self.agent.x - self.nest.x) ** 2 + (self.agent.y - self.nest.y) ** 2) ** 0.5
                        dist2 = ((self.prev_pos[0] - self.nest.x) ** 2 + (self.prev_pos[1] - self.nest.y) ** 2) ** 0.5
                        if dist1 < dist2:
                            self.score += self.nest_seek_reward * (abs(dist2 - dist1))
                            self.current_direction = "Towards Nest"
                        #punish if it instead moves away from nest
                        # elif dist1 > dist2:
                        #     self.score -= self.nest_seek_reward * (abs(dist2 - dist1))/2
                        # else:
                        #     #punish if it stays in the same place
                        #     self.score -= self.nest_seek_reward * 5

                    # if dist1!= dist2:
                    #     self.score += self.general_movement_reward_after_food_collection
                        #update previous position
                            self.prev_pos = (self.agent.x, self.agent.y)
            else:
                non_zero_flag = False
                #go through agent's input sensors for any non zero value
                for i in range(self.agent.sensor_count):
                    for j in range(len(self.agent.sensors[i])):
                        if self.agent.sensors[i][j] != 0:
                            non_zero_flag = True
                            break
                    if non_zero_flag:
                        break
                #if all values are zero and agent is still moving, reward it with persistent movement reward
                if not non_zero_flag and O1 != 0:
                    self.score += self.persistent_movement_reward
                    self.current_direction = "Exploring"

        #check if the current path the agent takes intersects with any food particles

        

        


        return True


    def place_pheromone(self):
        self.pheromones.append(Pheromone(self.agent.x, self.agent.y, .01))



    def loop(self):
        """
        Executes a single game loop.

        :returns: GameInformation instance stating score 
                  and hits of each paddle.
        """
         
        self._handle_collision()
        self.update_sensor_data()
        self.update_pheromones()
        
        game_info = GameInformation(
            self.score, self.food_collected, self.total_food, self.food_list)
        
         
        return game_info


