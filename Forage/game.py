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

    def __init__(self, window, window_width, window_height, arrangement_idx=0, obstacles = False, particles = 5, ricochet = True, obstacle_type="line", seeded=False, o_switch=False):
        
        
      
        self.window_width = window_width
        self.window_height = window_height

        self.agent = Agent(
            self.window_height // 2 , self.window_width // 2)
        self.total_food = particles
        food_pos = []
        buffer_distance = self.agent.sensor_length  # minimum Euclidean distance from agent (prevents spawn overlap)
        
        if seeded:
            random.seed(arrangement_idx)
            numpy.random.seed(arrangement_idx)

        while len(food_pos) < self.total_food:
            x = random.randint(400, 600)
            y = random.randint(100, 250)
            # Prevent food from spawning on top of the agent
            dist = ((self.agent.x - x)**2 + (self.agent.y - y)**2)**0.5
            if dist > buffer_distance:
                food_pos.append((x, y))

        # self.food_list = [Food(x, y) for x, y in food_pos]
        # food_pos = arrangements[arrangement_idx]
        self.food_list = [Food(x, y) for x, y in food_pos]
        

        # self.optimalTime = 250
        # self.TIME_BONUS =  2*self.agent.sensor_length/self.agent.vel + 2 + 20
        self.optimalTime = (400/self.agent.vel)*4
        
        self.score = 0
        self.food_collected = 0
        self.pheromones = []
        self.nest = Nest(self.agent.x, self.agent.y)
        self.window = window
        self.current_direction = ""

        self.discount_factor = 0.999
        self.carry_time = 0
        
        self.obstacles = [
            # Obstacle(300, 450, 300, 300)
            # Obstacle(200, 300, 700, 300)
            # Obstacle(600, 300, 600, 450)
            
        ]

        #offset angle randomly based on arrangement idx
        if o_switch:
            self.agent.theta = (arrangement_idx % 8) * (math.pi/4)
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

    def _draw_score(self):
        score_text = self.SCORE_FONT.render(
            f"{self.score}", 1, self.WHITE)
        
        direction_text = self.SCORE_FONT.render(
            f"{self.current_direction}", 1, self.WHITE)
         
        self.window.blit(score_text, (self.window_width //
                                           4 - score_text.get_width()//2, 20))
        
        self.window.blit(direction_text, (self.window_width //
                                             4 - direction_text.get_width()//2, 60))
         
    



    def _handle_collision(self):
       
        agent = self.agent
        nest = self.nest
        collision_threshold = self.agent.radius + self.food_list[0].radius if self.food_list else 0
        if True:
            for food in self.food_list:
                dist = ((agent.x - food.x) ** 2 + (agent.y - food.y) ** 2) ** 0.5
                # if dist < agent.radius + food.radius:
                if dist < collision_threshold:
                    self.score += 1
                    self.agent.x = food.x
                    self.agent.y = food.y
                    # agent.carrying_food = True
                    self.carry_time = 0
                    self.food_list.remove(food)
                    
                    break       
   
    def update_sensor_data(self):

        agent = self.agent
        nest = self.nest
        for i in range(agent.sensor_count):
            mini = 1000
             
            for food in self.food_list:
                dist = ((agent.x - food.x) ** 2 + (agent.y - food.y) ** 2) ** 0.5
                if mini <= dist:
                    continue
                #insert dist into sens
                mini = dist
                agent.sensors[i][0] = mini #if dist < agent.sensor_length else 0 (should be handled automaticaly I think)
                #calculate angle difference between food and agent.theta
                angle_to_food = math.atan2(food.y - agent.y, food.x - agent.x)
                relative_angle = (angle_to_food - agent.theta) % (2 * math.pi)
                agent.sensors[i][1] = relative_angle
    def update_pheromones(self):
        for i in range(len(self.pheromones)):
            strength =   self.pheromones[i].strength
            # strength -= 1.0/self.optimalTime
            strength *= .99
            if strength < 0.1:
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
        for i in range(agent.sensor_count):
            
            if agent.food_receptor:
                if agent.sensors[i][agent.sensor_segments] >0 :
                    messages.append(f"Food detected at Sensor {i}, {agent.sensors[i][agent.sensor_segments]} distance away")

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
        X = self.agent.x + self.agent.vel * O1* math.cos(self.agent.theta) + self.agent.vel * O2* math.cos(self.agent.theta + math.pi/2)
        Y = self.agent.y + self.agent.vel * O1* math.sin(self.agent.theta) + self.agent.vel * O2* math.sin(self.agent.theta + math.pi/2)

        distance_to_travel = ((X - old_X) ** 2 + (Y - old_Y) ** 2) ** 0.5

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
        return True


    def place_pheromone(self):
        self.pheromones.append(Pheromone(self.agent.x, self.agent.y, 1.0))



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


