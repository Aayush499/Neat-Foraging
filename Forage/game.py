from .agent import Agent
from .food import Food
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

    def __init__(self, window, window_width, window_height, arrangement_idx=0):
        self.window_width = window_width
        self.window_height = window_height

        self.agent = Agent(
            self.window_height // 2 , self.window_width // 2)
        self.total_food = 2
        y_offset_positive= math.sqrt(self.agent.sensor_length**2 - 90**2 )
        y_offset_negative= -math.sqrt(self.agent.sensor_length**2 - 90**2 )
        #create 4 possible food paths, one to the north,to the east, south and west
        food_pos_N = [(self.agent.x, self.agent.y+ (self.agent.sensor_length)*i) for i in range(1, self.total_food+1)]
        food_pos_E = [(self.agent.x+ (self.agent.sensor_length)*i, self.agent.y) for i in range(1, self.total_food+1)]
        food_pos_S = [(self.agent.x, self.agent.y- (self.agent.sensor_length)*i) for i in range(1, self.total_food+1)]
        food_pos_W = [(self.agent.x- (self.agent.sensor_length)*i, self.agent.y) for i in range(1, self.total_food+1)]
        food_pos_SW = [(self.agent.x, self.agent.y- (self.agent.sensor_length)), (self.agent.x-(self.agent.sensor_length), self.agent.y- (self.agent.sensor_length))]
        food_pos_SE = [(self.agent.x, self.agent.y- (self.agent.sensor_length)), (self.agent.x+(self.agent.sensor_length), self.agent.y- (self.agent.sensor_length))]
        food_pos_NW = [(self.agent.x, self.agent.y+ (self.agent.sensor_length)), (self.agent.x-(self.agent.sensor_length), self.agent.y+ (self.agent.sensor_length))]
        food_pos_NE = [(self.agent.x, self.agent.y+ (self.agent.sensor_length)), (self.agent.x+(self.agent.sensor_length), self.agent.y+ (self.agent.sensor_length))]
        food_pos_EN = [(self.agent.x+(self.agent.sensor_length), self.agent.y), (self.agent.x+(self.agent.sensor_length), self.agent.y+(self.agent.sensor_length))]
        food_pos_ES = [(self.agent.x+(self.agent.sensor_length), self.agent.y), (self.agent.x+(self.agent.sensor_length), self.agent.y-(self.agent.sensor_length))]
        food_pos_WN = [(self.agent.x-(self.agent.sensor_length), self.agent.y), (self.agent.x-(self.agent.sensor_length), self.agent.y+(self.agent.sensor_length))]
        food_pos_WS = [(self.agent.x-(self.agent.sensor_length), self.agent.y), (self.agent.x-(self.agent.sensor_length), self.agent.y-(self.agent.sensor_length))]
        food_pos_N_offset = [(self.agent.x, self.agent.y+ self.agent.sensor_length), (self.agent.x+90, self.agent.y+ self.agent.sensor_length + y_offset_positive)]
        food_pos_S_offset = [(self.agent.x, self.agent.y- self.agent.sensor_length), (self.agent.x-90, self.agent.y- self.agent.sensor_length + y_offset_negative)]
         #choose one of the 4 arrangements based on the arrangement_idx parameter
        arrangements = [food_pos_N, food_pos_E, food_pos_S, food_pos_W, food_pos_SW, food_pos_SE, food_pos_NW, food_pos_NE, food_pos_EN, food_pos_ES, food_pos_WN, food_pos_WS, food_pos_N_offset, food_pos_S_offset]
        # food_pos = random.choice(arrangements)
        food_pos = arrangements[arrangement_idx]
        self.food_list = [Food(x, y) for x, y in food_pos]


        self.optimalTime = 2*self.agent.sensor_length/self.agent.vel + 2 +20
        self.TIME_BONUS =  2*self.agent.sensor_length/self.agent.vel + 2 + 20
        
        self.score = 0
        self.food_collected = 0
        self.pheromones = []
        self.nest = Nest(self.agent.x, self.agent.y)
        self.window = window
        self.current_direction = ""

        self.discount_factor = 0.999
        self.carry_time = 0

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
        collision_threshold = 0.1
        if True:
            for food in self.food_list:
                dist = ((agent.x - food.x) ** 2 + (agent.y - food.y) ** 2) ** 0.5
                # if dist < agent.radius + food.radius:
                if dist < collision_threshold:
                    self.score += 1
                    
                    # agent.carrying_food = True
                    self.carry_time = 0
                    self.food_list.remove(food)
                    break
        # else:
        #     dist = ((agent.x - nest.x) ** 2 + (agent.y - nest.y) ** 2) ** 0.5
        #     self.carry_time += 1
        #     # if dist < agent.radius + nest.radius:
        #     if dist < collision_threshold:
               
        #         # self.score += 2 + math.pow(self.discount_factor , self.carry_time)

        #         self.food_collected += 1
        #         # agent.carrying_food = False
        #         self.TIME_BONUS += 2*self.agent.sensor_length/self.agent.vel  
        #         self.optimalTime += self.TIME_BONUS
            
        


                
            
   
    def update_sensor_data(self):

        agent = self.agent
        nest = self.nest
        for i in range(agent.sensor_count):
        #     angle = 2 * math.pi * i / agent.sensor_count
        #     def check_quadrant(agent, object):

        #         #calculate distance between agent and object

        #         dist = ((agent.x - object.x) ** 2 + (agent.y - object.y) ** 2) ** 0.5
        #         dx = object.x - agent.x
        #         dy = object.y - agent.y
        #         angle_to_food = math.atan2(dy, dx)            # Returns angle in radians

        #         relative_angle = (angle_to_food - agent.theta) % (2 * math.pi)   # Ensures value between 0 and 2π

        #         bin_width = (2 * math.pi) / agent.sensor_count
        #         sensor_index = int(relative_angle // bin_width)    # Integer division places angle into a sensor’s bin
        #         return sensor_index

        #     for j in range(agent.sensor_segments):

        #         # Reset sensor data
        #         agent.sensors[i][j] = 0
                 

        #         #check pheromones
        #         max_strength = 0
        #         for pheromone in self.pheromones:
        #             dist = ((agent.x - pheromone.x) ** 2 + (agent.y - pheromone.y) ** 2) ** 0.5
        #             if check_quadrant(agent, pheromone) == i and dist <= agent.sensor_length:
        #                 agent.sensors[i][j] = max(agent.sensors[i][j], pheromone.strength*(1 - dist/(agent.sensor_length+.1)))
        #                 # if pheromone.strength > max_strength:
        #                 #     max_strength = pheromone.strength
        #                 #     agent.sensors[i][j] = max_strength*(1 - dist/agent.sensor_length)  
                            
            
        #     agent.sensors[i][j+1] = 0  # Default value for no food detected
        #     # Check for food
        #     for food in self.food_list:
        #         dist = ((agent.x - food.x) ** 2 + (agent.y - food.y) ** 2) ** 0.5
        #         if check_quadrant(agent, food) == i and dist <= agent.sensor_length:
                    
        #             agent.sensors[i][j+1] = max(1 - (dist / (agent.sensor_length+.1)), agent.sensors[i][j+1]) #if dist < agent.sensor_length else 0 (should be handled automaticaly I think)

            
        #     #check nest
        #     if agent.nest_receptor:
        #         agent.sensors[i][j+2] = 0  # Default value for no nest detected
        #         dist = ((agent.x - nest.x) ** 2 + (agent.y - nest.y) ** 2) ** 0.5
        #         if check_quadrant(agent, nest) == i and dist <= agent.sensor_length:

        #             agent.sensors[i][j+2] = 1 - (dist / agent.sensor_length + .1)

        # #check if all sensors are zero, if so , turn discount factor to .1
        # all_zero = all(all(segment == 0 for segment in sensor) for sensor in agent.sensors)
        # if all_zero:
        #     self.optimalTime = 0
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


    def draw_sensors_collisions(self):
        agent = self.agent
        messages = []  # Store messages to display
        font = pygame.font.Font(None, 24)  # Create a font object
        #check agent's sensors for collisions with food, pheromones and nest and display a message in the window if there is a collision
        for i in range(agent.sensor_count):
            
            if agent.food_receptor:
                if agent.sensors[i][agent.sensor_segments] >0 :
                    messages.append(f"Food detected at Sensor {i}, {agent.sensors[i][agent.sensor_segments]} distance away")
                    # if agent.sensors[i][j][1] > 0:
                    #     messages.append(f"Pheromone detected at Sensor {i}, Segment {j}")
                    # if agent.sensors[i][j][2] == 1:
                    #     messages.append(f"Nest detected at Sensor {i}, Segment {j}")

        padding = 10
        line_height = 20
        start_y = self.window_height - (len(messages) * line_height) - padding  # Start from bottom

        for i, msg in enumerate(messages):
            text_surface = font.render(msg, True, (255, 255, 255))  # White text
            self.window.blit(text_surface, (padding, start_y + i * line_height))  # Align left with padding



    def draw(self, draw_score=True, draw_hits=False):
        self.window.fill(self.BLACK)

        self.draw_agent()
        self.draw_food()
        self.draw_pheromones()
        self.draw_nest()
        self._draw_score()
        self.draw_sensors_collisions()



    def move_agent(self, O1, O2): 

        self.agent.theta = self.agent.theta + math.pi * O2
        # X = self.agent.x + self.agent.vel * O1* math.cos(self.agent.theta) + self.agent.vel * O2* math.cos(self.agent.theta + math.pi/2)
        # Y = self.agent.y + self.agent.vel * O1* math.sin(self.agent.theta) + self.agent.vel * O2* math.sin(self.agent.theta + math.pi/2)
        #check whether the agent hits the borders of the screen is move is executed
        #movement directions are N, NE, E, SE, S, SW, W, NW
        #if it hits a border, return False else move the agent and return True
        #find new x and y component after moving in new theta direction
        X = self.agent.x + O1 * self.agent.vel * math.cos(self.agent.theta)
        Y = self.agent.y + O1 * self.agent.vel * math.sin(self.agent.theta)

        if X < 0 or X > self.window_width or Y < 0 or Y > self.window_height:
            return False
        self.agent.move(X, Y)

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


