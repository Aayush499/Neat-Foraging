from .agent import Agent
from .food import Food
from .pheromone import Pheromone
from .nest import Nest
from .foodTree import foodGenerator
from .optimalPath import bestPath
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

    def __init__(self, window, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

        self.agent = Agent(
            self.window_height // 2 , self.window_width // 2)
        self.total_food = 1
        food_pos, edges = foodGenerator((self.agent.x, self.agent.y), self.total_food, self.agent.sensor_length, self.agent.sensor_count)
        self.optimalTime = (((bestPath(edges, (self.agent.x, self.agent.y)))*self.agent.sensor_length)/self.agent.vel)*2
        # self.optimalTime  = 2*self.agent.sensor_length/self.agent.vel
        self.food_list = [Food(x, y) for x, y in food_pos]
        #pick a random sensor to place the first food
        # first_food = random.randint(0,self.agent.sensor_count-1)
        #BIASING THE FOOD TO THE EAST AND WEST
        # if random.random() < 0.8:
        #     first_food = random.choice([0, 2])
        # else:
        #     first_food = random.choice([1, 3])

            
        # angle = 2 * math.pi * first_food / self.agent.sensor_count
        # x1 = self.agent.x + self.agent.sensor_length * math.cos(angle)
        # y1 = self.agent.y - self.agent.sensor_length * math.sin(angle)
        # self.food_list = [Food(x1, y1)]


        self.score = 0
        self.food_collected = 0
        self.pheromones = []
        self.nest = Nest(self.agent.x, self.agent.y)
        self.window = window
        self.current_direction = ""

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
        if not agent.carrying_food:
            for food in self.food_list:
                dist = ((agent.x - food.x) ** 2 + (agent.y - food.y) ** 2) ** 0.5
                if dist < agent.radius + food.radius:
                    self.score += 1
                    
                    agent.carrying_food = True
                    self.food_list.remove(food)
                    break
        else:
            dist = ((agent.x - nest.x) ** 2 + (agent.y - nest.y) ** 2) ** 0.5
            if dist < agent.radius + nest.radius:
                self.score += 2
                self.food_collected += 1
                agent.carrying_food = False
            
   
    def update_sensor_data(self):

        agent = self.agent
        nest = self.nest
        for i in range(agent.sensor_count):
            angle = 2 * math.pi * i / agent.sensor_count
            def check_circle_intersection(d, centre, radius):
                    """Check for intersection of the line segment with a circle."""
                    f = numpy.array([x1 - centre[0], y1 - centre[1]])
                    a = d.dot(d)
                    b = 2 * f.dot(d)
                    c = f.dot(f) - radius**2
                    discriminant = b**2 - 4 * a * c

                    if(discriminant < 0):
                        return False
                    else:
                        discriminant = math.sqrt(discriminant)
                        t1 = (-b - discriminant) / (2 * a)
                        t2 = (-b + discriminant) / (2 * a)


                        if t1>=0 and t1<=1:
                            return True
                        elif t2>=0 and t2<=1:
                            return True
                        elif t1<0 and t2>1:
                            return True
                        else:
                            return False
            for j in range(agent.sensor_segments):
                segment_length_end = (j + 1) * ( agent.sensor_length / agent.sensor_segments)
                segment_length_start = j * (agent.sensor_length / agent.sensor_segments)
                x2 =  agent.x + segment_length_end * math.cos(angle)
                y2 = agent.y - segment_length_end * math.sin(angle)
                x1 = agent.x + segment_length_start * math.cos(angle)
                y1 = agent.y - segment_length_start * math.sin(angle)

                # Reset sensor data
                agent.sensors[i][j] = 0
                
                # Sensor line coefficients
                d = numpy.array([x2 - x1, y2 - y1])

                #check pheromones
                for pheromone in self.pheromones:
                    if check_circle_intersection(d, (pheromone.x,pheromone.y), pheromone.radius):
                        agent.sensors[i][j] = max(agent.sensors[i][j], pheromone.strength)
                
            x2 = agent.x + agent.sensor_length * math.cos(angle)
            y2 = agent.y - agent.sensor_length * math.sin(angle)
            x1 = agent.x
            y1 = agent.y
            d = numpy.array([x2 - x1, y2 - y1])   
            
            agent.sensors[i][j+1] = agent.sensor_length+1  # Default value for no food detected
            # Check for food
            for food in self.food_list:
                if check_circle_intersection(d, (food.x,food.y), food.radius):
                    dist = ((agent.x - food.x) ** 2 + (agent.y - food.y) ** 2) ** 0.5
                    agent.sensors[i][j+1] = dist

            
            #check nest
            agent.sensors[i][j+2] = agent.sensor_length+1  # Default value for no nest detected
            if check_circle_intersection(d, (nest.x,nest.y), nest.radius):
                dist = ((agent.x - nest.x) ** 2 + (agent.y - nest.y) ** 2) ** 0.5
                agent.sensors[i][j+2] = dist

    def update_pheromones(self):
        for i in range(len(self.pheromones)):
            x, y, strength = self.pheromones[i].x, self.pheromones[i].y, self.pheromones[i].strength
            strength -= 1.0/self.optimalTime
            if strength <= 0:
                self.pheromones[i] = None
            else:
                self.pheromones[i].strength = strength
        self.pheromones[:] = [p for p in self.pheromones if p is not None]

    def draw_agent(self):
        # pygame.draw.circle(self.window, self.agent.color, (self.agent.x,self.agent.y), self.agent.radius)

        # # Draw sensors
        # for i in range(self.agent.sensor_count):
        #     angle = 2 * math.pi * i / self.agent.sensor_count
        #     end_x = self.agent.x + self.agent.sensor_length * math.cos(angle)
        #     end_y = self.agent.y - self.agent.sensor_length * math.sin(angle)
        #     pygame.draw.line(self.window, self.agent.sensor_color, (self.agent.x,self.agent.y), (end_x, end_y), 1)

        self.agent.draw(self.window)

    def draw_pheromones(self):
        for pheromone in self.pheromones:
            # x, y, strength = pheromone.x, pheromone.y, pheromone.strength
            # if strength > 0:
            #     alpha = int(255 * strength)
            #     color = (255, 255, 0, alpha)
            #     pygame.draw.circle(self.window, pheromone.color, (x, y), pheromone.radius)
            pheromone.draw(self.window)

    def draw_food(self):
        # pygame.draw.circle(screen, GREEN, food_pos, food_radius)
        for food in self.food_list:
            # pygame.draw.circle(self.window, food.color, (food.x, food.y), food.radius)
            food.draw(self.window)

    def draw_nest(self):
        self.nest.draw(self.window)


    def draw_sensors_collisions(self):
        agent = self.agent
        messages = []  # Store messages to display
        font = pygame.font.Font(None, 24)  # Create a font object
        #check agent's sensors for collisions with food, pheromones and nest and display a message in the window if there is a collision
        for i in range(agent.sensor_count):
            
            if agent.sensors[i][-2] >0 and agent.sensors[i][-2] <= agent.sensor_length:
                messages.append(f"Food detected at Sensor {i}, {agent.sensors[i][-2]} distance away")
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



    def move_agent(self, move_direction):
        """
        Move the left or right paddle.

        :returns: boolean indicating if paddle movement is valid. 
                  Movement is invalid if it causes paddle to go 
                  off the screen
        """
       
        self.current_direction = move_direction
        #check whether the agent hits the borders of the screen is move is executed
        #movement directions are N, NE, E, SE, S, SW, W, NW
        #if it hits a border, return False else move the agent and return True

        if move_direction == "N":
            if self.agent.y - self.agent.vel < 0:
                return False
        elif move_direction == "NE":
            if self.agent.x + self.agent.vel*math.cos(math.pi/4) > self.window_width or self.agent.y - self.agent.vel*math.sin(math.pi/4) < 0:
                return False
        elif move_direction == "E":
            if self.agent.x + self.agent.vel > self.window_width:
                return False
        elif move_direction == "SE":
            if self.agent.x + self.agent.vel*math.cos(math.pi/4) > self.window_width or self.agent.y + self.agent.vel*math.sin(math.pi/4) > self.window_height:
                return False
        elif move_direction == "S":
            if self.agent.y + self.agent.vel > self.window_height:
                return False
        elif move_direction == "SW":
            if self.agent.x - self.agent.vel*math.cos(math.pi/4) < 0 or self.agent.y + self.agent.vel*math.sin(math.pi/4) > self.window_height:
                return False
        elif move_direction == "W":
            if self.agent.x - self.agent.vel < 0:
                return False
        elif move_direction == "NW":
            if self.agent.x - self.agent.vel*math.cos(math.pi/4) < 0 or self.agent.y - self.agent.vel*math.sin(math.pi/4) < 0:
                return False
            
        self.agent.move(move_direction)

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


