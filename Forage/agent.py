import pygame
import math
BLUE = (0, 0, 255)
PINK = (255, 192, 203)
YELLOW = (255, 255, 0)

class Agent:
      
    def __init__(self, x, y, pheromone_receptor=True, num_sensors=8):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.sensor_count = num_sensors
        self.theta = math.pi/2 #orientation of agent
        # self.sensor_count = 4
        
        # self.sensor_segments = 8
        self.sensor_segments = 1
        self.carrying_food = False
        self.radius = 1
        # self.sensor_length = 99
        self.sensor_length = 50
        self.color = BLUE
       
        self.vel = 3
        #keep a variable that tells us what types of sensors we have 
        self.nest_receptor = False
        self.food_receptor = True
        self.pheromone_receptor = pheromone_receptor
        self.carrying_food_receptor = False
        self.angle_receptor = False
        self.sensors = [[0 for _ in range(self.sensor_segments*self.pheromone_receptor + self.food_receptor + self.nest_receptor + self.angle_receptor) ] for _ in range(self.sensor_count)]
        #if there is no food receptor, then add two more slots for direct food distance and direct angle difference
        if not self.food_receptor:
            self.sensors.append([0,0])
        self.sensor_color = YELLOW



    def draw(self, win):
        pygame.draw.circle(
                win, self.color, (self.x, self.y), self.radius)
        # Draw sensors
        for i in range(self.sensor_count):
            angle = 2 * math.pi * i / self.sensor_count + self.theta
            end_x = self.x + self.sensor_length * math.cos(angle)
            end_y = self.y - self.sensor_length * math.sin(angle)
            pygame.draw.line(win, self.sensor_color, (self.x, self.y), (end_x, end_y), 1)
         

    def move(self, X, Y):
        self.x = X
        self.y = Y 
        



    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
