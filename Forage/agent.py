import pygame
import math
BLUE = (0, 0, 255)
PINK = (255, 192, 203)
YELLOW = (255, 255, 0)

class Agent:
      
    def __init__(self, x, y):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.sensor_count = 8
        self.sensor_segments = 8
        self.carrying_food = False
        self.radius = 10
        self.sensor_length = 100
        self.color = BLUE
        self.sensor_count = 8
        self.vel = 4
        self.sensors = [[[0, 0, 0] for _ in range(self.sensor_segments)] for _ in range(self.sensor_count)]
        self.sensor_color = YELLOW


    def draw(self, win):
        pygame.draw.circle(
                win, self.color, (self.x, self.y), self.radius)
        # Draw sensors
        for i in range(self.sensor_count):
            angle = 2 * math.pi * i / self.sensor_count
            end_x = self.x + self.sensor_length * math.cos(angle)
            end_y = self.y + self.sensor_length * math.sin(angle)
            pygame.draw.line(win, YELLOW, (self.x, self.y), (end_x, end_y), 1)
         

    def move(self, move_direction):
        if move_direction == "N":
            self.y -= self.vel
        elif move_direction == "NE":
            self.x += self.vel*math.cos(math.pi/4)
            self.y -= self.vel*math.sin(math.pi/4)
        elif move_direction == "E":
            self.x += self.vel
        elif move_direction == "SE":
            self.x += self.vel*math.cos(math.pi/4)
            self.y += self.vel*math.sin(math.pi/4)
        elif move_direction == "S":
            self.y += self.vel
        elif move_direction == "SW":
            self.x -= self.vel*math.cos(math.pi/4)
            self.y += self.vel*math.sin(math.pi/4)
        elif move_direction == "W":
            self.x -= self.vel
        elif move_direction == "NW":
            self.x -= self.vel*math.cos(math.pi/4)
            self.y -= self.vel*math.sin(math.pi/4)
        else:
            raise ValueError("Invalid move direction")
        



    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
