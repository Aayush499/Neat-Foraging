import pygame
import math
import sys
import numpy
from foodTree import foodGenerator

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Foraging Task")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PINK = (255, 192, 203)
# Clock for controlling the frame rate
clock = pygame.time.Clock()
FPS = 60

# Agent properties
agent_pos = [WIDTH // 2, HEIGHT // 2]
agent_radius = 10
agent_speed = 4
agent_color = BLUE
sensor_count = 8
sensor_length = 100
sensor_segments = 8
carrying_food = False
score = 0

# Pheromone properties
pheromones = []  # List of tuples: (x, y, strength)
pheromone_decay_rate = 0.01

# Food properties
food_pos, edges = foodGenerator(agent_pos, 9, sensor_length)  # Initial food position
food_radius = 15

# Nest properties
nest_pos = [WIDTH // 2, HEIGHT // 2]  # Initial nest position
nest_radius = 15

# Visibility and sensor arrays
sensor_data = [[[0, 0, 0] for _ in range(sensor_segments)] for _ in range(sensor_count)]

def draw_agent():
    pygame.draw.circle(screen, agent_color, agent_pos, agent_radius)

    # Draw sensors
    for i in range(sensor_count):
        angle = 2 * math.pi * i / sensor_count
        end_x = agent_pos[0] + sensor_length * math.cos(angle)
        end_y = agent_pos[1] + sensor_length * math.sin(angle)
        pygame.draw.line(screen, YELLOW, agent_pos, (end_x, end_y), 1)

def draw_pheromones():
    for pheromone in pheromones:
        x, y, strength = pheromone
        if strength > 0:
            alpha = int(255 * strength)
            color = (255, 255, 0, alpha)
            pygame.draw.circle(screen, YELLOW, (int(x), int(y)), 3)

def draw_food():
    # pygame.draw.circle(screen, GREEN, food_pos, food_radius)
    for food in food_pos:
        pygame.draw.circle(screen, GREEN, food, food_radius)

def draw_nest():
    pygame.draw.circle(screen, RED, nest_pos, nest_radius)

def update_pheromones():
    for i in range(len(pheromones)):
        x, y, strength = pheromones[i]
        strength -= pheromone_decay_rate
        if strength <= 0:
            pheromones[i] = None
        else:
            pheromones[i] = (x, y, strength)
    pheromones[:] = [p for p in pheromones if p is not None]

def update_sensor_data():
    for i in range(sensor_count):
        angle = 2 * math.pi * i / sensor_count
        for j in range(sensor_segments):
            segment_length_end = (j + 1) * (sensor_length / sensor_segments)
            segment_length_start = j * (sensor_length / sensor_segments)
            x2 = agent_pos[0] + segment_length_end * math.cos(angle)
            y2 = agent_pos[1] + segment_length_end * math.sin(angle)
            x1 = agent_pos[0] + segment_length_start * math.cos(angle)
            y1 = agent_pos[1] + segment_length_start * math.sin(angle)
            
            # Reset sensor data
            sensor_data[i][j] = [0, 0, 0]
            
            # Sensor line coefficients
            d = numpy.array([x2 - x1, y2 - y1])
            

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

                

            # Check for food
            for food in food_pos:
                if check_circle_intersection(d, food, food_radius):
                    sensor_data[i][j][0] = 1
                    print("Food detected by sensor", i, "at segment", j)

            # Check for pheromones
            for pheromone in pheromones:
                px, py, strength = pheromone
                if check_circle_intersection(d, (px, py), 3):
                    sensor_data[i][j][1] = max(sensor_data[i][j][1], strength)
                    # print("Pheromone detected by sensor", i, "at segment", j)

            # Check for nest
            if check_circle_intersection(d, nest_pos, nest_radius):
                sensor_data[i][j][2] = 1
                # print("Nest detected by sensor", i, "at segment", j)

def check_food_collisions():
    global carrying_food, agent_color, score
    
    # Check food collision
    if not carrying_food:
        for food in food_pos:
            if math.dist(agent_pos, food) < agent_radius + food_radius:
                food_pos.remove(food)
                carrying_food = True
                print("Food collected!")
                agent_color = PINK
                score += 10
                break
    
    # Check nest collision
    if carrying_food and math.dist(agent_pos, nest_pos) < agent_radius + nest_radius:
        carrying_food = False
        print("Food deposited!")
        agent_color = BLUE
        score += 20

def handle_movement():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        agent_pos[1] -= agent_speed
    if keys[pygame.K_s]:
        agent_pos[1] += agent_speed
    if keys[pygame.K_a]:
        agent_pos[0] -= agent_speed
    if keys[pygame.K_d]:
        agent_pos[0] += agent_speed

def place_pheromone():
    x, y = agent_pos
    pheromones.append((x, y, 1.0))  # Full strength pheromone

def main():
    running = True
    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                place_pheromone()

        handle_movement()
        update_pheromones()
        update_sensor_data()
        check_food_collisions()
        

        draw_nest()
        draw_food()
        draw_pheromones()
        draw_agent()
        # print(agent_pos[0], agent_pos[1])
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
