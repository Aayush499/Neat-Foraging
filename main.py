# https://neat-python.readthedocs.io/en/latest/xor_example.html
from Forage import Game
import pygame
import neat
import os
import time
import pickle
import matplotlib.pyplot as plt
from Forage.food import Food
import numpy as np
NUM_RUNS = 14
MAX_PLATEAU = 20  # Generations to wait before reset; adjust as needed

class ForageTask:
    def __init__(self, window, width, height, arrangement_idx = 0):
        if window is None:
            os.environ["SDL_VIDEODRIVER"] = "dummy"  # Use a dummy display to prevent a window from opening
            pygame.display.init()  # Initialize Pygame without a window
            window = pygame.Surface((width, height))  # Create an off-screen surface

        self.game = Game(window, width, height, arrangement_idx)
        self.foods = self.game.food_list
        self.agent = self.game.agent
        self.pheromone = self.game.pheromones
        self.nest = self.game.nest
        self.sparse = False  # Set to True for sparse rewards, False for dense rewards
        
    def manual_test(self, net, auto=False):
        """
        Manually control the agent using WASD keys and observe the network's outputs.
        """
        clock = pygame.time.Clock()
        run = True
        chk = False
        log = 0
        while run:
            clock.tick(60)
            self.game.loop()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                    x, y = pygame.mouse.get_pos()
                    newFood = Food(x, y)
                    self.game.food_list.append(newFood)
                    chk = True

            theta = self.agent.theta
            o1 = 0
            o2 = 0
            o3 = 0
            o4 = 0
            # Capture key presses
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_w]:
                o1 = 1
          
            elif keys[pygame.K_a]:
                o2 = -.25
            elif keys[pygame.K_d]:
                o3 = .25
             

            # Move agent manually
            self.game.move_agent(o1, o2, o3)

            # Gather sensor inputs for network evaluation
            self.game.update_sensor_data
            # Get the network's response
            sensor_inputs = []
            for sensor in self.agent.sensors:
                for cell in sensor:
                    sensor_inputs.append(cell)  # Append (food, pheromone, nest) values
            if self.agent.carrying_food_receptor:
                    sensor_inputs.append(self.agent.carrying_food) # Append carrying food value

            output = net.activate(sensor_inputs)
             
            O1 = output[0]
            O2 = output[1]
            O3 = output[2]
            O4 = output[3]    
            place_pheromone = O4 > 0.5  # Whether it wants to place pheromones
            
            # Display network's decision
            # print(f"Predicted Move: {predicted_move}, Pheromone: {place_pheromone}")

            if place_pheromone:
                self.game.place_pheromone()
            # if auto:
            #     self.game.move_agent(predicted_move)
            self.game.move_agent(o1, o2, o3, test = True)
            self.game.draw(draw_score=True)
            pygame.display.update()



    def test_ai(self, net):
        """
        Test the AI 
        """
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(60)
            game_info = self.game.loop()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
                
            # Move agent; retrieve outputs for display
            o_values = self.move_agent(net, test=True)  # See edit below for move_agent!

            self.game.draw(draw_score=True)

            # Display O1, O2, and O3 values on the window
            if o_values is not None:
                O1, O2, O3, O4 = o_values
                font = pygame.font.Font(None, 36)
                text = font.render(f"O1: {O1:.3f}, O2: {O2:.3f}, O3: {O3:.3f}, O4: {O4:.3f}", True, (255, 255, 255))
                self.game.window.blit(text, (10, 10))

            pygame.display.update()


    def train_ai(self, genome, config, draw=False):
        """
        Train the AI by passing  NEAT neural networks and the NEAt config object.
   
        """
         
        run = True
        start_time = time.time()

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.genome = genome

        clock = 0
        while run:
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         return True
            clock+=1
            game_info = self.game.loop()

            self.move_agent(net )

            if draw:
                self.game.draw(draw_score=False, draw_hits=True)

                pygame.display.update()

            # duration = time.time() - start_time
            if self.game.optimalTime <= clock or game_info.food_collected >= game_info.total_food :
                self.calculate_fitness(game_info)
                break
        
            

        return False

    def move_agent(self, net, test=False):
        """
        Determine where to move the agent based on the neural network that controls them. 
        """
        net, agent = net, self.agent 

        sensor_inputs = []
        for sensor in self.agent.sensors:
            for cell in sensor:
                sensor_inputs.append(cell)  # Append (food, pheromone, nest) values
        if self.agent.carrying_food_receptor:
            sensor_inputs.append(self.agent.carrying_food) # Append carrying food value
            
        output = net.activate(sensor_inputs)  

        O1 = output[0]
        O2 = output[1]
        

        valid = self.game.move_agent(O1, O2)

        if test:
            return O1, O2
        return None




    def calculate_fitness(self, game_info):
        if self.sparse:
            self.genome.fitness += game_info.food_collected >= game_info.total_food
        else:
            self.genome.fitness += game_info.score



def eval_genomes(genomes, config):
    """
    Run each genome a set number of time to determine the fitness.
    """
    width, height = 900, 900
    draw = False
    win = None if not draw else pygame.display.set_mode((width, height))
    # pygame.display.set_caption("Forage")

    for i, (genome_id, genome) in enumerate(genomes):
        print(round(i/len(genomes) * 100), end=" ")
        genome.fitness = 0
        for run in range(NUM_RUNS):
            forage = ForageTask(win, width, height, arrangement_idx=run)

            force_quit = forage.train_ai(genome, config, draw)
            
            if force_quit:
                quit()
        genome.fitness /= NUM_RUNS
        # print(f"Genome {genome_id}:, Fitness: {genome.fitness}")



def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-85')
    checkpoint_file = ''
    # checkpoint_file = 'checkpoints/replication/769'
    

    # Resume training if checkpoint exists
    if os.path.exists(checkpoint_file):
        print("Resuming from checkpoint...")
        p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    else:
        print("Starting fresh training...")
        p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, None, 'checkpoints/simple/'))
    
    winner = p.run(eval_genomes, 400)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    best_fitness = stats.get_fitness_stat(max)
    avg_fitness = stats.get_fitness_mean()

    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness, label="Best Fitness", color='blue')
    plt.plot(avg_fitness, label="Average Fitness", color='orange')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("Fitness Over Generations")
    plt.grid()
    plt.show()


def test_best_network(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    width, height = 900, 900
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Forage")
    foragetask = ForageTask(win, width, height, arrangement_idx=1)
    foragetask.test_ai(winner_net)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-replication-plateau')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    test_best_network(config)
