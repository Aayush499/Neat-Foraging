# https://neat-python.readthedocs.io/en/latest/xor_example.html
from Forage import Game
import pygame
import neat
import os
import time
import pickle
import matplotlib.pyplot as plt
from Forage.food import Food
NUM_RUNS = 5

class ForageTask:
    def __init__(self, window, width, height):
        if window is None:
            os.environ["SDL_VIDEODRIVER"] = "dummy"  # Use a dummy display to prevent a window from opening
            pygame.display.init()  # Initialize Pygame without a window
            window = pygame.Surface((width, height))  # Create an off-screen surface

        self.game = Game(window, width, height)
        self.foods = self.game.food_list
        self.agent = self.game.agent
        self.pheromone = self.game.pheromones
        self.nest = self.game.nest
        
    def manual_test(self, net):
        """
        Manually control the agent using WASD keys and observe the network's outputs.
        """
        clock = pygame.time.Clock()
        run = True
        
        while run:
            clock.tick(60)
            self.game.loop()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            # Capture key presses
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_w]:
                move_direction = "N"
            elif keys[pygame.K_s]:
                move_direction = "S"
            elif keys[pygame.K_a]:
                move_direction = "W"
            elif keys[pygame.K_d]:
                move_direction = "E"
            else:
                move_direction = None  # No movement
            #capture mouse click
            click = pygame.mouse.get_pressed()
            if click[0]:
                newFood = Food(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])
                self.game.food_list.append(newFood)
            

                
            # Move agent manually
            if move_direction:
                self.game.move_agent(move_direction)

            # Gather sensor inputs for network evaluation
            sensor_inputs = []
            for sensor in self.agent.sensors:
                for cell in sensor:
                    sensor_inputs.append(cell)  # Append (food, pheromone, nest) values
            sensor_inputs.append(self.agent.carrying_food)  # Append carrying food value
            
            # Get the network's response
            output = net.activate(sensor_inputs)
            move_actions = ["N", "E", "S", "W"]
            action_index = output.index(max(output[:4]))  # Get predicted move direction
            predicted_move = move_actions[action_index]
            place_pheromone = output[4] > 0.5  # Whether it wants to place pheromones

            # Display network's decision
            print(f"Predicted Move: {predicted_move}, Pheromone: {place_pheromone}")

            if place_pheromone:
                self.game.place_pheromone()

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

            sensor_inputs = []
            for sensor in self.agent.sensors:
                for cell in sensor:
                    sensor_inputs.append(cell)  # Append (food, pheromone, nest) values
            sensor_inputs.append(self.agent.carrying_food) # Append carrying food value

            output = net.activate(sensor_inputs)  # NEAT expects a list of inputs

            # move_actions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            move_actions = ["N", "E", "S", "W"]
            # action_index = output.index(max(output[:8]))
            action_index  = output.index(max(output[:4]))
            move_direction = move_actions[action_index] 
            # place_pheromone = output[8] > 0.5
            place_pheromone = output[4] > 0.5
            # movement = output[9] > 0.5


            # if movement:
            #     self.game.move_agent(move_direction)
            self.game.move_agent(move_direction)
            if place_pheromone:
                self.game.place_pheromone()



             
            self.game.draw(draw_score=True)
            pygame.display.update()

    def train_ai(self, genome, config, draw=False):
        """
        Train the AI by passing two NEAT neural networks and the NEAt config object.
        These AI's will play against eachother to determine their fitness.
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

            self.move_agent(net, self.genome)

            if draw:
                self.game.draw(draw_score=False, draw_hits=True)

                pygame.display.update()

            # duration = time.time() - start_time
            if self.game.optimalTime <= clock or game_info.food_collected >= game_info.total_food :
                self.calculate_fitness(game_info)
                break
        
            

        return False

    def move_agent(self, net, genome):
        """
        Determine where to move the agent  based on the  
        neural network that control them. 
        """
         
        net, agent  = net, self.agent 

        sensor_inputs = []
        for sensor in self.agent.sensors:
            for cell in sensor:
                sensor_inputs.append(cell)  # Append (food, pheromone, nest) values
        sensor_inputs.append(self.agent.carrying_food) # Append carrying food value
        
        output = net.activate(sensor_inputs)  # NEAT expects a list of inputs
        # move_actions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        move_actions = ["N", "E", "S", "W"]
        action_index = output.index(max(output[:4]))
        move_direction = move_actions[action_index] 
        # place_pheromone = output[8] > 0.5
        place_pheromone = output[4] > 0.5
        # movement = output[9] > 0.5

        valid = True

        # if movement:
        #     valid =self.game.move_agent(move_direction)
        valid = self.game.move_agent(move_direction)
        # else:
        #     genome.fitness -= 0.01  # we want to discourage no movement
        if place_pheromone:
            self.game.place_pheromone()


        
         
        # if not valid:  # If the movement makes the paddle go off the screen punish the AI
        #     genome.fitness -= 1

    def calculate_fitness(self, game_info):
        self.genome.fitness += game_info.score



def eval_genomes(genomes, config):
    """
    Run each genome a set number of time to determine the fitness.
    """
    width, height = 700, 500
    draw = False
    win = None if not draw else pygame.display.set_mode((width, height))
    # pygame.display.set_caption("Forage")

    for i, (genome_id, genome) in enumerate(genomes):
        print(round(i/len(genomes) * 100), end=" ")
        genome.fitness = 0
        for run in range(NUM_RUNS):
            forage = ForageTask(win, width, height)

            force_quit = forage.train_ai(genome, config, draw)
            
            if force_quit:
                quit()
        genome.fitness /= NUM_RUNS
        # print(f"Genome {genome_id}:, Fitness: {genome.fitness}")



def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-85')
    checkpoint_file = 'checkpoints/biased_west/49'
    # checkpoint_file = ''

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
    p.add_reporter(neat.Checkpointer(50, None, 'checkpoints/biased_west/'))
  
    winner = p.run(eval_genomes, 1)
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

    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Forage")
    foragetask = ForageTask(win, width, height)
    foragetask.test_ai(winner_net)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-reduced-input')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    test_best_network(config)
