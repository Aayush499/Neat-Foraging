# https://neat-python.readthedocs.io/en/latest/xor_example.html
import argparse
import subprocess
from Forage import Game
import pygame
import neat
import os
import time
import pickle
import matplotlib.pyplot as plt
from Forage.food import Food
import numpy as np
import glob
import csv
NUM_RUNS = 18
MAX_PLATEAU = 20  # Generations to wait before reset; adjust as needed
chosen_arrangement = [True] *18
# chosen_arrangement[6] = True
# chosen_arrangement = [False, True, True, True, True, True, False, True, False, False, False, False, True, True, True, True, False, False, ]
# chosen_arrangement = [False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, ]
WIDTH, HEIGHT = 1000, 1000

#create a function to generate a string prefix based on the parameters
def generate_prefix():
    global obstacles, particles, generations, movement_type, network_type, sub, ricochet, obstacle_type, seeded, o_switch
    return f"O{obstacles}-F{particles}-{movement_type}-G{generations}-N{network_type}-S{sub}-R{ricochet}-OT{obstacle_type}-SE{seeded}-OS{o_switch}"

class ForageTask:
    def __init__(self, window, width, height, arrangement_idx = 0):
        if window is None:
            os.environ["SDL_VIDEODRIVER"] = "dummy"  # Use a dummy display to prevent a window from opening
            pygame.display.init()  # Initialize Pygame without a window
            window = pygame.Surface((width, height))  # Create an off-screen surface
        global obstacles, particles, movement_type, ricochet, obstacle_type

        self.game = Game(window, width, height, arrangement_idx, obstacles=obstacles, particles=particles, ricochet=ricochet, obstacle_type=obstacle_type, seeded=seeded, o_switch=o_switch
        )
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
            place_pheromone = O4 > -0.5  # Whether it wants to place pheromones

            # Display network's decision
            # print(f"Predicted Move: {predicted_move}, Pheromone: {place_pheromone}")

            if place_pheromone:
                self.game.place_pheromone()
            # if auto:
            #     self.game.move_agent(predicted_move)
            self.game.move_agent(o1, o2, o3, test = True)
            self.game.draw(draw_score=True)
            pygame.display.update()



    # def test_ai(self, net):
    #     """
    #     Test the AI 
    #     """
    #     clock = pygame.time.Clock()
    #     run = True
    #     while run:
    #         clock.tick(60)
    #         game_info = self.game.loop()

    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 run = False
    #                 break
                
    #         # Move agent; retrieve outputs for display
    #         o_values = self.move_agent(net, test=True)  # See edit below for move_agent!

    #         self.game.draw(draw_score=True)

    #         # Display O1, O2, and O3 values on the window
    #         if o_values is not None:
    #             O1, O2, O3, O4 = o_values
    #             font = pygame.font.Font(None, 36)
    #             text = font.render(f"O1: {O1:.3f}, O2: {O2:.3f}, O3: {O3:.3f}, O4: {O4:.3f}", True, (255, 255, 255))
    #             self.game.window.blit(text, (10, 10))

    #         pygame.display.update()
    def test_ai(self, net, frame_dir='video_frames', global_frame=0):
        clock = pygame.time.Clock()

        run = True
        steps = 0
        while run and self.game.food_collected < self.game.total_food and steps < self.game.optimalTime:
            clock.tick(60)
            game_info = self.game.loop()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            self.game.draw(draw_score=True)
            sensor_inputs = self.move_agent(net, True)

            food_chk = False
            for sensor in self.agent.sensors:
            #if sensor[1] == -1 for all sensors, food not detected
                if sensor[1] -1:
                    food_chk = True
                    break
            if not food_chk:
                    
                    
                font = pygame.font.Font(None, 36)
                text = font.render("FOOD NOT DETECTED!", True, (255, 0, 0))  # Red color
                self.game.window.blit(text, (WIDTH/2 -100, HEIGHT/2 - 50)) 

            
            pygame.display.update()
            
            frame_filename = os.path.join(frame_dir, f"frame_{global_frame:05d}.png")
            pygame.image.save(self.game.window, frame_filename)
            global_frame += 1  # increment global frame on every image

            if self.game.optimalTime <= steps or game_info.food_collected >= game_info.total_food:
                break
            steps += 1

            

        
        return global_frame  # Return updated counter!



   




    def train_ai(self, genome, config, draw=False):
        """
        Train the AI by passing  NEAT neural networks and the NEAt config object.
   
        """
         
        run = True
        start_time = time.time()
        draw = False
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.genome = genome

        clock = 0

        if draw:
            win = pygame.display.set_mode((WIDTH, HEIGHT))
            self.game.window = win

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
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
        O3 = output[2]
        O4 = output[3]    
        place_pheromone = O4 > -0.25  # Whether it wants to place pheromones
        
        if place_pheromone:
            self.game.place_pheromone()
        
            



        valid = self.game.move_agent(O1, O2, O3)
       

        

        return sensor_inputs




    def calculate_fitness(self, game_info):
        if self.sparse:
            self.genome.fitness += game_info.food_collected >= game_info.total_food
        else:
            self.genome.fitness += game_info.score



def eval_genomes(genomes, config):
    """
    Run each genome a set number of time to determine the fitness.
    """
    width, height = WIDTH, HEIGHT
    draw = False
    win = None if not draw else pygame.display.set_mode((width, height))
    # pygame.display.set_caption("Forage")

    for i, (genome_id, genome) in enumerate(genomes):
        print(round(i/len(genomes) * 100), end=" ")
        genome.fitness = 0
        for run in range(NUM_RUNS):
           
            if not chosen_arrangement[run]:
                continue
            

            forage = ForageTask(win, width, height, arrangement_idx=run)

            force_quit = forage.train_ai(genome, config, draw)
            
            if force_quit:
                quit()
        genome.fitness /= NUM_RUNS
        # print(f"Genome {genome_id}:, Fitness: {genome.fitness}")

def get_latest_checkpoint(dir_path, pattern_prefix):
    files = glob.glob(os.path.join(dir_path, f"{pattern_prefix}*"))
    if not files:
        return ''  # No checkpoint files found
    # Sort files by creation time (or use modified time: os.path.getmtime)
    latest_file = max(files, key=os.path.getctime)
    return latest_file



def run_neat(config):
    global obstacles, particles, generations, movement_type, network_type, sub, use_checkpoint
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-85')
    prefix_string = generate_prefix()
    checkpoint_file = ''
    checkpoint_dir = 'checkpoints/'
    # checkpoint_file = 'checkpoints/replication/769'
    
    
    #create checkpoint name based on parameters
    prefix_string = generate_prefix()
    checkpoint_subdir = f"{checkpoint_dir}checkpoint-{prefix_string}/"
    

    csv_filename = os.path.join(checkpoint_subdir, 'fitness_history.csv')
    csv_filename_bigness = os.path.join(checkpoint_subdir, 'bigness_history.csv')



    os.makedirs(checkpoint_subdir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_subdir, 'neat-checkpoint-')
    checkpoint_file = get_latest_checkpoint(checkpoint_subdir, 'neat-checkpoint-')
    print("Checkpoint file:", checkpoint_file)

    if use_checkpoint != "":
        checkpoint_file = use_checkpoint
        print("Using user-specified checkpoint:", checkpoint_file)
    
    temp_generations = generations
    # Resume training if checkpoint exists
    if os.path.exists(checkpoint_file):
        print("Resuming from checkpoint...")
        #restore from the last file in the directory checkpoint_file
        
        p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        # Extract the generation number from the filename
        if use_checkpoint == "":
            last_gen = int(os.path.basename(checkpoint_file).split('-')[-1])
        else:
            last_gen = 0
        
        temp_generations -= last_gen  # Adjust remaining generations
        if temp_generations <= 0:
            print("Training already completed in previous runs.")
            return
        print(f"Continuing training for {temp_generations} more generations...")

    else:
        print("Starting fresh training...")
        p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, None, filename_prefix=checkpoint_prefix))
    
    winner = p.run(eval_genomes, temp_generations)
    best_dir = "best_networks"
    os.makedirs(best_dir, exist_ok=True)
    with open(os.path.join(best_dir, f"best-{prefix_string}.pickle"), "wb") as f:
        pickle.dump(winner, f)

    def append_fitness_data(best_fitness, avg_fitness):
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            for best, avg in zip(best_fitness, avg_fitness):
                writer.writerow([best, avg])

    def append_largeness_data(largest_genomes):
        with open(csv_filename_bigness, 'a', newline='') as f:
            writer = csv.writer(f)
            for genome in largest_genomes:
                writer.writerow([genome.key, len(genome.nodes), len(genome.connections)])

    best_genomes = stats.most_fit_genomes
    best_genomes_dir = os.path.join(checkpoint_subdir, "best_genomes_by_generation")
    os.makedirs(best_genomes_dir, exist_ok=True)
    for gen, genome in enumerate(best_genomes):
        filename = os.path.join(best_genomes_dir, f"best_genome_gen_{gen}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(genome, f)

    best_fitness = stats.get_fitness_stat(max)
    avg_fitness = stats.get_fitness_mean()
    largest_genomes_over_time = stats.most_big_genomes
    append_fitness_data(best_fitness, avg_fitness)
    append_largeness_data(largest_genomes_over_time)

    

    

    def load_fitness_history(csv_filename):
        best_data, avg_data = [], []
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Safeguard in case of blank lines
                if len(row) < 2:
                    continue
                best_data.append(float(row[0]))
                avg_data.append(float(row[1]))
        return best_data, avg_data

    def load_bigness_history(csv_filename_bigness):
        bigness_data = []
        with open(csv_filename_bigness, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Safeguard in case of blank lines
                if len(row) < 3:
                    continue
                bigness_data.append((int(row[0]), int(row[1]), int(row[2])))  # (genome_id, num_nodes, num_connections)
        return bigness_data
    # Usage before plotting:
    all_best_fitness, all_avg_fitness = load_fitness_history(csv_filename)
    all_largest_genomes = load_bigness_history(csv_filename_bigness)
     

    node_counts = [row[1] for row in all_largest_genomes]
    conn_counts = [row[2] for row in all_largest_genomes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Fitness plot
    ax1.plot(all_best_fitness, label="Best Fitness", color="blue")
    ax1.plot(all_avg_fitness, label="Avg Fitness", color="orange")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.grid()

    # Bigness plot
    ax2.plot(node_counts, label="Largest Genome Nodes", color="green")
    ax2.plot(conn_counts, label="Largest Genome Conns", color="purple", linestyle="dashed")
    ax2.set_ylabel("Network Size")
    ax2.set_xlabel("Generation")
    ax2.legend()
    ax2.grid()

    plt.suptitle("Fitness and Complexity Over Generations")
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_subdir, "fitness_and_bigness_stacked.png"))
    plt.show()


   
   



# def test_best_network(config):
#     with open("best.pickle", "rb") as f:
#         winner = pickle.load(f)
#     for i in range(NUM_RUNS):
#         if not chosen_arrangement[i]:
#             continue
#         winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

#         width, height = 900, 900
#         win = pygame.display.set_mode((width, height))
#         pygame.display.set_caption("Forage")
#         foragetask = ForageTask(win, width, height, arrangement_idx=i)
#         foragetask.test_ai(winner_net)

def create_video_from_frames(frame_dir, output_filename, framerate=30):
    # The frame filenames should be named sequentially as frame_00000.png, frame_00001.png, etc.
    # FFmpeg pattern for input files: frame_%05d.png (same as your saving code)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",                       # Overwrite without asking
        "-framerate", str(framerate),
        "-i", os.path.join(frame_dir, "frame_%05d.png"),
        "-c:v", "libx264",          # Codec
        "-profile:v", "high",
        "-crf", "20",               # Quality: lower is better (18-23 is common)
        "-pix_fmt", "yuv420p",      # Pixel format
        output_filename
    ]
    subprocess.run(ffmpeg_cmd, check=True)

def test_best_network(config):
    global obstacles, particles, generations, movement_type, network_type, sub, ricochet, best_file
    prefix_string = generate_prefix()
    frame_dir = f'video_dir/{prefix_string}_frames'
    # os.makedirs(frame_dir, exist_ok=True)
    #make directory if it doesn't exist, if it exists, delete all files in it
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    else:
        files = glob.glob(os.path.join(frame_dir, '*'))
        for f in files:
            os.remove(f)
            
    best_dir = "best_networks"
    filename = os.path.join(best_dir, f"best-{prefix_string}.pickle")
    if best_file != "":
        filename = os.path.join(best_dir, f"{best_file}.pickle")

    if not os.path.exists(filename):
        print(f"Best network file {filename} not found!")
        return
    with open(filename, "rb") as f:
        winner = pickle.load(f)

    global_frame = 0  # NEW: global frame counter

    for i in range(NUM_RUNS):
        if not chosen_arrangement[i]:
            continue
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        width, height = WIDTH, HEIGHT
        win = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Forage")
        foragetask = ForageTask(win, width, height, arrangement_idx=i)

        # Pass the global_frame **by reference** & get back updated value
        global_frame = foragetask.test_ai(winner_net, frame_dir=frame_dir, global_frame=global_frame)

    video_name = f"{prefix_string}.mp4"
    output_path = os.path.join("video_dir", video_name)
    create_video_from_frames(frame_dir, output_path, framerate=30)
    print("Video created at:", output_path)


def parser():
    import argparse
    parser = argparse.ArgumentParser(description="Run NEAT Foraging Task")
    parser.add_argument("--particles", type=int, default=2, help="Number of food particles")
    parser.add_argument("--obstacles", type=str, default="False", help="Use obstacles or not")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")
    # parser.add_argument("--config", type=str, default="config-replication-plateau", help="Config filename")
    parser.add_argument("--movement_type", type=str, default="holonomic", help="Type of agent movement"
                        )
    parser.add_argument("--network", type=str, default="ff", help="Type of neural network")
    parser.add_argument("--test", type=str, default="False", help="Test the best network after training")
    #add an argument for adding a sub number for multiple runs
    parser.add_argument("--sub", type=str, default="0", help="Sub title for multiple runs")
    parser.add_argument("--ricochet", type=str, default="False", help="Ricochet off walls or not")
    parser.add_argument("--best", type=str, default="", help="Best network file to test")
    parser.add_argument("--obstacle_type", type=str, default="line", help="Type of obstacle arrangement")
    parser.add_argument("--seeded", type=str, default="False", help="Use seeded random or not") 
    parser.add_argument("--orientation_switching", type=str, default="True", help="Use orientation switching or not")
    parser.add_argument("--use_checkpoint", type=str, default="", help="Use checkpoint or not")
    args = parser.parse_args()
    return args
    

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    
    # config_path = os.path.join(local_dir, 'config-replication-plateau')
    args = parser()
    global obstacles, particles, generations, movement_type, network_type, sub, best_file, ricochet, obstacle_type, seeded, o_switch, use_checkpoint
    use_checkpoint = args.use_checkpoint
    seeded = str2bool(args.seeded)
    obstacle_type = args.obstacle_type
    obstacles = str2bool(args.obstacles)
    particles = args.particles
    generations = args.generations
    movement_type = args.movement_type
    network_type = args.network
    o_switch = str2bool(args.orientation_switching)
    sub = args.sub
    test_run = str2bool(args.test)
    ricochet = str2bool(args.ricochet)
    best_file = args.best
    if args.network == "ff":
        config_filename = 'config-simple-ff'
    elif args.network == "recursive":
        config_filename = 'config-simple-recursive'
    else:
        raise ValueError("Invalid network type. Choose 'ff' or 'recursive'.")
    config_path = os.path.join(local_dir+'/configs/', config_filename)

    # config_path = os.path.join(local_dir, args.config)
    print("Using config file:", config_path)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)


    if not test_run:
        run_neat(config)
    test_best_network(config)
