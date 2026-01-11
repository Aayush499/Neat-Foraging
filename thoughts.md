## Markers

- # Life
    Should marker life be dependent on the game's clock? or the agent's movmement?

- # Cool things 
    That quadratic technique is actually kinda clever for detecting colissions

 
 

# Encoding
- Is there a way that I can encode certain relationships into integers?
- Example, I feel like there should be some way to turn the whole tuple <touching food> <pheromone strength> <nest boolean> into one single value

# physarum 
- This mold travels using slime. Once it finds a food source, it can signal through the slie that food has been found to the rest of its body
- There is something we ca learn from it about encoding information through pheromones specifically how it can tell when a slime leads to a dead end

# Movement
- movement is the 8 cardinal directions
- Specifically, we assign confidence values to teh 8 cardinal directions before choosign the highest one
- Would it be better to simply ask for the value of the angle of movement, then process it to map it to the 8 cardinal directions

# Fitness
- Should fitness just be the amount of food collected?
    - Should I divide it by duration? The max duration is already capped off at the optimal time it would take to find and retrieve everything
- Discouraging staying still. Is this good for learning?

# What does the agent know
- Currently al of the sensor readings + whether it is holding food or not
- Must it also know which direction it is facing?
    - Could that be important to know in order to understand the spatial relations between teh sensors?


# Bugs
- I could be wrong, but I think there's something wrong witht eh food detection with the sensors, but it seems to work fine when I control it. What's going on?