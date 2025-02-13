## Initial Test
- Just running basic experiments on my first set up
- Used the config.text file for onfigurations
    ### Result
    -  No notieable improvement after 400 generations


## 4 sensor test
- reduced sensors to 4 and sensor segments to 4
- changed configurations to config-base

## Redecued segment test
- Now only pheromones are detected through segments.
- Food detected returns the distance of the food to teh agent.
- ### FAiled after 2000 runs
    - only seemed to be able to solve North and South cases
    - made some changes to config:
        - 3. Increase Population Size

        - Your population size is 70, which is quite small for complex behaviors. Try increasing it to 300–500 to allow for more diverse strategies.

        4. Adjust Mutation Parameters

            Lower conn_delete_prob (0.5 → 0.2): Prevents the model from randomly removing useful connections too often.
            Increase bias_mutate_power (0.5 → 1.0): Helps adjust neuron biases more drastically, which may improve symmetry learning.
            Increase weight_mutate_power (0.5 → 1.0): Similar to biases, this allows for more significant updates in learned behavior.
