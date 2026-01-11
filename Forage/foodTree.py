import random
import math

def foodGenerator(root_coords, num_of_food, sensor_length, num_sensors):
    def polar_to_cartesian(origin, angle, length):
        return (
            origin[0] + length * math.cos(angle),
            origin[1] + length * math.sin(angle)
        )

    food_coords = [root_coords]
    edges = []
    max_children = 3
    angle_step = 2 * math.pi / num_sensors
    food_tree = {root_coords: []}

    viable_parents = [root_coords]
    angle_offsets = list(range(num_sensors))

    while len(food_coords) < num_of_food and viable_parents:
        parent = random.choice(viable_parents)
        used_angles = [child[1] for child in food_tree[parent]]
        available_angles = [a for a in angle_offsets if a not in used_angles]

        if not available_angles:
            viable_parents.remove(parent)
            continue

        angle_index = random.choice(available_angles)
        angle = angle_index * angle_step
        new_food = polar_to_cartesian(parent, angle, sensor_length)

        # Collision check
        too_close = any(
            math.dist(new_food, existing) < sensor_length * 0.5
            for existing in food_coords
        )
        if too_close:
            available_angles.remove(angle_index)
            continue

        food_coords.append(new_food)
        edges.append((parent, new_food))
        food_tree[parent].append((new_food, angle_index))
        food_tree[new_food] = []
        viable_parents.append(new_food)

        if len(food_tree[parent]) >= max_children:
            viable_parents.remove(parent)

    return food_coords[1:], edges  # Exclude root from food_coords
