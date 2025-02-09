from collections import defaultdict, deque

def bestPath(edges, root):
    # Build adjacency list
    tree = defaultdict(list)
    for u, v in edges:
        tree[u].append(v)
        tree[v].append(u)  # Since it's an undirected tree

    # BFS to compute depths
    depths = {root: 0}  # Dictionary to store depth of each node
    queue = deque([root])

    while queue:
        node = queue.popleft()
        for neighbor in tree[node]:
            if neighbor not in depths:  # Avoid visiting the parent node
                depths[neighbor] = depths[node] + 1
                queue.append(neighbor)

    #return the sum of all the depths
    return sum(depths.values())