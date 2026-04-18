import os
import random
import multiprocessing
import importlib.util
from collections import defaultdict
from copy import deepcopy

# Load the local submission module dynamically
spec = importlib.util.spec_from_file_location("submission", "submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

# Wrap the agent to accept a specific config
def make_agent(config):
    def wrapper(obs):
        return sub.agent(obs, override_config=config)
    return wrapper

# Default baseline configuration
DEFAULT_CONFIG = {
    "enemy_multiplier": 2.0,
    "finishing_multiplier": 1.5,
    "early_neutral_multiplier": 1.6,
    "safe_neutral_early_multiplier": 1.4,
    "contested_neutral_penalty": 0.25,
    "prod_weight": 15.0,
    "iw_weight": 3.0,
    "contested_margin": 1.4,
    "cost_turns_weight": 0.5,
    "funnel_finishing_ratio": 0.80,
    "funnel_ratio": 0.65,
}

# Mutation scales (std dev for Gaussian mutation)
SCALES = {
    "enemy_multiplier": 0.2,
    "finishing_multiplier": 0.15,
    "early_neutral_multiplier": 0.2,
    "safe_neutral_early_multiplier": 0.2,
    "contested_neutral_penalty": 0.1,
    "prod_weight": 3.0,
    "iw_weight": 1.0,
    "contested_margin": 0.15,
    "cost_turns_weight": 0.1,
    "funnel_finishing_ratio": 0.05,
    "funnel_ratio": 0.08,
}

def mutate(config):
    new_cfg = deepcopy(config)
    # Mutate 1-3 genes per generation
    num_mutations = random.randint(1, 3)
    keys = random.sample(list(new_cfg.keys()), num_mutations)
    for k in keys:
        new_cfg[k] += random.gauss(0, SCALES[k])
        # Keep positive bounds
        new_cfg[k] = max(0.01, round(new_cfg[k], 3))
    return new_cfg

def evaluate_match(match_args):
    # Isolated import for multiprocessing
    from kaggle_environments import make
    c1, c2, match_id = match_args
    env = make("orbit_wars", debug=False)
    
    a1 = make_agent(c1)
    a2 = make_agent(c2)
    env.run([a1, a2])
    
    steps = env.steps
    final = steps[-1]
    
    r1 = final[0].reward or 0
    r2 = final[1].reward or 0
    
    obs = final[0].observation
    all_pl = obs.planets
    p1_ships = sum(p[5] for p in all_pl if p[1] == 0)
    p2_ships = sum(p[5] for p in all_pl if p[1] == 1)
    
    # +1 win, 0 draw, -1 loss
    score = 1 if r1 > r2 else (-1 if r2 > r1 else 0)
    lead = p1_ships - p2_ships
    return match_id, score, lead

def run_tournament(population, games_per_variant=5):
    # Round-Robin / Swiss mix
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    match_args = []
    
    for i in range(len(population)):
        for _ in range(games_per_variant):
            # Pick random opponent
            while True:
                j = random.randint(0, len(population) - 1)
                if i != j: break
            # Each variant needs to play as player 1
            match_args.append((population[i], population[j], i))

    print(f"Running {len(match_args)} parallel matches...")
    results = pool.map(evaluate_match, match_args)
    pool.close()
    pool.join()
    
    fitness = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "total_lead": 0})
    for m_id, score, lead in results:
        if score == 1:
            fitness[m_id]["wins"] += 1
        elif score == -1:
            fitness[m_id]["losses"] += 1
        else:
            fitness[m_id]["draws"] += 1
        fitness[m_id]["total_lead"] += lead
        
    return fitness

if __name__ == "__main__":
    POP_SIZE = 5
    GENERATIONS = 2
    
    # Initialize pop
    population = [DEFAULT_CONFIG]
    for _ in range(POP_SIZE - 1):
        population.append(mutate(DEFAULT_CONFIG))
        
    print(f"Starting Evolution. Pop size: {POP_SIZE}, Generations: {GENERATIONS}")
    
    for gen in range(GENERATIONS):
        print(f"\\n=== Generation {gen+1} ===")
        fitness = run_tournament(population, games_per_variant=2)
        
        # Score calculation: 3 pts for win, 1 for draw
        scores = []
        for i in range(POP_SIZE):
            stat = fitness[i]
            points = stat["wins"] * 3 + stat["draws"] * 1
            avg_lead = stat["total_lead"] / max(1, (stat["wins"] + stat["losses"] + stat["draws"]))
            scores.append((points, avg_lead, i))
            
        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        print(f"Best Agent in Gen {gen+1}:")
        best_idx = scores[0][2]
        print(f"  Points: {scores[0][0]}, Avg Lead: {scores[0][1]:.1f}")
        print(f"  Config: {population[best_idx]}")
        
        # Elitism + Crossover + Mutation
        next_gen = [population[scores[0][2]], population[scores[1][2]]] # Keep top 2
        
        while len(next_gen) < POP_SIZE:
            # Tournament selection for parent
            p1 = population[random.choice(scores[:POP_SIZE//2])[2]]
            next_gen.append(mutate(p1))
            
        population = next_gen
        
    print("\\nEvolution Complete. Best global config discovered:")
    print(population[0])
