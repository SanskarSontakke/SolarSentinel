import os
import sys
import json
import time
from pathlib import Path
from kaggle_environments import make
import importlib.util

def load_agent(filepath: str):
    path = Path(filepath).resolve()
    spec = importlib.util.spec_from_file_location(f"agent_{path.stem}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent

def run_diagnostic(mode, agent_paths):
    agents = [load_agent(p) for p in agent_paths]
    env = make("orbit_wars", debug=True)
    
    print(f"--- Running Diagnostic Match [{mode}] ---")
    print(f"Agents: {agent_paths}")
    
    env.run(agents)
    
    steps = env.steps
    total_steps = len(steps)
    
    report = []
    
    # Analyze phases
    for interval in range(0, total_steps, 50):
        step_data = steps[interval]
        obs = step_data[0].observation
        planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
        
        ships = [0] * len(agents)
        prod = [0] * len(agents)
        planet_counts = [0] * len(agents)
        
        for p in planets:
            owner = p[1]
            if owner != -1:
                planet_counts[owner] += 1
                ships[owner] += p[5]
                prod[owner] += p[6]
        
        report.append({
            "step": interval,
            "ships": list(ships),
            "prod": list(prod),
            "planets": list(planet_counts)
        })

    # Final State
    final = steps[-1]
    final_obs = final[0].observation
    final_planets = final_obs.planets if hasattr(final_obs, "planets") else final_obs.get("planets", [])
    
    final_ships = [0] * len(agents)
    for p in final_planets:
        if p[1] != -1: final_ships[p[1]] += p[5]
        
    final_rewards = [s.reward or 0 for s in final]
    winner = final_rewards.index(max(final_rewards))

    diagnostic = {
        "mode": mode,
        "total_steps": total_steps,
        "winner": winner,
        "final_ships": final_ships,
        "timeline": report
    }
    
    print(f"--- Diagnostic Complete ---")
    print(f"Winner: P{winner} ({agent_paths[winner]})")
    print(f"Total Steps: {total_steps}")
    
    return diagnostic

if __name__ == "__main__":
    # Run 2p
    report_2p = run_diagnostic("2p", ["submission.py", "snapshots/test_agent.py"])
    with open("diagnostic_2p.json", "w") as f:
        json.dump(report_2p, f, indent=2)
        
    # Run 4p
    report_4p = run_diagnostic("1v3", ["submission.py", "snapshots/test_agent.py", "snapshots/test_agent.py", "snapshots/test_agent.py"])
    with open("diagnostic_4p.json", "w") as f:
        json.dump(report_4p, f, indent=2)
