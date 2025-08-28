import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import random

# ----------- PARÁMETROS DE Q-LEARNING -----------
ALPHA = 0.2
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.01

# ----------- AGENTE -----------
class Robot(ap.Agent):
    
    def setup(self):
        self.has_cargo = False
        self.q_table = {}
        self.steps_in_epoch = 0
        self.epoch_durations = []
        self.current_epoch_path = []
        self.last_epoch_path = []
        
    def step(self):
        self.epsilon = self.model.p.epsilon
        current_pos = tuple(self.model.grid.positions[self])
        
        state = (current_pos, self.has_cargo)
        
        # Decidir la acción
        if random.random() < self.epsilon:
            action = self.model.random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        else:
            q_values = {a: self.q_table.get((state, a), 0) for a in [(0, 1), (0, -1), (1, 0), (-1, 0)]}
            if not q_values:
                action = self.model.random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            else:
                max_q = max(q_values.values())
                best_actions = [a for a, q in q_values.items() if q == max_q]
                action = self.model.random.choice(best_actions)
        
        self.action = action
        
    def learn(self, old_state, action, reward, new_state):
        old_q = self.q_table.get((old_state, action), 0)
        
        next_q_values = {a: self.q_table.get((new_state, a), 0) for a in [(0, 1), (0, -1), (1, 0), (-1, 0)]}
        if not next_q_values:
            max_next_q = 0
        else:
            max_next_q = max(next_q_values.values())
        
        new_q = old_q + ALPHA * (reward + GAMMA * max_next_q - old_q)
        self.q_table[(old_state, action)] = new_q

# ----------- MODELO -----------
class WarehouseModel(ap.Model):
    
    def setup(self):
        self.grid_size = (30, 30)
        self.grid = ap.Grid(self, self.grid_size, track_empty=True)
        
        self.warehouse_map = np.zeros(self.grid_size, dtype=int)
        self.warehouse_map[0:5, 0:5] = 1
        self.warehouse_map[25:30, 25:30] = 2
        self.warehouse_map[9:25, 5:8] = -1
        self.warehouse_map[9:25, 14:17] = -1
        self.warehouse_map[3:17, 22:25] = -1
        
        n_agents = 6
        self.agents = ap.AgentList(self, n_agents, Robot)

        free_cells = list(zip(*np.where(self.warehouse_map == 0)))
        positions = [tuple(self.random.choice(free_cells)) for _ in self.agents]
        self.grid.add_agents(self.agents, positions)
        
        self.p.epsilon = EPSILON_START
            
    def step(self):
        self.p.epsilon = max(EPSILON_MIN, self.p.epsilon * EPSILON_DECAY)
        
        for agent in self.agents:
            agent.step()
        
        for agent in self.agents:
            old_pos = tuple(self.grid.positions[agent])
            old_state = (old_pos, agent.has_cargo)
            action = agent.action
            
          
            agent.current_epoch_path.append(old_pos)
            agent.steps_in_epoch += 1
            
            next_pos_x, next_pos_y = old_pos[0] + action[0], old_pos[1] + action[1]
            next_pos = (next_pos_x, next_pos_y)
            reward = -1
            
            is_collision = False
            
            if not (0 <= next_pos_x < self.grid_size[0] and 0 <= next_pos_y < self.grid_size[1] and self.warehouse_map[next_pos] != -1):
                reward = -50
                is_collision = True
            
            for other_agent in self.agents:
                if other_agent != agent and tuple(self.grid.positions[other_agent]) == next_pos:
                    reward = -50
                    is_collision = True
                    break
            
            if not is_collision:
                self.grid.move_to(agent, next_pos)
                
            new_pos = tuple(self.grid.positions[agent])
            
            if self.warehouse_map[new_pos] == (2 if agent.has_cargo else 1):
                reward = 100
                
              
                agent.epoch_durations.append(agent.steps_in_epoch)
                agent.steps_in_epoch = 0
                agent.last_epoch_path = list(agent.current_epoch_path)
                agent.current_epoch_path = []
                
                if not agent.has_cargo:
                    agent.has_cargo = True
                else:
                    agent.has_cargo = False
            
            new_state = (new_pos, agent.has_cargo)
            agent.learn(old_state, action, reward, new_state)

    def update(self):
        # Deshabilitar la animación en cada paso
        pass
        

    def custom_cmap(self):
        from matplotlib.colors import ListedColormap, BoundaryNorm
        colors = ['red', 'white', 'blue', 'green']
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap

# Ejecutar simulación
if __name__ == '__main__':
    model = WarehouseModel()
    results = model.run(steps=200000)