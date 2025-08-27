import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import random

# ----------- PARÁMETROS DE Q-LEARNING -----------
ALPHA = 0.2 # que tan rapido el agente incorpora nueva info en la tabla Q, 1 = solo info nueva , 0 = solo info vieja
GAMMA = 0.95 # importancia de las recompensas a corto plazo vs largo plazo (0 = solo corto plazo, 1 = largo plazo)
EPSILON_START = 1.0 # con que frecuencia el agente elige una accion aleatoria en vez de la mejor conocida
EPSILON_DECAY = 0.9999 # tasa de decaimiento de epsilon por paso 
EPSILON_MIN = 0.01 # valor minimo de epsilon

# ----------- AGENTE -----------
class Robot(ap.Agent):
    
    def setup(self):
        self.has_cargo = False
        self.q_table = {}
        
    def step(self):
        self.epsilon = self.model.p.epsilon
        current_pos = tuple(self.model.grid.positions[self])
        
        state = (current_pos, self.has_cargo)
        
        # Decidir la acción: "exploración" vs. "explotación"
        if random.random() < self.epsilon:
            # Exploración: elige una acción aleatoria
            action = self.model.random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        else:
            # Explotación: elige la mejor acción de la tabla Q
            q_values = {a: self.q_table.get((state, a), 0) for a in [(0, 1), (0, -1), (1, 0), (-1, 0)]}
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = self.model.random.choice(best_actions)
        
        self.action = action
        
    def learn(self, old_state, action, reward, new_state):
        old_q = self.q_table.get((old_state, action), 0)
        
        next_q_values = {a: self.q_table.get((new_state, a), 0) for a in [(0, 1), (0, -1), (1, 0), (-1, 0)]}
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
        
        # El valor de epsilon se inicializa aquí
        self.p.epsilon = EPSILON_START
            
    def step(self):
        self.p.epsilon = max(EPSILON_MIN, self.p.epsilon * EPSILON_DECAY)
        
        for agent in self.agents:
            agent.step()
        
        for agent in self.agents:
            old_pos = tuple(self.grid.positions[agent])
            old_state = (old_pos, agent.has_cargo)
            action = agent.action
            
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
                if not agent.has_cargo:
                    agent.has_cargo = True
                else:
                    agent.has_cargo = False
            
            new_state = (new_pos, agent.has_cargo)
            
            agent.learn(old_state, action, reward, new_state)

    def update(self):
        plt.clf()
        grid_img = np.copy(self.warehouse_map)
        
        for agent in self.agents:
            x, y = self.grid.positions[agent]
            grid_img[x, y] = 3
        
        plt.imshow(grid_img, cmap=self.custom_cmap(), origin="upper")
        plt.title(f"Almacén - Paso {self.t}")
        plt.pause(0.0001)
        
    def end(self):
        plt.show()
        
    def custom_cmap(self):
        from matplotlib.colors import ListedColormap, BoundaryNorm
        colors = ['red', 'white', 'blue', 'green', 'purple']
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap

# Ejecutar simulación
model = WarehouseModel()
results = model.run(steps=50000)