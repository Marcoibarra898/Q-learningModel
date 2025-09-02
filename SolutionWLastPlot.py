import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation

# ----------- PARÁMETROS DE Q-LEARNING -----------
ALPHA = 0.2
GAMMA = 0.95
EPSILON_START = 0.5
EPSILON_DECAY = 0.9995   # Decae más rápido que 0.9999
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
        self.completed_full_epoch = False
        self.is_waiting = False
        
    def step(self):
        if self.is_waiting:
            self.action = (0, 0)
            return
            
        self.epsilon = self.model.p.epsilon
        current_pos = tuple(self.model.grid.positions[self])
        
        state = (current_pos, self.has_cargo)
        
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
        
        self.n_agents = 6
        self.agents = ap.AgentList(self, self.n_agents, Robot)

        # Cambiar a True si quieres muros aleatorios
        self.dynamic_walls = False

        self.spawn_walls()
        
        free_cells = list(zip(*np.where(self.warehouse_map == 0)))
        positions = [tuple(self.random.choice(free_cells)) for _ in self.agents]
        self.grid.add_agents(self.agents, positions)
        
        self.p.epsilon = EPSILON_START
        self.all_agents_completed = False
            
    def step(self):
        self.p.epsilon = max(EPSILON_MIN, self.p.epsilon * EPSILON_DECAY)
        
        if self.all_agents_completed:
            self.all_agents_completed = False
            self.reset_for_new_epoch()
            return
        
        planned_moves = {}
        for agent in self.agents:
            if not agent.is_waiting:
                agent.step()
            else:
                agent.action = (0, 0)
            
            old_pos = tuple(self.grid.positions[agent])
            old_state = (old_pos, agent.has_cargo)
            action = agent.action
            next_pos_x, next_pos_y = old_pos[0] + action[0], old_pos[1] + action[1]
            next_pos = (next_pos_x, next_pos_y)
            
            planned_moves[agent] = {'old_pos': old_pos, 'next_pos': next_pos, 'action': action, 'reward': -1, 'old_state': old_state}
        
        for agent, move in planned_moves.items():
            next_pos = move['next_pos']
            old_pos = move['old_pos']
            
            if not (0 <= next_pos[0] < self.grid_size[0] and 0 <= next_pos[1] < self.grid_size[1] and self.warehouse_map[next_pos] != -1):
                move['reward'] = -200
                move['next_pos'] = old_pos
                
            for other_agent, other_move in planned_moves.items():
                if agent != other_agent:
                    if next_pos == other_move['next_pos'] or (next_pos == other_move['old_pos'] and old_pos == other_move['next_pos']):
                        move['reward'] = -100
                        move['next_pos'] = old_pos
        
        completed_agents_count = 0
        for agent, move in planned_moves.items():
            old_pos = move['old_pos']
            action = move['action']
            reward = move['reward']
            next_pos = move['next_pos']
            old_state = move['old_state']
            
            if not agent.is_waiting:
                agent.steps_in_epoch += 1

            target_pos = (27, 27) if agent.has_cargo else (2, 2)
            old_dist = abs(old_pos[0] - target_pos[0]) + abs(old_pos[1] - target_pos[1])
            new_dist = abs(next_pos[0] - target_pos[0]) + abs(next_pos[1] - target_pos[1])
            
            if new_dist < old_dist:
                reward += 1   # Ajustado (antes +2)
            elif new_dist > old_dist:
                reward -= 1   # Ajustado (antes -2)
                
            if agent.has_cargo and self.warehouse_map[next_pos] == 1:
                reward = -50
            elif not agent.has_cargo and self.warehouse_map[next_pos] == 2:
                reward = -50
            
            self.grid.move_to(agent, next_pos)
            
            new_pos = tuple(self.grid.positions[agent])
            agent.current_epoch_path.append(new_pos)  # 🔹 Guardar ruta
            
            if self.warehouse_map[new_pos] == 2 and agent.has_cargo:
                reward = 100
                agent.epoch_durations.append(agent.steps_in_epoch)
                agent.steps_in_epoch = 0
                agent.last_epoch_path = list(agent.current_epoch_path)  # 🔹 Guardar última época
                agent.current_epoch_path = []
                agent.has_cargo = False
                agent.completed_full_epoch = True
                agent.is_waiting = True
            
            elif self.warehouse_map[new_pos] == 1 and not agent.has_cargo:
                reward = 100
                agent.has_cargo = True
            
            new_state = (new_pos, agent.has_cargo)
            agent.learn(old_state, action, reward, new_state)
            
            if agent.completed_full_epoch:
                completed_agents_count += 1
        
        if completed_agents_count == len(self.agents):
            self.all_agents_completed = True
            print("Todos los agentes han completado una época. Reiniciando el entorno.")

    def reset_for_new_epoch(self):
        for agent in self.agents:
            agent.completed_full_epoch = False
            agent.is_waiting = False
        
        if self.dynamic_walls:
            self.warehouse_map[self.warehouse_map == -1] = 0
            self.spawn_walls()
        
    def spawn_walls(self):
        if not self.dynamic_walls:
            # 🔹 Muros fijos de ejemplo
            self.warehouse_map[10:15, 5] = -1
            self.warehouse_map[20, 15:25] = -1
        else:
            num_walls = 3
            for _ in range(num_walls):
                start_row = self.random.randint(5, self.grid_size[0] - 5)
                start_col = self.random.randint(5, self.grid_size[1] - 5)
                length = self.random.randint(5, 15)
                is_horizontal = self.random.choice([True, False])
                
                if is_horizontal:
                    end_col = min(start_col + length, self.grid_size[1])
                    self.warehouse_map[start_row, start_col:end_col] = -1
                else:
                    end_row = min(start_row + length, self.grid_size[0])
                    self.warehouse_map[start_row:end_row, start_col] = -1

    def end(self):
        plt.figure(figsize=(12, 8))
        for i, agent in enumerate(self.agents):
            plt.plot(agent.epoch_durations, label=f'Agente {i+1}')

        plt.title('Duración de las épocas de los agentes a lo largo del tiempo')
        plt.xlabel('Número de Época')
        plt.ylabel('Duración de la Época (pasos)')
        plt.legend()
        plt.grid(True)
        plt.show()

        self.visualize_last_epoch()

    def visualize_last_epoch(self):
        fig, ax = plt.subplots(figsize=(10, 10))

        cmap = self.custom_cmap()
        im = ax.imshow(self.warehouse_map, cmap=cmap, interpolation='nearest', origin='lower')
        
        points = [ax.plot([], [], 'o', markersize=10, label=f'Agente {i+1}')[0] for i in range(len(self.agents))]
        
        ax.set_title('Última Época de los Agentes (completa)')
        ax.set_xlabel('Coordenada X')
        ax.set_ylabel('Coordenada Y')
        ax.legend()
        ax.grid(True)

        def update(frame):
            for i, agent in enumerate(self.agents):
                if not agent.last_epoch_path:
                    continue
                if frame < len(agent.last_epoch_path):
                    pos = agent.last_epoch_path[frame]
                    points[i].set_data([pos[1]], [pos[0]])
            return points

        max_path_length = max(len(agent.last_epoch_path) for agent in self.agents) if self.agents else 0
        
        if max_path_length > 0:
            ani = FuncAnimation(fig, update, frames=max_path_length, interval=50, blit=False, repeat=False)
            plt.show()
        else:
            print("No hay datos de la última época completa para visualizar.")
        
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
    results = model.run(steps=100000)  # 🔹 Reducido para probar más rápido
    