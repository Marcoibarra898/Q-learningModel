import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle
import os

# ----------- PARAMETROS DE Q-LEARNING -----------
ALPHA = 0.2
GAMMA = 0.95
EPSILON_START = 0.5
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01

# ----------- PARAMETROS DE LA SIMULACION -----------
ERROR_PROBABILITY = 0.0005
ERROR_COOLDOWN_STEPS = 50
LOW_BATTERY_THRESHOLD = 30
BATTERY_DEPLETION_PER_STEP = 0.2
BATTERY_IDLE_DEPLETION = 0.05
BATTERY_RECHARGE_RATE = 2.0

# ----------- AGENTE -----------
class Robot(ap.Agent):
    
    def setup(self):
        self.has_cargo = False
        self.current_task = 'get_main_cargo'
        self.status = 'normal'
        self.error_cooldown_timer = 0
        self.battery_level = 100
        self.steps_in_epoch = 0
        self.epoch_durations = []
        self.current_epoch_path = []
        self.last_epoch_path = []
        self.battery_history = [100]
        self.error_incidents = 0
        
    def step(self):
        if self.status == 'in_error':
            if self.error_cooldown_timer > 0:
                self.error_cooldown_timer -= 1
                self.action = (0, 0)
            else:
                self.status = 'normal'
                self.current_task = self.model.random.choice(['get_main_cargo', 'get_shelf_cargo'])
                self.action = (0, 0)
            return
        
        self.epsilon = self.model.p.epsilon
        
        if self.battery_level <= 0:
            self.battery_level = 0
            self.action = (0, 0)
            return
        
        if self.current_task == 'recharge' and self.battery_level < 100:
            self.action = (0, 0)
            self.battery_level = min(100, self.battery_level + BATTERY_RECHARGE_RATE)
            return

        self.battery_level -= BATTERY_DEPLETION_PER_STEP
        self.battery_history.append(self.battery_level)
        
        battery_range = self.get_battery_range()
        current_pos = tuple(self.model.grid.positions[self])
        state = (current_pos, self.has_cargo, self.current_task, self.status, battery_range)
        
        if random.random() < self.epsilon:
            action = self.model.random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        else:
            q_values = {a: self.model.q_table.get((state, a), 0) for a in [(0, 1), (0, -1), (1, 0), (-1, 0)]}
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = self.model.random.choice(best_actions)
        
        self.action = action
        
    def get_battery_range(self):
        if self.battery_level > 75:
            return 'full'
        elif self.battery_level > 30:
            return 'medium'
        elif self.battery_level > 0:
            return 'low'
        else:
            return 'empty'
            
    def learn(self, old_state, action, reward, new_state):
        old_q = self.model.q_table.get((old_state, action), 0)
        
        next_q_values = {a: self.model.q_table.get((new_state, a), 0) for a in [(0, 1), (0, -1), (1, 0), (-1, 0)]}
        max_next_q = max(next_q_values.values()) if next_q_values else 0
        
        new_q = old_q + ALPHA * (reward + GAMMA * max_next_q - old_q)
        self.model.q_table[(old_state, action)] = new_q

# ----------- MODELO -----------
class WarehouseModel(ap.Model):
    
    def setup(self):
        self.grid_size = (30, 30)
        self.grid = ap.Grid(self, self.grid_size, track_empty=True)
        
        self.n_agents = 6
        self.agents = ap.AgentList(self, self.n_agents, Robot)
        
        self.dynamic_walls = True
        
        if os.path.exists('q_table.pkl'):
            with open('q_table.pkl', 'rb') as f:
                self.q_table = pickle.load(f)
            print("Tabla Q cargada. Reanudando entrenamiento...")
        else:
            self.q_table = {}
            print("No se encontro tabla Q. Iniciando entrenamiento desde cero...")
        
        self.create_new_map()
        
        free_cells = list(zip(*np.where(self.warehouse_map == 0)))
        
        initial_positions = []
        if free_cells:
            for _ in range(self.n_agents):
                initial_positions.append(tuple(self.random.choice(free_cells)))
        
        self.grid.add_agents(self.agents, initial_positions)

        for agent in self.agents:
            agent.current_task = 'get_main_cargo'
            agent.has_cargo = False
            agent.status = 'normal'
            agent.battery_level = 100
        
        self.p.epsilon = EPSILON_START
            
    def _get_target_pos(self, agent):
        if agent.status == 'in_error':
            waiting_zone_coords = np.argwhere(self.warehouse_map == 4)
            if len(waiting_zone_coords) > 0:
                distances = np.sum(np.abs(waiting_zone_coords - np.array(self.grid.positions[agent])), axis=1)
                return waiting_zone_coords[np.argmin(distances)]
            else:
                return self.grid.positions[agent]
        
        if agent.current_task == 'recharge':
            recharge_zone_coords = np.argwhere(self.warehouse_map == 5)
            if len(recharge_zone_coords) > 0:
                distances = np.sum(np.abs(recharge_zone_coords - np.array(self.grid.positions[agent])), axis=1)
                return recharge_zone_coords[np.argmin(distances)]
            else:
                return self.grid.positions[agent]

        load_zone_coords = np.argwhere(self.warehouse_map == 1)
        unload_zone_coords = np.argwhere(self.warehouse_map == 2)
        shelf_coords = np.argwhere(self.warehouse_map == 3)

        load_zone_pos = load_zone_coords.mean(axis=0) if len(load_zone_coords) > 0 else self.grid.positions[agent]
        unload_zone_pos = unload_zone_coords.mean(axis=0) if len(unload_zone_coords) > 0 else self.grid.positions[agent]
        
        if agent.current_task == 'get_main_cargo':
            return load_zone_pos
        elif agent.current_task == 'deliver_cargo':
            return unload_zone_pos
        elif agent.current_task == 'get_shelf_cargo':
            if len(shelf_coords) > 0:
                distances = np.sum(np.abs(shelf_coords - np.array(self.grid.positions[agent])), axis=1)
                return shelf_coords[np.argmin(distances)]
            else:
                return self.grid.positions[agent]
        return self.grid.positions[agent]

    def step(self):
        self.p.epsilon = max(EPSILON_MIN, self.p.epsilon * EPSILON_DECAY)
        
        for agent in self.agents:
            if agent.status == 'normal' and self.random.random() < ERROR_PROBABILITY:
                agent.status = 'in_error'
                agent.has_cargo = False
                agent.error_incidents += 1
            
            if agent.status == 'normal' and agent.current_task != 'recharge' and agent.battery_level < LOW_BATTERY_THRESHOLD:
                print(f"Agente {agent.id} tiene bateria baja. Tarea de recarga asignada.")
                agent.current_task = 'recharge'

        planned_moves = {}
        for agent in self.agents:
            agent.step()
            
            old_pos = tuple(self.grid.positions[agent])
            old_state = (old_pos, agent.has_cargo, agent.current_task, agent.status, agent.get_battery_range())
            action = agent.action
            next_pos_x, next_pos_y = old_pos[0] + action[0], old_pos[1] + action[1]
            next_pos = (next_pos_x, next_pos_y)
            
            planned_moves[agent] = {'old_pos': old_pos, 'next_pos': next_pos, 'action': action, 'reward': -1, 'old_state': old_state}
        
        for agent, move in planned_moves.items():
            next_pos = move['next_pos']
            old_pos = move['old_pos']
            
            if not (0 <= next_pos[0] < self.grid_size[0] and 0 <= next_pos[1] < self.grid_size[1] and self.warehouse_map[next_pos] != -1):
                move['reward'] = -100
                move['next_pos'] = old_pos
            
            for other_agent, other_move in planned_moves.items():
                if agent != other_agent:
                    if next_pos == other_move['next_pos'] or (next_pos == other_move['old_pos'] and old_pos == other_move['next_pos']):
                        move['reward'] = -50
                        move['next_pos'] = old_pos
        
        for agent, move in planned_moves.items():
            old_pos = move['old_pos']
            action = move['action']
            reward = move['reward']
            next_pos = move['next_pos']
            old_state = move['old_state']
            
            if agent.status != 'in_error' and agent.current_task != 'recharge' and not (old_pos == next_pos and agent.battery_level <= 0):
                agent.steps_in_epoch += 1
            
            target_pos = self._get_target_pos(agent)
            
            current_agent_pos_np = np.array(old_pos)
            target_pos_np = np.array(target_pos)
            next_pos_np = np.array(next_pos)

            old_dist = np.abs(current_agent_pos_np - target_pos_np).sum()
            new_dist = np.abs(next_pos_np - target_pos_np).sum()
            
            if new_dist < old_dist:
                reward += 10
            elif new_dist > old_dist:
                reward -= 10
            
            if agent.has_cargo and self.warehouse_map[next_pos] == 1 and agent.current_task != 'deliver_cargo':
                reward = -50
            elif not agent.has_cargo and self.warehouse_map[next_pos] == 2 and agent.current_task != 'get_main_cargo':
                reward = -50
            
            if old_pos == next_pos and agent.battery_level <= 0:
                reward = -500

            self.grid.move_to(agent, next_pos)
            new_pos = tuple(self.grid.positions[agent])
            agent.current_epoch_path.append(new_pos)
            
            if agent.status == 'in_error':
                if self.warehouse_map[new_pos] == 4:
                    reward = 200
                    agent.error_cooldown_timer = ERROR_COOLDOWN_STEPS
                    agent.last_epoch_path = list(agent.current_epoch_path)
                    agent.current_epoch_path = []
            
            elif agent.current_task == 'recharge':
                if self.warehouse_map[new_pos] == 5 and agent.battery_level >= 100:
                    reward = 150
                    agent.current_task = self.random.choice(['get_main_cargo', 'get_shelf_cargo'])
                elif self.warehouse_map[new_pos] == 5 and agent.battery_level < 100:
                    reward = 50

            elif agent.current_task == 'deliver_cargo':
                if self.warehouse_map[new_pos] == 2 and agent.has_cargo:
                    reward = 100
                    agent.epoch_durations.append(agent.steps_in_epoch)
                    agent.steps_in_epoch = 0
                    agent.last_epoch_path = list(agent.current_epoch_path)
                    agent.current_epoch_path = []
                    agent.has_cargo = False
                    
                    if agent.battery_level < LOW_BATTERY_THRESHOLD:
                        agent.current_task = 'recharge'
                    else:
                        agent.current_task = self.random.choice(['get_main_cargo', 'get_shelf_cargo'])

            elif agent.current_task == 'get_main_cargo':
                if self.warehouse_map[new_pos] == 1 and not agent.has_cargo:
                    reward = 100
                    agent.has_cargo = True
                    agent.current_task = 'deliver_cargo'
            
            elif agent.current_task == 'get_shelf_cargo':
                if self.warehouse_map[new_pos] == 3 and not agent.has_cargo:
                    reward = 100
                    agent.has_cargo = True
                    agent.current_task = 'deliver_cargo'

            new_state = (new_pos, agent.has_cargo, agent.current_task, agent.status, agent.get_battery_range())
            agent.learn(old_state, action, reward, new_state)

    def create_new_map(self):
        self.warehouse_map = np.zeros(self.grid_size, dtype=int)
        zone_size = (5, 5)

        load_x, load_y = self.random.randint(0, self.grid_size[0] - zone_size[0]), self.random.randint(0, self.grid_size[1] - zone_size[1])
        self.warehouse_map[load_x:load_x + zone_size[0], load_y:load_y + zone_size[1]] = 1

        while True:
            unload_x, unload_y = self.random.randint(0, self.grid_size[0] - zone_size[0]), self.random.randint(0, self.grid_size[1] - zone_size[1])
            if np.sum(self.warehouse_map[unload_x:unload_x + zone_size[0], unload_y:unload_y + zone_size[1]] != 0) == 0:
                self.warehouse_map[unload_x:unload_x + zone_size[0], unload_y:unload_y + zone_size[1]] = 2
                break
                
        if self.dynamic_walls:
            num_walls = 3
            for _ in range(num_walls):
                start_row, start_col = self.random.randint(5, self.grid_size[0] - 5), self.random.randint(5, self.grid_size[1] - 5)
                length = self.random.randint(5, 15)
                is_horizontal = self.random.choice([True, False])
                
                if is_horizontal:
                    end_col = min(start_col + length, self.grid_size[1])
                    if np.sum(self.warehouse_map[start_row, start_col:end_col] != 0) == 0:
                        self.warehouse_map[start_row, start_col:end_col] = -1
                else:
                    end_row = min(start_row + length, self.grid_size[0])
                    if np.sum(self.warehouse_map[start_row:end_row, start_col] != 0) == 0:
                        self.warehouse_map[start_row:end_row, start_col] = -1
        
        wall_coords = np.argwhere(self.warehouse_map == -1)
        num_estantes = 4
        if len(wall_coords) > 0:
            for _ in range(num_estantes):
                chosen_wall_pos = self.random.choice(wall_coords)
                possible_shelf_positions = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    adj_x, adj_y = chosen_wall_pos[0] + dx, chosen_wall_pos[1] + dy
                    if 0 <= adj_x < self.grid_size[0] and 0 <= adj_y < self.grid_size[1]:
                        if self.warehouse_map[adj_x, adj_y] == 0:
                            possible_shelf_positions.append((adj_x, adj_y))
                if possible_shelf_positions:
                    shelf_pos_to_place = self.random.choice(possible_shelf_positions)
                    self.warehouse_map[shelf_pos_to_place] = 3

        num_waiting_zones = 2
        waiting_zone_size = (3, 3)
        for _ in range(num_waiting_zones):
            while True:
                wait_x = self.random.randint(0, self.grid_size[0] - waiting_zone_size[0])
                wait_y = self.random.randint(0, self.grid_size[1] - waiting_zone_size[1])
                if np.sum(self.warehouse_map[wait_x:wait_x + waiting_zone_size[0], wait_y:wait_y + waiting_zone_size[1]] != 0) == 0:
                    self.warehouse_map[wait_x:wait_x + waiting_zone_size[0], wait_y:wait_y + waiting_zone_size[1]] = 4
                    break
        
        num_recharge_zones = 2
        recharge_zone_size = (3, 3)
        for _ in range(num_recharge_zones):
            while True:
                recharge_x = self.random.randint(0, self.grid_size[0] - recharge_zone_size[0])
                recharge_y = self.random.randint(0, self.grid_size[1] - recharge_zone_size[1])
                if np.sum(self.warehouse_map[recharge_x:recharge_x + recharge_zone_size[0], recharge_y:recharge_y + recharge_zone_size[1]] != 0) == 0:
                    self.warehouse_map[recharge_x:recharge_x + recharge_zone_size[0], recharge_y:recharge_y + recharge_zone_size[1]] = 5
                    break
    
    def end(self):
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)
        print("Tabla Q guardada en 'q_table.pkl'")
        
        plt.figure(figsize=(12, 8))
        for i, agent in enumerate(self.agents):
            plt.plot(agent.epoch_durations, label=f'Agente {i+1}')
        plt.title('Duracion de las Epocas de los agentes a lo largo del tiempo')
        plt.xlabel('Numero de Epoca')
        plt.ylabel('Duracion de la Epoca (pasos)')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 8))
        for i, agent in enumerate(self.agents):
            plt.plot(agent.battery_history, label=f'Agente {i+1}')
        plt.title('Nivel de Bateria de los agentes a lo largo de la simulacion')
        plt.xlabel('Pasos de la simulacion')
        plt.ylabel('Nivel de Bateria (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        agent_names = [f'Agente {i+1}' for i in range(len(self.agents))]
        error_counts = [agent.error_incidents for agent in self.agents]
        plt.bar(agent_names, error_counts, color='skyblue')
        plt.title('Numero de Fallos por Agente')
        plt.xlabel('Agente')
        plt.ylabel('Numero de Fallos')
        plt.grid(axis='y', linestyle='--')
        plt.show()

        self.visualize_last_epoch()

    def visualize_last_epoch(self):
        fig, ax = plt.subplots(figsize=(10, 10))

        cmap = self.custom_cmap()
        im = ax.imshow(self.warehouse_map, cmap=cmap, interpolation='nearest', origin='lower')
        
        points = [ax.plot([], [], 'o', markersize=10, label=f'Agente {i+1}')[0] for i in range(len(self.agents))]
        
        ax.set_title('Ultima Epoca de los Agentes (completa)')
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
            print("No hay datos de la ultima epoca completa para visualizar.")
        
    def custom_cmap(self):
        colors = ['red', 'white', 'blue', 'green', 'yellow', 'orange', 'purple']
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap

# Ejecutar simulacion
if __name__ == '__main__':
    model = WarehouseModel()
    results = model.run(steps=50000)