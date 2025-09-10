import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle

# ----------- PARÁMETROS DE Q-LEARNING -----------
ALPHA = 0.2
GAMMA = 0.95
EPSILON_START = 0.5 
EPSILON_DECAY = 0.995  # Decaimiento del 0.5% por cada época completada
EPSILON_MIN = 0.01

# ----------- PARÁMETROS DE LA SIMULACIÓN -----------
ERROR_PROBABILITY = 0.0005 
ERROR_COOLDOWN_STEPS = 50 
LOW_BATTERY_THRESHOLD = 30 # Umbral de batería para ir a recargar
BATTERY_DEPLETION_PER_STEP = 0.2 # Consumo de batería por movimiento
BATTERY_IDLE_DEPLETION = 0.05 # Consumo de batería por estar inmóvil
BATTERY_RECHARGE_RATE = 2.0 # Velocidad de recarga por paso

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
        # Individual Q-table for each agent
        self.q_table = {}
        # Position history to prevent oscillation
        self.position_history = []
        self.max_history_length = 10
        # Task assignment tracking
        self.assigned_task_id = None
        
    def step(self):
        # Manejo del estado de falla y recarga
        if self.status == 'in_error':
            if self.error_cooldown_timer > 0:
                self.error_cooldown_timer -= 1
                self.action = (0, 0)
            else:
                self.status = 'normal'
                self.current_task = self.model.random.choice(['get_main_cargo', 'get_shelf_cargo'])
                self.action = (0, 0)
            return
        
        # Lógica de toma de decisiones y consumo de batería
        self.epsilon = self.model.p.epsilon
        
        if self.battery_level <= 0:
            self.battery_level = 0
            self.action = (0, 0)
            return
        
        # Manejo de recarga
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
            q_values = {a: self.q_table.get((state, a), 0) for a in [(0, 1), (0, -1), (1, 0), (-1, 0)]}
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = self.model.random.choice(best_actions)
        
        # Prevent oscillation by avoiding recently visited positions
        next_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
        if next_pos in self.position_history[-3:]:  # Avoid last 3 positions
            # Choose a different action that doesn't lead to recent positions
            available_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            safe_actions = []
            for alt_action in available_actions:
                alt_next_pos = (current_pos[0] + alt_action[0], current_pos[1] + alt_action[1])
                if alt_next_pos not in self.position_history[-3:]:
                    safe_actions.append(alt_action)
            
            if safe_actions:
                action = self.model.random.choice(safe_actions)
        
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
        old_q = self.q_table.get((old_state, action), 0)
        
        next_q_values = {a: self.q_table.get((new_state, a), 0) for a in [(0, 1), (0, -1), (1, 0), (-1, 0)]}
        max_next_q = max(next_q_values.values()) if next_q_values else 0
        
        new_q = old_q + ALPHA * (reward + GAMMA * max_next_q - old_q)
        self.q_table[(old_state, action)] = new_q

# ----------- MODELO -----------
class WarehouseModel(ap.Model):
    
    def setup(self):
        self.grid_size = (30, 30)
        self.grid = ap.Grid(self, self.grid_size, track_empty=True)
        
        self.n_agents = 6
        self.agents = ap.AgentList(self, self.n_agents, Robot)
        
        self.dynamic_walls = True
        # Task assignment tracking to prevent overlapping
        self.assigned_tasks = {
            'get_main_cargo': set(),
            'get_shelf_cargo': set(),
            'deliver_cargo': set(),
            'recharge': set()
        }
        self.task_locations = {}  # Track which agents are assigned to specific locations
        
        self.create_new_map()
        
        free_cells = list(zip(*np.where(self.warehouse_map == 0)))
        
        initial_positions = []
        if free_cells:
            for _ in range(self.n_agents):
                initial_positions.append(tuple(self.random.choice(free_cells)))
        
        self.grid.add_agents(self.agents, initial_positions)

        for agent in self.agents:
            agent.has_cargo = False
            agent.status = 'normal'
            agent.battery_level = 100
            # Assign initial task to prevent overlapping
            self.assign_task(agent)
        
        self.p.epsilon = EPSILON_START
    
    def assign_task(self, agent):
        """Assign a task to an agent, avoiding conflicts with other agents"""
        # Remove agent from current task assignment
        if agent.assigned_task_id:
            self.assigned_tasks[agent.current_task].discard(agent.assigned_task_id)
            if agent.assigned_task_id in self.task_locations:
                del self.task_locations[agent.assigned_task_id]
        
        # Determine available tasks based on agent state
        available_tasks = []
        
        if agent.battery_level < LOW_BATTERY_THRESHOLD:
            available_tasks = ['recharge']
        elif agent.has_cargo:
            available_tasks = ['deliver_cargo']
        else:
            # Check for available cargo tasks
            if len(self.assigned_tasks['get_main_cargo']) < 2:  # Max 2 agents on main cargo
                available_tasks.append('get_main_cargo')
            if len(self.assigned_tasks['get_shelf_cargo']) < 2:  # Max 2 agents on shelf cargo
                available_tasks.append('get_shelf_cargo')
        
        if not available_tasks:
            # If no specific tasks available, assign randomly
            available_tasks = ['get_main_cargo', 'get_shelf_cargo']
        
        # Choose task
        chosen_task = self.random.choice(available_tasks)
        
        # Generate unique task ID
        task_id = f"{agent.id}_{chosen_task}_{self.random.randint(1000, 9999)}"
        
        # Assign task
        agent.current_task = chosen_task
        agent.assigned_task_id = task_id
        self.assigned_tasks[chosen_task].add(task_id)
        
        return task_id
            
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
        
        for agent in self.agents:
            if agent.status == 'normal' and self.random.random() < ERROR_PROBABILITY:
                agent.status = 'in_error'
                agent.has_cargo = False
                agent.error_incidents += 1
                # Remove from task assignments when in error
                if agent.assigned_task_id:
                    self.assigned_tasks[agent.current_task].discard(agent.assigned_task_id)
                    if agent.assigned_task_id in self.task_locations:
                        del self.task_locations[agent.assigned_task_id]
                    agent.assigned_task_id = None
            
            if agent.status == 'normal' and agent.current_task != 'recharge' and agent.battery_level < LOW_BATTERY_THRESHOLD:
                print(f"Agente {agent.id} tiene batería baja. Tarea de recarga asignada.")
                self.assign_task(agent)

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
            elif next_pos == old_pos:
                move['reward'] = -10 
            
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
            
            reward += (old_dist - new_dist) * 5

            if old_pos == next_pos and agent.battery_level <= 0:
                reward = -500

            self.grid.move_to(agent, next_pos)
            new_pos = tuple(self.grid.positions[agent])
            agent.current_epoch_path.append(new_pos)
            
            # Update position history to prevent oscillation
            agent.position_history.append(new_pos)
            if len(agent.position_history) > agent.max_history_length:
                agent.position_history.pop(0)
            
            if agent.status == 'in_error':
                if self.warehouse_map[new_pos] == 4:
                    reward = 200
                    agent.error_cooldown_timer = ERROR_COOLDOWN_STEPS
                    agent.last_epoch_path = list(agent.current_epoch_path)
                    agent.current_epoch_path = []
            
            elif agent.current_task == 'recharge':
                if self.warehouse_map[new_pos] == 5 and agent.battery_level >= 100:
                    reward = 150
                    # Complete recharge task and assign new task
                    if agent.assigned_task_id:
                        self.assigned_tasks['recharge'].discard(agent.assigned_task_id)
                    self.assign_task(agent)
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
                    
                    self.p.epsilon = max(EPSILON_MIN, self.p.epsilon * EPSILON_DECAY)
                    
                    # Complete delivery task and assign new task
                    if agent.assigned_task_id:
                        self.assigned_tasks['deliver_cargo'].discard(agent.assigned_task_id)
                    self.assign_task(agent)

            elif agent.current_task == 'get_main_cargo':
                if self.warehouse_map[new_pos] == 1 and not agent.has_cargo:
                    reward = 100
                    agent.has_cargo = True
                    # Complete cargo pickup task and assign delivery task
                    if agent.assigned_task_id:
                        self.assigned_tasks['get_main_cargo'].discard(agent.assigned_task_id)
                    agent.current_task = 'deliver_cargo'
                    agent.assigned_task_id = f"{agent.id}_deliver_cargo_{self.random.randint(1000, 9999)}"
                    self.assigned_tasks['deliver_cargo'].add(agent.assigned_task_id)
            
            elif agent.current_task == 'get_shelf_cargo':
                if self.warehouse_map[new_pos] == 3 and not agent.has_cargo:
                    reward = 100
                    agent.has_cargo = True
                    # Complete cargo pickup task and assign delivery task
                    if agent.assigned_task_id:
                        self.assigned_tasks['get_shelf_cargo'].discard(agent.assigned_task_id)
                    agent.current_task = 'deliver_cargo'
                    agent.assigned_task_id = f"{agent.id}_deliver_cargo_{self.random.randint(1000, 9999)}"
                    self.assigned_tasks['deliver_cargo'].add(agent.assigned_task_id)

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
        # 1. Gráfica de Duración de las Épocas
        plt.figure(figsize=(12, 8))
        for i, agent in enumerate(self.agents):
            plt.plot(agent.epoch_durations, label=f'Agente {i+1}')
        plt.title('Duración de las Épocas de los agentes a lo largo del tiempo')
        plt.xlabel('Número de Época')
        plt.ylabel('Duración de la Época (pasos)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 2. Gráfica de Nivel de Batería de los Agentes
        plt.figure(figsize=(12, 8))
        for i, agent in enumerate(self.agents):
            plt.plot(agent.battery_history, label=f'Agente {i+1}')
        plt.title('Nivel de Batería de los agentes a lo largo de la simulación')
        plt.xlabel('Pasos de la simulación')
        plt.ylabel('Nivel de Batería (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 3. Gráfica de Incidencias de Fallos
        plt.figure(figsize=(10, 6))
        agent_names = [f'Agente {i+1}' for i in range(len(self.agents))]
        error_counts = [agent.error_incidents for agent in self.agents]
        plt.bar(agent_names, error_counts, color='skyblue')
        plt.title('Número de Fallos por Agente')
        plt.xlabel('Agente')
        plt.ylabel('Número de Fallos')
        plt.grid(axis='y', linestyle='--')
        plt.show()

        # Visualización de la última época
        self.visualize_last_epoch()
        
        # Save individual Q-tables for each agent
        agent_q_tables = {}
        for i, agent in enumerate(self.agents):
            agent_q_tables[f'agent_{i}'] = agent.q_table
            
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(agent_q_tables, f)
            
        print("Tablas Q individuales guardadas en 'q_table.pkl'")
        
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
        colors = ['red', 'white', 'blue', 'green', 'yellow', 'orange', 'purple']
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap

# Ejecutar simulación
if __name__ == '__main__':
    model = WarehouseModel()
    results = model.run(steps=500000) # Aumenta los pasos para un mejor entrenamiento