import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle

# ----------- PARÁMETROS DE Q-LEARNING -----------
ALPHA = 0.2
GAMMA = 0.95
EPSILON_START = 0.5
EPSILON_DECAY_EPOCH = 0.995  # Decaimiento por completar una época
EPSILON_DECAY_STEP = 0.99995  # Decaimiento lento en cada paso
EPSILON_MIN = 0.01

# ----------- PARÁMETROS DE LA SIMULACIÓN -----------
ERROR_PROBABILITY = 0.0005
ERROR_COOLDOWN_STEPS = 50
LOW_BATTERY_THRESHOLD = 30  # Umbral de batería para ir a recargar
BATTERY_DEPLETION_PER_STEP = 0.2  # Consumo de batería por movimiento
BATTERY_IDLE_DEPLETION = 0.05  # Consumo de batería por estar inmóvil
BATTERY_RECHARGE_RATE = 2.0  # Velocidad de recarga por paso

# ----------- RECOMPENSAS NORMALIZADAS (componentes) -----------
REWARDS = {
    "hit_wall": -10,
    "idle": -2,
    "collision": -8,
    "no_battery": -10,
    "step_penalty": -1,
    "proximity_scale": 1.0,   # multiplicador para (old_dist - new_dist)
    "proximity_cap": 5,       # máximo aportado por proximidad
    "wait_zone": 8,
    "recharge_full": 10,
    "recharge_partial": 5,
    "deliver": 10,
    "pickup": 8
}

# Helper: clamp final reward to [-10, 10]
def clamp_reward(x):
    return float(np.clip(x, -10, 10))


# ----------- AGENTE -----------
class Robot(ap.Agent):

    def setup(self):
        self.has_cargo = False
        self.current_task = 'get_main_cargo'
        self.status = 'normal'
        self.error_cooldown_timer = 0
        self.battery_level = 100.0
        self.steps_in_epoch = 0
        self.epoch_durations = []
        self.current_epoch_path = []
        self.last_epoch_path = []
        self.battery_history = [100.0]
        self.error_incidents = 0
        self.path_history = []
        self.action = (0, 0)
        # epsilon is read from model.p.epsilon during step()

    def step(self):
        # Manejo del estado de falla
        if self.status == 'in_error':
            if self.error_cooldown_timer > 0:
                self.error_cooldown_timer -= 1
                self.action = (0, 0)
            else:
                self.status = 'normal'
                # reinstaurar tarea aleatoria al salir de error
                self.current_task = self.model.random.choice(['get_main_cargo', 'get_shelf_cargo'])
                self.action = (0, 0)
            return

        # Obtener epsilon del modelo (global)
        self.epsilon = self.model.p.epsilon

        # Si sin batería, sin movimiento
        if self.battery_level <= 0:
            self.battery_level = 0
            self.action = (0, 0)
            return

        # Manejo de recarga (si ya está en tarea 'recharge' la recarga ocurre en step del agente)
        if self.current_task == 'recharge' and self.battery_level < 100:
            # quedarse en la celda de recarga y recargar
            self.action = (0, 0)
            self.battery_level = min(100.0, self.battery_level + BATTERY_RECHARGE_RATE)
            self.battery_history.append(self.battery_level)
            return

        # Consumo por movimiento (se aplica antes de decidir para simular consumo durante el movimiento)
        self.battery_level -= BATTERY_DEPLETION_PER_STEP
        # evitar negativos
        if self.battery_level < 0:
            self.battery_level = 0.0
        self.battery_history.append(self.battery_level)

        # Construcción del estado para consulta Q
        battery_range = self.get_battery_range()
        current_pos = tuple(self.model.grid.positions[self])
        state = (current_pos, self.has_cargo, self.current_task, self.status, battery_range)

        # Epsilon-greedy usando el RNG del modelo (reproducible)
        if self.model.random.random() < self.epsilon:
            action = self.model.random.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)])  # incluir quedarse quieto
        else:
            possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
            q_values = {a: self.model.q_table.get((state, a), 0) for a in possible_actions}
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
        next_q_values = {a: self.model.q_table.get((new_state, a), 0) for a in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]}
        max_next_q = max(next_q_values.values()) if next_q_values else 0
        new_q = old_q + ALPHA * (reward + GAMMA * max_next_q - old_q)
        self.model.q_table[(old_state, action)] = new_q


# ----------- MODELO -----------
class WarehouseModel(ap.Model):

    def setup(self):
        # dimensiones
        self.grid_size = (30, 30)
        self.grid = ap.Grid(self, self.grid_size, track_empty=True)

        # agentes
        self.n_agents = 6
        self.agents = ap.AgentList(self, self.n_agents, Robot)

        self.dynamic_walls = True
        self.q_table = {}

        # crear mapa
        self.create_new_map()

        # celdas libres
        free_cells = list(zip(*np.where(self.warehouse_map == 0)))

        # asignar posiciones iniciales (evitar duplicados)
        initial_positions = []
        if free_cells:
            while len(initial_positions) < self.n_agents:
                pos = tuple(self.random.choice(free_cells))
                if pos not in initial_positions:
                    initial_positions.append(pos)

        self.grid.add_agents(self.agents, initial_positions)

        for agent in self.agents:
            agent.current_task = 'get_main_cargo'
            agent.has_cargo = False
            agent.status = 'normal'
            agent.battery_level = 100.0
            agent.steps_in_epoch = 0
            agent.epoch_durations = []
            agent.current_epoch_path = []
            agent.last_epoch_path = []
            agent.battery_history = [100.0]
            agent.error_incidents = 0

        # epsilon global
        self.p.epsilon = EPSILON_START

    def _get_target_pos(self, agent):
        # si está en error -> zona de espera (4)
        if agent.status == 'in_error':
            waiting_zone_coords = np.argwhere(self.warehouse_map == 4)
            if len(waiting_zone_coords) > 0:
                distances = np.sum(np.abs(waiting_zone_coords - np.array(self.grid.positions[agent])), axis=1)
                return tuple(waiting_zone_coords[np.argmin(distances)])
            else:
                return tuple(self.grid.positions[agent])

        # si está recargando -> zona de recarga (5)
        if agent.current_task == 'recharge':
            recharge_zone_coords = np.argwhere(self.warehouse_map == 5)
            if len(recharge_zone_coords) > 0:
                distances = np.sum(np.abs(recharge_zone_coords - np.array(self.grid.positions[agent])), axis=1)
                return tuple(recharge_zone_coords[np.argmin(distances)])
            else:
                return tuple(self.grid.positions[agent])

        # zonas regulares
        load_zone_coords = np.argwhere(self.warehouse_map == 1)
        unload_zone_coords = np.argwhere(self.warehouse_map == 2)
        shelf_coords = np.argwhere(self.warehouse_map == 3)

        load_zone_pos = tuple(load_zone_coords.mean(axis=0).astype(int)) if len(load_zone_coords) > 0 else tuple(self.grid.positions[agent])
        unload_zone_pos = tuple(unload_zone_coords.mean(axis=0).astype(int)) if len(unload_zone_coords) > 0 else tuple(self.grid.positions[agent])

        if agent.current_task == 'get_main_cargo':
            return load_zone_pos
        elif agent.current_task == 'deliver_cargo':
            return unload_zone_pos
        elif agent.current_task == 'get_shelf_cargo':
            if len(shelf_coords) > 0:
                distances = np.sum(np.abs(shelf_coords - np.array(self.grid.positions[agent])), axis=1)
                return tuple(shelf_coords[np.argmin(distances)])
            else:
                return tuple(self.grid.positions[agent])
        return tuple(self.grid.positions[agent])

    def step(self):
        # Decaimiento por paso (lento)
        self.p.epsilon = max(EPSILON_MIN, self.p.epsilon * EPSILON_DECAY_STEP)

        # posibilidad de fallo por agente
        for agent in self.agents:
            if agent.status == 'normal' and self.random.random() < ERROR_PROBABILITY:
                agent.status = 'in_error'
                agent.has_cargo = False
                agent.error_incidents += 1

            if agent.status == 'normal' and agent.current_task != 'recharge' and agent.battery_level < LOW_BATTERY_THRESHOLD:
                # asignar recarga
                # print(f"Agente {agent.id} tiene batería baja. Tarea de recarga asignada.")
                agent.current_task = 'recharge'

        # 1) cada agente decide su acción (en su step) y guardamos movimientos planeados
        planned_moves = {}
        for agent in self.agents:
            agent.step()

            old_pos = tuple(self.grid.positions[agent])
            old_state = (old_pos, agent.has_cargo, agent.current_task, agent.status, agent.get_battery_range())
            action = agent.action
            next_pos = (old_pos[0] + action[0], old_pos[1] + action[1])

            # inic. reward con paso estándar
            planned_moves[agent] = {
                'old_pos': old_pos,
                'next_pos': next_pos,
                'action': action,
                'reward': REWARDS['step_penalty'],
                'old_state': old_state
            }

        # 2) Validación básica: fuera de límites o muro -> penalizar y quedarse
        for agent, move in planned_moves.items():
            nx, ny = move['next_pos']
            if not (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]) or self.warehouse_map[move['next_pos']] == -1:
                move['reward'] = REWARDS['hit_wall']
                move['next_pos'] = move['old_pos']

        # 3) Detectar múltiples agentes intentando la misma celda
        pos_to_agents = {}
        for agent, move in planned_moves.items():
            pos_to_agents.setdefault(move['next_pos'], []).append(agent)

        conflict_agents = set()
        for pos, agents_in_pos in pos_to_agents.items():
            if len(agents_in_pos) > 1:
                # conflicto: más de 1 agente quiere la misma celda
                conflict_agents.update(agents_in_pos)

        # 4) Detectar ciclos / intercambios (A->B, B->C, C->A o A<->B)
        # Mapeo old_pos -> agent
        oldpos_to_agent = {m['old_pos']: agent for agent, m in planned_moves.items()}

        # construir grafo dirigido agente -> agente (si next_pos coincide con old_pos de otro agente)
        adj = {}
        for agent, move in planned_moves.items():
            nxt = move['next_pos']
            if nxt in oldpos_to_agent and oldpos_to_agent[nxt] != agent:
                adj.setdefault(agent, []).append(oldpos_to_agent[nxt])

        # DFS para detectar ciclos y agregar agentes de ciclo a conflict_agents
        visited = set()
        stack_nodes = set()
        cycle_nodes = set()

        def dfs(u, path):
            if u in path:
                idx = path.index(u)
                cycle_nodes.update(path[idx:])
                return
            if u in visited:
                return
            visited.add(u)
            path.append(u)
            for v in adj.get(u, []):
                dfs(v, path)
            path.pop()

        for agent in list(adj.keys()):
            if agent not in visited:
                dfs(agent, [])

        # añadir nodos de ciclo a conflictos
        conflict_agents.update(cycle_nodes)

        # 5) Aplicar penalizaciones por conflicto: agentes implicados no se mueven
        for agent in conflict_agents:
            move = planned_moves[agent]
            move['reward'] = REWARDS['collision']
            move['next_pos'] = move['old_pos']

        # 6) Ejecutar movimientos válidos y computar recompensas finales
        for agent, move in planned_moves.items():
            old_pos = move['old_pos']
            action = move['action']
            reward = float(move['reward'])  # empieza con step_penalty, o hit_wall, o collision, etc.
            next_pos = move['next_pos']
            old_state = move['old_state']

            # Sólo contar pasos en época si no está en error ni recargando (y si efectivamente puede moverse)
            if agent.status != 'in_error' and agent.current_task != 'recharge' and not (old_pos == next_pos and agent.battery_level <= 0):
                agent.steps_in_epoch += 1

            # Distancias (Manhattan)
            current_agent_pos_np = np.array(old_pos)
            target_pos = self._get_target_pos(agent)
            target_pos_np = np.array(target_pos)
            next_pos_np = np.array(next_pos)

            old_dist = np.abs(current_agent_pos_np - target_pos_np).sum()
            new_dist = np.abs(next_pos_np - target_pos_np).sum()

            # RECOMPENSA DE PROXIMIDAD (con tope)
            proximity_delta = (old_dist - new_dist) * REWARDS['proximity_scale']
            proximity_delta = float(np.clip(proximity_delta, -REWARDS['proximity_cap'], REWARDS['proximity_cap']))
            reward += proximity_delta

            # PEQUEÑA PENALIZACIÓN POR CADA PASO (ya incluida en step_penalty)
            # reward -= 1  # (omitido porque ya usamos REWARDS['step_penalty'])

            if old_pos == next_pos and agent.battery_level <= 0:
                # castigo fuerte por quedarse sin batería y no moverse
                reward = REWARDS['no_battery']

            # Mover agente en la rejilla
            self.grid.move_to(agent, next_pos)
            new_pos = tuple(self.grid.positions[agent])
            agent.current_epoch_path.append(new_pos)

            # Eventos especiales (prioritarios)
            #  - Si está en estado de error y alcanzó zona de espera (4)
            if agent.status == 'in_error':
                if self.warehouse_map[new_pos] == 4:
                    reward += REWARDS['wait_zone']
                    agent.error_cooldown_timer = ERROR_COOLDOWN_STEPS
                    agent.last_epoch_path = list(agent.current_epoch_path)
                    agent.current_epoch_path = []

            #  - Si está recargando
            elif agent.current_task == 'recharge':
                if self.warehouse_map[new_pos] == 5 and agent.battery_level >= 100.0:
                    reward += REWARDS['recharge_full']
                    # completar recarga -> cambiar tarea
                    agent.current_task = self.random.choice(['get_main_cargo', 'get_shelf_cargo'])
                elif self.warehouse_map[new_pos] == 5 and agent.battery_level < 100.0:
                    reward += REWARDS['recharge_partial']

            #  - Si está entregando carga
            elif agent.current_task == 'deliver_cargo':
                if self.warehouse_map[new_pos] == 2 and agent.has_cargo:
                    reward += REWARDS['deliver']
                    agent.epoch_durations.append(agent.steps_in_epoch)
                    agent.steps_in_epoch = 0
                    agent.last_epoch_path = list(agent.current_epoch_path)
                    agent.current_epoch_path = []
                    agent.has_cargo = False

                    # Decaimiento de epsilon por completar la época
                    self.p.epsilon = max(EPSILON_MIN, self.p.epsilon * EPSILON_DECAY_EPOCH)

                    if agent.battery_level < LOW_BATTERY_THRESHOLD:
                        agent.current_task = 'recharge'
                    else:
                        agent.current_task = self.random.choice(['get_main_cargo', 'get_shelf_cargo'])

            #  - Si está recogiendo carga del load zone
            elif agent.current_task == 'get_main_cargo':
                if self.warehouse_map[new_pos] == 1 and not agent.has_cargo:
                    reward += REWARDS['pickup']
                    agent.has_cargo = True
                    agent.current_task = 'deliver_cargo'

            #  - Si está recogiendo de estante
            elif agent.current_task == 'get_shelf_cargo':
                if self.warehouse_map[new_pos] == 3 and not agent.has_cargo:
                    reward += REWARDS['pickup']
                    agent.has_cargo = True
                    agent.current_task = 'deliver_cargo'

            # Recortar recompensa final al rango [-10, 10]
            reward = clamp_reward(reward)

            # Aprendizaje Q
            new_state = (new_pos, agent.has_cargo, agent.current_task, agent.status, agent.get_battery_range())
            agent.learn(old_state, action, reward, new_state)

        # final de step()

    def create_new_map(self):
        # inicializar mapa con ceros
        self.warehouse_map = np.zeros(self.grid_size, dtype=int)
        zone_size = (5, 5)

        # colocar load zone (1)
        load_x = self.random.randint(0, self.grid_size[0] - zone_size[0])
        load_y = self.random.randint(0, self.grid_size[1] - zone_size[1])
        self.warehouse_map[load_x:load_x + zone_size[0], load_y:load_y + zone_size[1]] = 1

        # colocar unload zone (2) en área libre
        while True:
            unload_x = self.random.randint(0, self.grid_size[0] - zone_size[0])
            unload_y = self.random.randint(0, self.grid_size[1] - zone_size[1])
            if np.sum(self.warehouse_map[unload_x:unload_x + zone_size[0], unload_y:unload_y + zone_size[1]] != 0) == 0:
                self.warehouse_map[unload_x:unload_x + zone_size[0], unload_y:unload_y + zone_size[1]] = 2
                break

        # muros dinámicos (-1)
        if self.dynamic_walls:
            num_walls = 3
            for _ in range(num_walls):
                start_row = self.random.randint(5, self.grid_size[0] - 5)
                start_col = self.random.randint(5, self.grid_size[1] - 5)
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

        # crear estantes adyacentes a algunos muros (3)
        wall_coords = np.argwhere(self.warehouse_map == -1)
        num_estantes = 4
        if len(wall_coords) > 0:
            for _ in range(num_estantes):
                chosen_wall_pos = tuple(self.random.choice(wall_coords))
                possible_shelf_positions = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    adj_x, adj_y = chosen_wall_pos[0] + dx, chosen_wall_pos[1] + dy
                    if 0 <= adj_x < self.grid_size[0] and 0 <= adj_y < self.grid_size[1]:
                        if self.warehouse_map[adj_x, adj_y] == 0:
                            possible_shelf_positions.append((adj_x, adj_y))
                if possible_shelf_positions:
                    shelf_pos_to_place = self.random.choice(possible_shelf_positions)
                    self.warehouse_map[shelf_pos_to_place] = 3

        # zonas de espera (4)
        num_waiting_zones = 2
        waiting_zone_size = (3, 3)
        for _ in range(num_waiting_zones):
            while True:
                wait_x = self.random.randint(0, self.grid_size[0] - waiting_zone_size[0])
                wait_y = self.random.randint(0, self.grid_size[1] - waiting_zone_size[1])
                if np.sum(self.warehouse_map[wait_x:wait_x + waiting_zone_size[0], wait_y:wait_y + waiting_zone_size[1]] != 0) == 0:
                    self.warehouse_map[wait_x:wait_x + waiting_zone_size[0], wait_y:wait_y + waiting_zone_size[1]] = 4
                    break

        # zonas de recarga (5)
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
        plt.bar(agent_names, error_counts)
        plt.title('Número de Fallos por Agente')
        plt.xlabel('Agente')
        plt.ylabel('Número de Fallos')
        plt.grid(axis='y', linestyle='--')
        plt.show()

        # Visualización de la última época
        self.visualize_last_epoch()

        # Guardar tabla Q
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)
        print("Tabla Q guardada en 'q_table.pkl'")

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
    # OJO: puedes ajustar el número de pasos; 50000 es alto para pruebas.
    results = model.run(steps=500000)
    