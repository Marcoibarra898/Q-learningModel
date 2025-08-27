import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop


# ----------- A* PATHFINDING -----------
def astar(grid, start, goal, temporary_obstacles=[]):
    """Encuentra un camino usando A* evitando celdas -1 (prohibidas) y obstáculos temporales."""
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
    visited = set()
    
    temp_obstacles_set = set(temporary_obstacles)

    while open_set:
        f, g, current, path = heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:  # 4 direcciones
            nx, ny = current[0] + dx, current[1] + dy
            # Se evita tanto las zonas prohibidas (-1) como los obstáculos temporales
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] != -1 and (nx, ny) not in temp_obstacles_set:
                new_cost = g + 1
                heappush(open_set, (new_cost + heuristic((nx, ny), goal), new_cost, (nx, ny), path + [(nx, ny)]))
    return None

def heuristic(a, b):
    # Distancia Manhattan (para grids)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# ----------- AGENTE -----------
class Robot(ap.Agent):
    
    def setup(self):
        self.has_cargo = False
        self.path = []
        
    def step(self):
        if not self.path:
            if not self.has_cargo:
                targets = list(zip(*np.where(self.model.warehouse_map == 1)))
            else:
                targets = list(zip(*np.where(self.model.warehouse_map == 2)))
            
            if targets:
                goal = self.model.random.choice(targets)
                start = tuple(self.model.grid.positions[self])
                self.path = astar(self.model.warehouse_map, start, goal)
            else:
                self.path = []


# ----------- MODELO -----------
class WarehouseModel(ap.Model):
    
    def setup(self):
        self.grid_size = (30, 30)
        self.grid = ap.Grid(self, self.grid_size, track_empty=True)
        
        # (0 = libre, 1 = carga, 2 = descarga, -1 = prohibido
        self.warehouse_map = np.zeros(self.grid_size, dtype=int)
        
        self.warehouse_map[0:5, 0:5] = 1   # Zona de carga
        self.warehouse_map[25:30, 25:30] = 2 # Zona de descarga
        
        # Zonas prohibidas ahora son -1 para que A* las evite
        self.warehouse_map[9:25, 5:8] = -1
        self.warehouse_map[9:25, 14:17] = -1
        self.warehouse_map[3:17, 22:25] = -1
        
        n_agents = 6
        self.agents = ap.AgentList(self, n_agents, Robot)

        free_cells = list(zip(*np.where(self.warehouse_map == 0)))
        positions = [tuple(self.random.choice(free_cells)) for _ in self.agents]
        self.grid.add_agents(self.agents, positions)
            
    def step(self):
        self.agents.step()
        
        sorted_agents = sorted(self.agents, key=self.get_priority, reverse=True)
        
        new_occupied_positions = set()
        
        for agent in sorted_agents:
            if agent.path and len(agent.path) > 1:
                current_pos = tuple(self.grid.positions[agent])
                next_pos = agent.path[1]
                
                # Comprobar si la siguiente posición está ocupada por otro agente
                next_pos_occupied = next_pos in self.grid.positions.keys()
                
                if next_pos_occupied:
                    # Encontrar el agente que ocupa la celda de destino
                    blocking_agent = None
                    for other_agent in self.agents:
                        if tuple(self.grid.positions[other_agent]) == next_pos:
                            blocking_agent = other_agent
                            break
                            
                    # Comparar prioridades
                    if self.get_priority(agent) > self.get_priority(blocking_agent):
                        # Este agente tiene mayor prioridad, el agente bloqueado debe moverse
                        # El agente con menor prioridad recalcula su ruta en el próximo ciclo
                        # y este agente con mayor prioridad esperará a que se mueva.
                        pass
                    else:
                        # Este agente tiene menor prioridad o igual, debe recalcular su camino.
                        # Consideramos la posición del agente que bloquea como un obstáculo temporal
                        
                        # Lista de posiciones ocupadas por agentes
                        occupied_positions = list(self.grid.positions.keys())
                        
                        # Recalcular ruta considerando la posición del otro agente como obstáculo
                        new_path = astar(self.warehouse_map, current_pos, agent.path[-1], temporary_obstacles=occupied_positions)
                        
                        if new_path:
                            agent.path = new_path
                        # Si no se encuentra una nueva ruta, simplemente espera en su lugar.
                        pass
                
                # Si la posición está libre (o lo estará en este mismo paso por un agente de mayor prioridad)
                # entonces puede moverse
                if next_pos not in new_occupied_positions:
                    self.grid.move_to(agent, next_pos)
                    agent.path.pop(0)
                    new_occupied_positions.add(next_pos)
            
            elif agent.path and len(agent.path) == 1:
                # Llegó al destino
                if not agent.has_cargo:
                    agent.has_cargo = True
                else:
                    agent.has_cargo = False
                agent.path = []
    
    def get_priority(self, agent):
        """Asigna una prioridad al agente para el movimiento de este turno."""
        # Prioridad 1: Agente con carga (va a la zona de descarga)
        if agent.has_cargo:
            priority = 1000  # Un valor alto para asegurar la prioridad
        else:
            priority = 0
            
        # Prioridad 2: El que tiene el camino más corto
        # Solo aplicable si ambos tienen la misma prioridad de zona
        if len(agent.path) > 1:
            priority -= len(agent.path)
        
        # Prioridad 3: Aleatoria en caso de empate
        priority += self.random.random()
        
        return priority
        
    def update(self):
        plt.clf()
        grid_img = np.copy(self.warehouse_map)

        for agent in self.agents:
            x, y = self.grid.positions[agent]
            grid_img[x, y] = 3

        plt.imshow(grid_img, cmap=self.custom_cmap(), origin="upper")
        plt.title(f"Almacén - Paso {self.t}")
        plt.pause(0.3)
        
    def end(self):
        plt.show()
        
    def custom_cmap(self):
        from matplotlib.colors import ListedColormap, BoundaryNorm
        colors = ['red', 'white', 'blue', 'green', 'purple']
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap
        
        
# Ejecutar simulación con animación
model = WarehouseModel()
results = model.run(steps=200)