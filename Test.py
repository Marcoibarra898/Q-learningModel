import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop


# ----------- A* PATHFINDING -----------
def astar(grid, start, goal):
    """Encuentra un camino usando A* evitando celdas -1 (prohibidas)."""
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        f, g, current, path = heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:  # 4 direcciones
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] != -1:
                new_cost = g + 1
                heappush(open_set, (new_cost + heuristic((nx, ny), goal), new_cost, (nx, ny), path + [(nx, ny)]))
    return None

def heuristic(a, b):
    # Distancia Manhattan (para grids)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# ----------- AGENTE -----------
class Robot(ap.Agent):

    def setup(self):
        # Inicializa objetivos
        self.has_cargo = False
        self.path = []

    def step(self):
        # Si no tiene ruta, calcularla
        if not self.path:
            if not self.has_cargo:
                # Ir a zona de carga
                targets = list(zip(*np.where(self.model.warehouse_map == 1)))
            else:
                # Ir a zona de descarga
                targets = list(zip(*np.where(self.model.warehouse_map == 2)))

            # Escoger un destino aleatorio de la zona
            if targets:
                goal = self.model.random.choice(targets)
                start = tuple(self.model.grid.positions[self])
                self.path = astar(self.model.warehouse_map, start, goal)
                
        else:
            # Mover un paso en el camino
            if len(self.path) > 1:
                next_pos = self.path[1]
                self.model.grid.move_to(self, tuple(next_pos))
                self.path.pop(0)
            else:
                # Llegó al destino
                if not self.has_cargo:
                    self.has_cargo = True  # recoge carga
                else:
                    self.has_cargo = False  # entrega carga
                self.path = []  # recalcular siguiente objetivo

# ----------- MODELO -----------
class WarehouseModel(ap.Model):
    
    def setup(self):
        self.grid_size = (30, 30)
        self.grid = ap.Grid(self, self.grid_size, track_empty=True)
        
        # (0 = libre, 1 = carga, 2 = descarga, -1 = prohibido
        self.warehouse_map = np.zeros(self.grid_size, dtype=int)
        
        self.warehouse_map[0:5, 0:5] = 1   # Zona de carga
        self.warehouse_map[25:30, 25:30] = 2 # Zona de descarga
        self.warehouse_map[9:25, 5:8] = -1  # Zona prohibida
        self.warehouse_map[9:25, 14:17] = -1
        self.warehouse_map[3:17, 22:25] = -1
        
        n_agents = 6
        self.agents = ap.AgentList(self, n_agents, Robot)

        # Crear posiciones iniciales para todos los agentes
        free_cells = list(zip(*np.where(self.warehouse_map == 0)))
        positions = [tuple(self.random.choice(free_cells)) for _ in self.agents]

        # Añadir todos los agentes al grid en esas posiciones
        self.grid.add_agents(self.agents, positions)
            
    def step(self):
        self.agents.step()
        
    def update(self):
        plt.clf()
        grid_img = np.copy(self.warehouse_map)

        # Pintar agentes con valor 3
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
        colors = ['red','white', 'blue', 'green', 'purple']  # libre, carga, descarga, prohibido
        bounds = [-1.5,-0.5, 0.5, 1.5, 2.5, 3.5]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap
        
        
# Ejecutar simulación con animación
model = WarehouseModel()
results = model.run(steps=50)