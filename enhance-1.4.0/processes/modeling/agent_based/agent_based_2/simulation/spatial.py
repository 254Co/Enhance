class SpatialSimulation:
    """
    Spatial-based simulation for agent-based modeling.
    """
    def __init__(self, grid):
        self.grid = grid

    def simulate(self, steps):
        """
        Run the spatial-based simulation.

        Parameters:
        steps (int): Number of simulation steps.
        """
        for _ in range(steps):
            for cell in self.grid.cells:
                self.grid.cells[cell].agent.step()
            for cell in self.grid.cells:
                neighbors = self.grid.get_neighbors(cell)
                for neighbor in neighbors:
                    self.grid.cells[cell].agent.interact(self.grid.cells[neighbor].agent)
