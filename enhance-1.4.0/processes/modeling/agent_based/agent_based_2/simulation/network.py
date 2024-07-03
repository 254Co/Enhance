class NetworkSimulation:
    """
    Network-based simulation for agent-based modeling.
    """
    def __init__(self, network):
        self.network = network

    def simulate(self, steps):
        """
        Run the network-based simulation.

        Parameters:
        steps (int): Number of simulation steps.
        """
        for _ in range(steps):
            for node in self.network.nodes:
                self.network.nodes[node]['agent'].step()
            for edge in self.network.edges:
                self.network.nodes[edge[0]]['agent'].interact(self.network.nodes[edge[1]]['agent'])
