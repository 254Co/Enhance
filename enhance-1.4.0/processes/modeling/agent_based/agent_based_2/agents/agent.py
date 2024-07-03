class Agent:
    """
    Basic Agent class for agent-based modeling.
    """
    def __init__(self, agent_id, initial_state):
        self.agent_id = agent_id
        self.state = initial_state

    def step(self):
        """
        Define the agent's behavior at each step.
        """
        pass

    def update_state(self, new_state):
        """
        Update the agent's state.
        """
        self.state = new_state

    def get_state(self):
        """
        Get the agent's current state.
        """
        return self.state
