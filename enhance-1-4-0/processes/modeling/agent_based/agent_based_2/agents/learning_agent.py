from .agent import Agent

class LearningAgent(Agent):
    """
    Learning Agent class for agent-based modeling with learning capabilities.
    """
    def __init__(self, agent_id, initial_state, learning_rate):
        super().__init__(agent_id, initial_state)
        self.learning_rate = learning_rate

    def learn(self, reward):
        """
        Update the agent's state based on the received reward.
        """
        self.state += self.learning_rate * reward
