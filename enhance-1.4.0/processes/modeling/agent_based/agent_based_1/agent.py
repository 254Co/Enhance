class Agent:
    def __init__(self, id, strategy):
        self.id = id
        self.strategy = strategy
        self.history = []

    def decide(self, context):
        # Implement decision-making process based on context and history
        return self.strategy(context)
