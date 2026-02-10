class OptimizationInterrupted(Exception):
    def __init__(self, message, solution):
        super().__init__(message)
        self.solution = solution
