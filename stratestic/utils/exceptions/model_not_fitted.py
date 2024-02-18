class ModelNotFitted(Exception):
    def __init__(self, *args):

        self.message = "The model has not been fitted yet."

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return self.__class__.__name__
