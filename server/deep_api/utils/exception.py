"""
Base Exception.
Declare custom exceptions for better error handling.
"""


class Error(Exception):
    def __init__(self, message, exc):
        self.message = message
        self.exc = exc

    def __str__(self):
        return f'[ ERROR ] : {self.message} : {self.exc}'

    def log(self):
        return self.__str__()


class ModelInitializationError(Error):
    def __init__(self, exc):
        super(ModelInitializationError, self).__init__(
            'An error ocurred while initializing the model.', exc)


class DreamError(Error):
    def __init__(self):
        super(ProcessingError, self).__init__(
            'An error ocurred while running the Deep Dream algorithm.', exc)


class PreProcessingError(Error):
    def __init__(self, exc):
        super(PreProcessingError, self).__init__(
            'An error ocurred while processing the input.', exc)


class PostProcessingError(Error):
    def __init__(self, exc):
        super(PostProcessingError, self).__init__(
            'An error ocurred while processing the resulting image.', exc)


class ConfigError(Error):
    def __init__(self, exc):
        super(ConfigError, self).__init__(
            'An error ocurred while reading the config.', exc)
