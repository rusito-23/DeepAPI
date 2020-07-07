"""
Base Exception.
Declare custom exceptions for better error handling.
"""


class Error(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code

    def __str__(self):
        return f'[ ERROR ] CODE: {self.code}  MESSAGE: {self.message}'

    def log(self):
        return self.__str__()

    def as_res(self):
        return (self.message, self.code)


class ModelInitializationError(Error):
    message = 'An error ocurred while initializing the model.'
    code = 500


class DreamError(Error):
    message = 'An error ocurred while running the Deep Dream algorithm.'
    code = 500


class PreProcessingError(Error):
    message = 'An error ocurred while processing the input.'
    code = 500


class PostProcessingError(Error):
    message = 'An error ocurred while processing the resulting image.'
    code = 500


class ConfigError(Error):
    message = 'An error ocurred while reading the config.'
    code = 400


class UnknownStyle(Error):
    code = 400

    def __init__(self, style):
        self.message = f'Unknown Style Name: {style}'
