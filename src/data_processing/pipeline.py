from .data_processor import DataProcessor
import pandas as pd

class Pipeline(DataProcessor):
    def __init__(self, *args):
        assert all([isinstance(x, DataProcessor) for x in args]), "Pipeline only accespt `DataProcessor`"
        self.stages = list(args)

    def process_data(self, data):
        for stage in self.stages:
            data = stage(data)
        return data