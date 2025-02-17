from .data_processor import DataProcessor
import pandas as pd

class GeometryFeatures(DataProcessor):
    def process_data(self, df: pd.DataFrame):
        geometry = df.geometry

        return df.assign(
            geometry_area=geometry.area,
            geometry_perimiter=geometry.length,
            geometry_compactness=geometry.length**2/(geometry.area + 0.00001),
            geometry_centroid_x=geometry.centroid.x,
            geometry_centroid_y=geometry.centroid.y,
        ).drop(columns='geometry')
