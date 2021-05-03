import pandas as pd
import geopandas as gpd
import pathlib

def assign_geocoord_to_id2(df: pd.DataFrame) -> pd.DataFrame:   
    """Assigns latitude and longitude (in WGS84) values to a DataFrame with an ID2 column

    :param df: A DataFrame with a column named "ID2"
    :rtype: pd.DataFrame with added columns "Latitude" and "Longitude"
    """
    
    cwgp = gpd.read_file(pathlib.Path(__file__).parent / "data" / "cookwest_georeferencepoint_20190924.geojson")
    cegp = gpd.read_file(pathlib.Path(__file__).parent / "data" / "cookeast_georeferencepoint_20190924.geojson")
    
    
    gp = pd.concat([cwgp, cegp])
    gp = gp.assign(Latitude = gp.geometry.y,Longitude = gp.geometry.x)
    gp = gp.drop(["geometry"], axis = 1)

    result = df.copy()
    result = result.merge(gp, on = "ID2")

    return result
