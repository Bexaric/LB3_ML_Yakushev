from math import pi
from pydantic_settings import BaseSettings, SettingsConfigDict

class RobotSettings(BaseSettings):
    RANDOM_STATE: int = 42
    
    DIST_CENTER: float = 0.125
    WHEEL_ANGLE: float = 30 / 180 * pi
    WHEEL_RADIUS: float = 0.04 

    PROJECT_DIR: str = 'C:/ML_Labs/LB3_ML_Yakushev'
    DATASET_PATH: str = f'{PROJECT_DIR}/data/Data_Set_(A+B).xlsx'

    # Configuration for loading data
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

robot = RobotSettings()