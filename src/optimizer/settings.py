import json

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict, BaseModel
from dotenv import load_dotenv
load_dotenv()


class EnvSettings(BaseSettings):
    # max_num_of_steps: int = Field(alias='MAX_NUM_OF_STEPS', default=50)
    epochs_num: int = Field(alias='EPOCHS_NUM', default=10)

    model_config = ConfigDict(frozen=True)


class GeneratorSettings(BaseModel):
    max_truck_num: int
    min_requests_num: int
    max_requests_num: int
    simulator_start_date: str
    simulator_end_date: str
    load_point_names: list[str]
    unload_point_names: list[str]
    capacities_variants: list[int]
    min_distance: int
    max_distance: int

    model_config = ConfigDict(frozen=True)


class ObservationFeatureConfig(BaseModel):
    use_time_windows: bool = True
    use_executed_requests: bool = True
    use_unfinished_ratio: bool = True
    use_current_selection: bool = True
    use_next_request_tw: bool = True
    use_pairwise_features: bool = True
    pairwise_lookahead_requests: int = 1

    model_config = ConfigDict(frozen=True)


ENV_SETTINGS = EnvSettings()

with open('input/generator_settings.json', 'r') as f:
    __raw_gen_data = json.load(f)

GENERATOR_SETTINGS = GeneratorSettings(**__raw_gen_data)
del __raw_gen_data

DEFAULT_OBSERVATION_FEATURES = ObservationFeatureConfig()
