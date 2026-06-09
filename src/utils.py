from dataclasses import dataclass

@dataclass
class IterationMetadata:
    bearing_position: str
    sampling_rate: int
    fault_diam_mm: float
    motor_load_hp: int
    motor_speed_rpm: int
    telemetry_position: str
    mat_url: str|None
    mat_filename: str|None
