import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import InferencePipeline


class SensorResponse(BaseModel):
    sensor_id: str|None = None
    timestamp: str|None = None
    feat_avg_power: float|None = None
    feat_narrowband_power_rate: float|None = None
    feat_kurtosis: float|None = None
    feat_dominant_freq: float|None = None
    feat_most_relevant_binned_freq: float|None = None
    feat_most_relevant_binned_power: float|None = None
    feat_binned_fft_power_bin_0: float|None = None
    feat_binned_fft_power_bin_1: float|None = None
    feat_binned_fft_power_bin_2: float|None = None
    feat_binned_fft_power_bin_3: float|None = None
    feat_binned_fft_power_bin_4: float|None = None
    feat_binned_fft_power_bin_5: float|None = None
    feat_binned_fft_power_bin_6: float|None = None
    feat_binned_fft_power_bin_7: float|None = None
    feat_binned_fft_power_bin_8: float|None = None
    feat_binned_fft_power_bin_9: float|None = None


class ModelResponse(BaseModel):
    model_type: str
    model_name: str
    prediction: list


DEFAULT_MODELS = {
    "fault_binary":"model/pipeline_fault_binary_c25cc66bf776c3afc4de19be9909ff23.joblib",
    "fault_multiclass":"model/pipeline_fault_multiclass_ca26556f5f6446f0d69c4439a2001b4d.joblib",
    "position_multiclass":"model/pipeline_position_multiclass_ca26556f5f6446f0d69c4439a2001b4d.joblib"
}


app = FastAPI()


@app.get("/")
def root() -> dict:
    return {"message": "Welcome to the Fault Detection API. Use the /predict/{model_type} endpoint to make predictions. Models available: fault_binary, fault_multiclass, position_multiclass."}


@app.post("/predict/{model_type}")
async def predict(sensor_response: SensorResponse, model_type: str) -> ModelResponse:
    sensor_response = {k: v for k, v in sensor_response.model_dump().items() if k not in ["sensor_id", "timestamp"]}
    model_path = DEFAULT_MODELS.get(model_type,"")

    if not model_path:
        raise HTTPException(status_code=422, detail=f"Model type '{model_type}' not found.")

    try:
        prediction = InferencePipeline.inference_from_file(model_path,sensor_response)
        return ModelResponse(
            model_type=model_type,
            model_name=os.path.basename(model_path),
            prediction=prediction.tolist()
        )
    except Exception as e:
        with open("error.log", "a") as f:
            f.write(str(e) + "\n")
        raise HTTPException(status_code=500, detail=str(e))