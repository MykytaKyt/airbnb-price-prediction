from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib  # Use this for loading your trained model

app = FastAPI()


# Define a Pydantic model for the request data
class HouseData(BaseModel):
    bedrooms: int
    bathrooms: float
    # Add more fields as needed


# Load your trained machine learning model (replace 'your_model.pkl' with your actual model file)
model = joblib.load('your_model.pkl')


@app.post("/predict")
async def predict_house_price(data: HouseData):
    try:
        # Convert input data to a dictionary
        input_data = data.dict()

        # Preprocess input data if needed
        # For example, you may need to perform one-hot encoding

        # Make predictions using the loaded model
        predicted_price = model.predict([list(input_data.values())])

        # Return the prediction as JSON
        return {"predicted_price": predicted_price[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
