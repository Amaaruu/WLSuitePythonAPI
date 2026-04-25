from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(title="LAndIng AI Service")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class ProjectData(BaseModel):
    projectId: int
    projectName: str
    businessSector: str
    communicationTone: str
    colorPalette: str
    userPlan: str  # Puede ser "BASIC", "INTERMEDIATE" o "PREMIUM"

#EL DICCIONARIO DE PLANES
MODELOS_IA = {
    # Plan Básico: Gemini 
    "BASIC": "google/gemini-2.0-flash-exp:free", 
    
    # Plan Intermedio: ChatGPT
    "INTERMEDIATE": "openai/gpt-4o-mini", 
    
    # Plan Premium: Claude 3.5 Sonnet
    "PREMIUM": "anthropic/claude-3.5-sonnet"
}

@app.get("/")
def home():
    return {"message": "Motor de IA Multimodelo Activo"}

@app.post("/api/v1/ai/generate")
async def generate_landing(data: ProjectData):
    try:
        # Buscamos qué modelo le toca. 
        # Si envían un plan raro por error, le damos el BASIC por defecto.
        plan_usuario = data.userPlan.upper()
        modelo_elegido = MODELOS_IA.get(plan_usuario, MODELOS_IA["BASIC"])
        
        print(f"[{plan_usuario}] Generando web para: {data.projectName}")
        print(f"Usando el modelo: {modelo_elegido}")
        
        prompt_ingenieria = f"""
        Eres un desarrollador web experto en UI/UX.
        Tu tarea es crear una Landing Page en un solo archivo usando HTML5 y Tailwind CSS (vía CDN).
        
        Requerimientos del cliente:
        - Nombre de la empresa: {data.projectName}
        - Rubro o sector: {data.businessSector}
        - Tono de comunicación: {data.communicationTone}
        - Colores corporativos principales: {data.colorPalette}
        
        La página debe tener: Un Header, una sección Hero persuasiva, características y un Footer.
        IMPORTANTE: Devuelve ÚNICAMENTE el código HTML.
        """

        #Inyectamos la variable 'modelo_elegido' en la llamada a OpenRouter
        response = client.chat.completions.create(
            model=modelo_elegido, 
            messages=[
                {"role": "user", "content": prompt_ingenieria}
            ]
        )

        codigo_generado = response.choices[0].message.content

        return {
            "status": "success",
            "projectId": data.projectId,
            "plan_usado": plan_usuario,
            "ai_engine": modelo_elegido,
            "generatedHtml": codigo_generado
        }
    except Exception as e:
        print(f"Error en la IA: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al comunicarse con la IA")