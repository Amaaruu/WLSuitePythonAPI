from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(title="WLSuite AI Engine")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class ProjectData(BaseModel):
    projectId: int
    projectName: str
    businessSector: str
    communicationTone: str
    colorPalette: str  # Ya no dicta el diseño, pero le da contexto a la IA sobre la vibra de la marca
    userPlan: str

# DICCIONARIO DE PLANES
MODELOS_IA = {
    "BASIC": "google/gemini-2.0-flash-exp:free", 
    "INTERMEDIATE": "openai/gpt-4o-mini", 
    "PREMIUM": "anthropic/claude-3.5-sonnet"
}

@app.get("/")
def home():
    return {"message": "Motor de IA Multimodelo Activo"}

@app.post("/api/v1/ai/generate")
async def generate_landing(data: ProjectData):
    try:
        plan_usuario = data.userPlan.upper()
        modelo_elegido = MODELOS_IA.get(plan_usuario, MODELOS_IA["BASIC"])
        
        print(f"[{plan_usuario}] Generando estructura web para: {data.projectName}")
        print(f"Usando el modelo: {modelo_elegido}")
        
        # El nuevo prompt: Estructurado, enfocado en conversiones locales y JSON
        prompt_ingenieria = f"""
        Eres un Copywriter experto en CRO (Conversion Rate Optimization) para e-commerce.
        Tu tarea es estructurar los textos persuasivos de una "Landing Page de un Solo Producto" (Single-Product Funnel).
        
        Contexto del proyecto:
        - Producto/Empresa: {data.projectName}
        - Sector: {data.businessSector}
        - Tono de comunicación: {data.communicationTone}
        - Mercado Objetivo: Chile (Enfocado en validación local y arbitraje geográfico)
        
        Instrucciones: Crea textos cortos, directos y minimalistas. No uses lenguaje corporativo aburrido.
        
        IMPORTANTE: Devuelve ÚNICAMENTE un objeto JSON válido, sin texto adicional, sin comillas invertidas ni bloques de markdown (```json).
        Debes seguir EXACTAMENTE esta estructura:
        {{
            "hero": {{
                "headline": "Título principal persuasivo (máx 8 palabras)",
                "subheadline": "Subtítulo que ataque un punto de dolor o resalte el valor principal",
                "ctaButton": "Texto del botón de acción principal (ej. Comprar ahora)"
            }},
            "features": [
                {{"title": "Característica 1", "description": "Descripción breve y concisa"}},
                {{"title": "Característica 2", "description": "Descripción breve y concisa"}},
                {{"title": "Característica 3", "description": "Descripción breve y concisa"}}
            ],
            "socialProof": {{
                "urgencyText": "Texto de escasez (ej. Oferta válida solo por hoy en todo Chile)",
                "shippingText": "Texto de logística (ej. Envíos rápidos por BlueExpress)"
            }},
            "footer": {{
                "contact": "Soporte: contacto@{data.projectName.lower().replace(' ', '')}.cl"
            }}
        }}
        """

        response = client.chat.completions.create(
            model=modelo_elegido, 
            messages=[
                {"role": "user", "content": prompt_ingenieria}
            ]
        )

        respuesta_ia = response.choices[0].message.content

        # Limpieza: Por si la IA decide envolver la respuesta en bloques de código markdown
        respuesta_limpia = re.sub(r'^```json\s*', '', respuesta_ia)
        respuesta_limpia = re.sub(r'^```\s*', '', respuesta_limpia)
        respuesta_limpia = re.sub(r'\s*```$', '', respuesta_limpia)
        respuesta_limpia = respuesta_limpia.strip()

        # Parseamos el string a un diccionario nativo de Python
        try:
            ai_metadata_json = json.loads(respuesta_limpia)
        except json.JSONDecodeError:
            # Fallback seguro por si la IA alucina y daña el formato
            ai_metadata_json = {
                "error": "El modelo no generó un JSON válido",
                "raw_response": respuesta_ia
            }

        return {
            "status": "success",
            "projectId": data.projectId,
            "plan_usado": plan_usuario,
            "ai_engine": modelo_elegido,
            "aiMetadata": ai_metadata_json # <-- Devolvemos el JSON estructurado, listo para Java
        }
        
    except Exception as e:
        print(f"Error en la IA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al comunicarse con la IA: {str(e)}")