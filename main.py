from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import json
import re
import traceback
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(title="WLSuite AI Engine - Progressive Disclosure")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class ProjectData(BaseModel):
    projectId: int
    userPlan: str
    projectName: str
    projectIdea: str
    callToAction: str
    
    businessSector: Optional[str] = None
    communicationTone: Optional[str] = None
    colorPalette: Optional[str] = None
    
    visualStyle: Optional[str] = None
    animationLevel: Optional[str] = None
    customPrompt: Optional[str] = None

# DICCIONARIO DE MODELOS COMERCIALES CORREGIDO
MODELOS_IA = {
    "BASIC": "google/gemini-flash-1.5",
    "INTERMEDIATE": "openai/gpt-4o-mini", 
    "PREMIUM": "anthropic/claude-3.5-sonnet"
}

@app.get("/")
def home():
    return {"message": "Motor de IA Multimodelo Activo con Progressive Disclosure"}

@app.post("/api/v1/ai/generate")
async def generate_landing(data: ProjectData):
    try:
        plan_usuario = data.userPlan.upper()
        modelo_elegido = MODELOS_IA.get(plan_usuario, MODELOS_IA["BASIC"])
        
        print(f"[{plan_usuario}] Generando arquitectura web para: {data.projectName}")
        print(f"Usando el motor: {modelo_elegido}")
        
        prompt_ingenieria = f"""
        Eres un Copywriter experto en CRO (Conversion Rate Optimization) para e-commerce.
        Tu tarea es estructurar los textos persuasivos de una "Landing Page de un Solo Producto".
        
        Contexto del núcleo del negocio:
        - Empresa/Producto: {data.projectName}
        - Idea o propuesta de valor: {data.projectIdea}
        - Llamado a la acción (CTA) deseado: {data.callToAction}
        - Mercado Objetivo: Chile (Enfocado en validación local)
        
        Instrucciones: Crea textos cortos, directos y minimalistas. No uses lenguaje corporativo aburrido.
        """

        if plan_usuario in ["INTERMEDIATE", "PREMIUM"]:
            prompt_ingenieria += f"""
            Contexto estratégico y de marca:
            - Sector de la industria: {data.businessSector or 'No especificado'}
            - Tono de la marca: {data.communicationTone or 'No especificado'}
            - Paleta de colores / Identidad: {data.colorPalette or 'No especificado'}
            Ajusta tu redacción para que haga 'match' perfecto con este tono de comunicación.
            """

        if plan_usuario == "PREMIUM":
            prompt_ingenieria += f"""
            Directrices Avanzadas de UX:
            - Estilo visual objetivo: {data.visualStyle or 'Moderno'}
            - Nivel de animaciones esperado: {data.animationLevel or 'Suaves'}
            Ten en cuenta esta estética al momento de redactar el copy.
            """
            if data.customPrompt:
                prompt_ingenieria += f"""
                ATENCIÓN - REQUERIMIENTOS ESPECIALES DEL CLIENTE (PRIORIDAD MÁXIMA):
                El cliente ha solicitado específicamente lo siguiente: "{data.customPrompt}"
                Debes integrar obligatoriamente esta solicitud dentro de la estructura de tu respuesta.
                """

        if plan_usuario == "BASIC":
            estructura_json = """
            {
                "hero": {"headline": "Título principal", "subheadline": "Breve descripción", "ctaButton": "Botón"},
                "features": [
                    {"title": "Característica 1", "description": "Breve"},
                    {"title": "Característica 2", "description": "Breve"}
                ],
                "footer": {"contact": "contacto@empresa.cl"}
            }
            """
        elif plan_usuario == "INTERMEDIATE":
            estructura_json = """
            {
                "hero": {"headline": "...", "subheadline": "...", "ctaButton": "..."},
                "features": [{"title": "...", "description": "..."}, {"title": "...", "description": "..."}],
                "socialProof": {"urgencyText": "Texto oferta", "shippingText": "Texto envío"},
                "faq": [{"question": "Pregunta 1", "answer": "Respuesta 1"}, {"question": "Pregunta 2", "answer": "Respuesta 2"}],
                "footer": {"contact": "contacto@empresa.cl"}
            }
            """
        else: 
            estructura_json = """
            {
                "hero": {"headline": "...", "subheadline": "...", "ctaButton": "..."},
                "features": [{"title": "...", "description": "..."}, {"title": "...", "description": "..."}],
                "socialProof": {"urgencyText": "...", "shippingText": "...", "testimonials": [{"name": "...", "quote": "..."}]},
                "pricing": [{"planName": "...", "price": "...", "benefits": ["...", "..."]}],
                "objections": [{"objection": "Objeción frecuente", "rebuttal": "Respuesta"}],
                "customSection": {"title": "Título", "content": "Contenido"},
                "footer": {"contact": "contacto@empresa.cl"}
            }
            """

        prompt_ingenieria += f"""
        IMPORTANTE: Devuelve ÚNICAMENTE un objeto JSON válido. Sigue EXACTAMENTE esta estructura:
        {estructura_json}
        """

        response = client.chat.completions.create(
            model=modelo_elegido, 
            messages=[
                {"role": "user", "content": prompt_ingenieria}
            ]
        )

        respuesta_ia = response.choices[0].message.content

        # Extracción segura a prueba de errores
        match = re.search(r'\{.*\}', respuesta_ia, re.DOTALL)
        if match:
            respuesta_limpia = match.group(0)
        else:
            respuesta_limpia = respuesta_ia

        try:
            ai_metadata_json = json.loads(respuesta_limpia)
        except json.JSONDecodeError:
            ai_metadata_json = {
                "error": "El modelo no generó un JSON válido",
                "raw_response": respuesta_ia
            }

        return {
            "status": "success",
            "projectId": data.projectId,
            "plan_usado": plan_usuario,
            "ai_engine": modelo_elegido,
            "aiMetadata": ai_metadata_json 
        }
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error Crítico en la IA:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Error interno IA: {str(e)}")