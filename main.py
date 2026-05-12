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


PLAN_ALIASES = {
    "BASICO": "BASIC", "BÁSICO": "BASIC",
    "INTERMEDIO": "INTERMEDIATE",
    "PREMIUM": "PREMIUM",
    "BASIC": "BASIC", "INTERMEDIATE": "INTERMEDIATE",
    "basico": "BASIC", "básico": "BASIC",
    "intermedio": "INTERMEDIATE", "premium": "PREMIUM",
}

MODELOS_IA = {
    "BASIC":        "google/gemini-2.5-flash-lite",
    "INTERMEDIATE": "openai/gpt-4o-mini",
    "PREMIUM":      "anthropic/claude-sonnet-4-5",
}

PLAN_FALLBACK = "BASIC"


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS DE SISTEMA — uno por plan, cada uno con un rol y objetivo claro
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASIC = """
Eres un experto en marketing digital y diseño web especializado en crear landing pages 
de alta conversión para pequeños negocios en Chile.

Tu trabajo es generar el contenido estratégico y copywriting para una landing page 
profesional de un solo producto o servicio.

PRINCIPIOS DE CALIDAD QUE DEBES APLICAR SIEMPRE:
- El headline debe ser el texto más poderoso de la página: claro, específico y orientado al beneficio. 
  Evita títulos genéricos. Máximo 10 palabras. Ejemplo de malo: "Bienvenidos a nuestra empresa". 
  Ejemplo de bueno: "Frutas y verduras frescas directo al campo, entregadas en tu puerta".
- El subheadline expande el headline con el beneficio principal y elimina la principal duda del cliente. 
  2-3 oraciones máximo.
- El CTA debe ser un verbo de acción específico, no genérico. 
  Malo: "Enviar". Bueno: "Pedir mi caja de verduras ahora".
- Las features deben describir BENEFICIOS reales para el cliente, no características técnicas. 
  Cada una: título de 3-5 palabras + descripción de 1-2 oraciones concretas.
- El footer debe tener un email de contacto realista y profesional basado en el nombre del negocio.
- El contenido debe sonar humano, cercano y chileno. Sin lenguaje corporativo ni frases hechas.
- Adapta el tono y vocabulario al tipo de negocio.

FORMATO DE RESPUESTA:
Devuelve ÚNICAMENTE un objeto JSON válido, sin texto adicional, sin markdown, sin explicaciones.
"""

SYSTEM_PROMPT_INTERMEDIATE = """
Eres un estratega de marketing digital y especialista en UX copywriting con experiencia 
en negocios chilenos. Tu expertise incluye CRO (Conversion Rate Optimization), 
psicología del consumidor y arquitectura de información.

Tu trabajo es generar el contenido completo y estratégico para una landing page 
profesional de alta conversión.

PRINCIPIOS ESTRATÉGICOS QUE DEBES APLICAR:

HERO SECTION:
- Headline: poderoso, específico, orientado al beneficio principal. Máximo 10 palabras.
  Usa la fórmula: [Resultado deseado] + [Diferenciador] + [Para quién].
- Subheadline: elimina la principal objeción y refuerza la propuesta de valor. 2-3 oraciones.
- CTA: verbo de acción + beneficio inmediato. Ejemplo: "Recibir mi primera caja hoy".

FEATURES (mínimo 3, idealmente 4):
- Cada feature = un beneficio real, no una característica.
- Estructura: ícono emoji relevante + título 3-5 palabras + descripción 2 oraciones concretas.
- Orden: del beneficio más importante al menos importante.

SOCIAL PROOF:
- urgencyText: genera urgencia real y creíble (escasez, tiempo limitado, demanda alta).
  Ejemplo: "Solo quedan 8 cajas disponibles esta semana".
- shippingText: beneficio logístico concreto. Ejemplo: "Despacho gratis en Santiago en 24h".

FAQ (mínimo 3 preguntas):
- Responde las dudas reales que impiden la compra.
- Las preguntas deben ser las que un cliente real haría antes de comprar.
- Las respuestas deben ser cortas, directas y tranquilizadoras.

TONO:
- Adapta completamente el lenguaje al sector y tono de comunicación indicado.
- Chileno, humano, concreto. Sin corporativismos.

FORMATO DE RESPUESTA:
Devuelve ÚNICAMENTE un objeto JSON válido, sin texto adicional, sin markdown, sin explicaciones.
"""

SYSTEM_PROMPT_PREMIUM = """
Eres un director creativo y estratega de conversión de clase mundial, especializado en 
landing pages premium para el mercado latinoamericano. Combinas expertise en:
- Copywriting persuasivo y psicología del consumidor
- Diseño UX/UI y arquitectura de experiencias digitales  
- Estrategia de marca y posicionamiento
- CRO avanzado y optimización de embudos de venta

Tu trabajo es generar el contenido más completo, sofisticado y persuasivo posible para 
una landing page premium que destaque frente a la competencia.

ESTÁNDARES DE CALIDAD PREMIUM:

HERO SECTION — Primera impresión que no se puede fallar:
- Headline: el mejor texto de toda la página. Debe generar deseo inmediato.
  Fórmula avanzada: [Emoción/Resultado] + [Diferenciador único] + [Credibilidad implícita].
  Máximo 10 palabras. Sin clichés. Sin palabras vacías.
- Subheadline: 2-3 oraciones que expanden el headline, eliminan la duda principal 
  y crean anticipación. Debe fluir naturalmente del headline.
- CTA principal: específico, con urgencia implícita, orientado al resultado.
- badge: una etiqueta corta de credibilidad (Ej: "⭐ Más de 500 clientes satisfechos", 
  "#1 en Santiago", "Desde 2018").

FEATURES (mínimo 4, idealmente 5-6):
- Cada feature comunica un beneficio emocional + funcional.
- Estructura: emoji + título poderoso + descripción 2-3 oraciones ricas en detalles.
- El conjunto debe contar una historia coherente del producto/servicio.

SOCIAL PROOF — El motor de la confianza:
- urgencyText: urgencia creíble y específica.
- shippingText: beneficio logístico premium.
- testimonials (mínimo 2): testimonios detallados y creíbles.
  Cada uno con: nombre, cargo/contexto, cita específica con detalle concreto, ciudad.
  Los testimonios deben mencionar resultados específicos, no ser genéricos.

PRICING (si aplica al negocio):
- Solo incluir si el negocio tiene variantes de precio lógicas.
- Si no aplica (ej: verdulería), usa esta sección para destacar un pack o promoción especial.
- Cada opción con nombre creativo, precio, lista de beneficios y badge (recomendado, popular, etc).

OBJECTIONS — Elimina el miedo a comprar:
- Mínimo 3 objeciones reales que el cliente típico tendría.
- Las respuestas deben ser empáticas primero, luego racionales.
- Orden: de la más común a la más específica.

CUSTOM SECTION — Sección diferenciadora:
- Si hay customPrompt del cliente, úsalo como guía principal.
- Si no, crea una sección que sea el diferenciador único del negocio 
  (historia de origen, proceso artesanal, garantía especial, etc).

CONSISTENCIA VISUAL:
- Asegúrate que el tono del copy sea coherente de principio a fin.
- El estilo visual solicitado debe reflejarse en la energía del texto 
  (minimalista = textos precisos y elegantes, bold = textos directos e impactantes).

FORMATO DE RESPUESTA:
Devuelve ÚNICAMENTE un objeto JSON válido, sin texto adicional, sin markdown, sin explicaciones.
"""


# ─────────────────────────────────────────────────────────────────────────────
# ESTRUCTURAS JSON — más ricas y con guías inline para el modelo
# ─────────────────────────────────────────────────────────────────────────────

ESTRUCTURA_BASIC = {
    "hero": {
        "headline": "Título poderoso orientado al beneficio principal (máx 10 palabras)",
        "subheadline": "2-3 oraciones que expanden el headline y eliminan la duda principal del cliente",
        "ctaButton": "Verbo de acción + beneficio (ej: Pedir mi caja ahora)"
    },
    "features": [
        {
            "emoji": "🎯",
            "title": "Beneficio principal en 3-5 palabras",
            "description": "Descripción concreta de 1-2 oraciones que explica el beneficio real para el cliente"
        },
        {
            "emoji": "⚡",
            "title": "Segundo beneficio clave",
            "description": "Descripción concreta y específica del beneficio"
        },
        {
            "emoji": "✅",
            "title": "Tercer beneficio diferenciador",
            "description": "Por qué esto importa para el cliente en su vida cotidiana"
        }
    ],
    "footer": {
        "contact": "contacto@nombredelNegocio.cl",
        "tagline": "Frase corta de cierre que refuerza la propuesta de valor"
    }
}

ESTRUCTURA_INTERMEDIATE = {
    "hero": {
        "headline": "Título poderoso orientado al beneficio principal (máx 10 palabras)",
        "subheadline": "2-3 oraciones que expanden el headline, eliminan la duda principal y crean deseo",
        "ctaButton": "Verbo de acción específico + beneficio inmediato",
        "badge": "Etiqueta de credibilidad corta (ej: +200 clientes satisfechos)"
    },
    "features": [
        {
            "emoji": "🎯",
            "title": "Beneficio principal",
            "description": "2 oraciones concretas sobre cómo esto mejora la vida/negocio del cliente"
        },
        {
            "emoji": "⚡",
            "title": "Velocidad o conveniencia",
            "description": "2 oraciones sobre el ahorro de tiempo o facilidad"
        },
        {
            "emoji": "🏆",
            "title": "Calidad o diferenciador",
            "description": "2 oraciones sobre qué hace único a este negocio"
        },
        {
            "emoji": "🤝",
            "title": "Confianza o garantía",
            "description": "2 oraciones sobre por qué es seguro comprar/contratar"
        }
    ],
    "socialProof": {
        "urgencyText": "Texto de urgencia creíble y específico (escasez o tiempo limitado)",
        "shippingText": "Beneficio logístico concreto (despacho, tiempo, costo)"
    },
    "faq": [
        {
            "question": "La pregunta más frecuente antes de comprar",
            "answer": "Respuesta directa, tranquilizadora y específica"
        },
        {
            "question": "Pregunta sobre el proceso o cómo funciona",
            "answer": "Explicación clara y simple del proceso"
        },
        {
            "question": "Pregunta sobre garantía o qué pasa si no quedo satisfecho",
            "answer": "Respuesta que elimine el riesgo percibido"
        }
    ],
    "footer": {
        "contact": "contacto@nombreDelNegocio.cl",
        "tagline": "Frase de cierre que refuerza la propuesta de valor"
    }
}

ESTRUCTURA_PREMIUM = {
    "hero": {
        "headline": "El mejor headline posible: poderoso, específico, orientado al resultado (máx 10 palabras)",
        "subheadline": "2-3 oraciones que expanden el headline, eliminan la duda principal y generan anticipación",
        "ctaButton": "CTA con verbo de acción + resultado esperado + urgencia implícita",
        "badge": "Etiqueta de autoridad y credibilidad (ej: ⭐ +500 familias satisfechas desde 2019)"
    },
    "features": [
        {
            "emoji": "🥇",
            "title": "Beneficio principal y más poderoso",
            "description": "2-3 oraciones ricas en detalles sobre el beneficio emocional y funcional"
        },
        {
            "emoji": "⚡",
            "title": "Velocidad, frescura o inmediatez",
            "description": "2-3 oraciones sobre la experiencia superior del cliente"
        },
        {
            "emoji": "🌿",
            "title": "Calidad, origen o proceso",
            "description": "2-3 oraciones sobre el estándar de calidad y cómo se logra"
        },
        {
            "emoji": "🤝",
            "title": "Confianza y respaldo",
            "description": "2-3 oraciones sobre la garantía, trayectoria o comunidad de clientes"
        },
        {
            "emoji": "💎",
            "title": "Diferenciador único y exclusivo",
            "description": "2-3 oraciones sobre lo que nadie más en el mercado ofrece"
        }
    ],
    "socialProof": {
        "urgencyText": "Urgencia creíble, específica y basada en demanda real",
        "shippingText": "Beneficio logístico premium con detalle (tiempo, costo, cobertura)",
        "testimonials": [
            {
                "name": "Nombre real chileno",
                "context": "Descripción del cliente (ej: Madre de familia, Santiago)",
                "quote": "Testimonio detallado que menciona un resultado específico y concreto (2-3 oraciones)",
                "rating": 5
            },
            {
                "name": "Otro nombre real chileno",
                "context": "Descripción del cliente (ej: Dueño de restaurante, Providencia)",
                "quote": "Testimonio diferente que destaca otro aspecto positivo con detalle específico",
                "rating": 5
            }
        ]
    },
    "pricing": [
        {
            "planName": "Nombre creativo del plan/producto base",
            "badge": "",
            "price": "Precio sugerido realista en CLP",
            "period": "por semana / por unidad / etc",
            "benefits": [
                "Beneficio 1 concreto",
                "Beneficio 2 concreto",
                "Beneficio 3 concreto"
            ],
            "ctaButton": "CTA específico para este plan"
        },
        {
            "planName": "Nombre creativo del plan/producto premium",
            "badge": "Más popular",
            "price": "Precio sugerido realista en CLP",
            "period": "por semana / por unidad / etc",
            "benefits": [
                "Todo lo anterior",
                "Beneficio extra 1",
                "Beneficio extra 2",
                "Beneficio extra 3"
            ],
            "ctaButton": "CTA específico para este plan"
        }
    ],
    "objections": [
        {
            "objection": "La objeción más común que impide la compra",
            "rebuttal": "Respuesta empática primero, luego racional. 2-3 oraciones que eliminan el miedo."
        },
        {
            "objection": "Objeción sobre precio o valor",
            "rebuttal": "Respuesta que reenmarca el precio como inversión con resultado concreto"
        },
        {
            "objection": "Objeción sobre confianza o experiencia previa negativa",
            "rebuttal": "Respuesta empática que valida la experiencia y ofrece garantía diferenciadora"
        }
    ],
    "customSection": {
        "title": "Título de la sección diferenciadora del negocio",
        "subtitle": "Subtítulo que complementa y da contexto",
        "content": "Contenido rico y detallado (3-4 oraciones) sobre el diferenciador único: historia de origen, proceso artesanal, misión, garantía especial, o lo que el customPrompt indique",
        "ctaButton": "CTA de esta sección"
    },
    "faq": [
        {
            "question": "La pregunta más frecuente antes de comprar",
            "answer": "Respuesta directa y tranquilizadora con detalle específico"
        },
        {
            "question": "¿Cómo funciona el proceso de compra o entrega?",
            "answer": "Explicación paso a paso, simple y clara"
        },
        {
            "question": "¿Qué garantía tienen sus productos/servicios?",
            "answer": "Garantía específica que elimine completamente el riesgo percibido"
        },
        {
            "question": "Pregunta específica del sector o tipo de negocio",
            "answer": "Respuesta detallada y experta que demuestre conocimiento del sector"
        }
    ],
    "footer": {
        "contact": "contacto@nombreDelNegocio.cl",
        "tagline": "Frase de cierre poderosa que resume la propuesta de valor en una oración",
        "socialLinks": {
            "instagram": "@nombreDelNegocio",
            "whatsapp": "+569XXXXXXXX"
        }
    }
}


@app.get("/")
def home():
    return {"message": "Motor de IA Multimodelo Activo con Progressive Disclosure"}


@app.post("/api/v1/ai/generate")
async def generate_landing(data: ProjectData):
    try:
        # ── 1. Normalizar plan
        plan_raw = (data.userPlan or "").strip()
        plan_normalizado = PLAN_ALIASES.get(plan_raw) or PLAN_ALIASES.get(plan_raw.upper())

        if not plan_normalizado:
            print(f"[WARN] Plan '{plan_raw}' no reconocido. Usando fallback '{PLAN_FALLBACK}'.")
            plan_normalizado = PLAN_FALLBACK

        modelo_elegido = MODELOS_IA[plan_normalizado]

        print(f"[{plan_normalizado}] Generando landing para: {data.projectName}")
        print(f"Usando el motor: {modelo_elegido}")

        # ── 2. Seleccionar system prompt y estructura por plan
        if plan_normalizado == "BASIC":
            system_prompt = SYSTEM_PROMPT_BASIC
            estructura = ESTRUCTURA_BASIC
        elif plan_normalizado == "INTERMEDIATE":
            system_prompt = SYSTEM_PROMPT_INTERMEDIATE
            estructura = ESTRUCTURA_INTERMEDIATE
        else:
            system_prompt = SYSTEM_PROMPT_PREMIUM
            estructura = ESTRUCTURA_PREMIUM

        # ── 3. Construir el user prompt con todo el contexto del negocio
        contexto_negocio = f"""
INFORMACIÓN DEL NEGOCIO:
- Nombre: {data.projectName}
- Descripción: {data.projectIdea}
- Call to Action deseado: {data.callToAction}
- Mercado objetivo: Chile"""

        if plan_normalizado in ["INTERMEDIATE", "PREMIUM"]:
            contexto_negocio += f"""
- Sector de la industria: {data.businessSector or 'No especificado'}
- Tono de comunicación de la marca: {data.communicationTone or 'Profesional'}
- Paleta de colores / Identidad visual: {data.colorPalette or 'No especificado'}"""

        if plan_normalizado == "PREMIUM":
            contexto_negocio += f"""
- Estilo visual objetivo: {data.visualStyle or 'Moderno y minimalista'}
- Nivel de animaciones: {data.animationLevel or 'Suaves'}"""
            if data.customPrompt:
                contexto_negocio += f"""

INSTRUCCIÓN ESPECIAL DEL CLIENTE (prioridad máxima — intégrala en customSection y donde sea relevante):
{data.customPrompt}"""

        estructura_str = json.dumps(estructura, ensure_ascii=False, indent=2)

        user_prompt = f"""{contexto_negocio}

TAREA:
Genera el contenido completo para la landing page de este negocio.
Aplica todos tus conocimientos de copywriting, CRO y estrategia de marca.
El contenido debe ser 100% específico para este negocio — nunca genérico.
Adapta el tono, vocabulario y ejemplos al sector y tipo de cliente de este negocio.

ESTRUCTURA JSON REQUERIDA:
Sigue exactamente esta estructura. Las descripciones entre comillas son guías — reemplázalas con contenido real:

{estructura_str}

REGLAS ESTRICTAS:
1. Devuelve ÚNICAMENTE el objeto JSON. Sin texto antes ni después.
2. Sin bloques de código markdown (no uses ```json).
3. Todo el contenido en español chileno.
4. Todos los campos deben tener contenido real y específico — nunca dejes placeholders.
5. El email del footer debe ser realista basado en el nombre del negocio.
6. Los emojis en features deben ser relevantes al beneficio que describen."""

        # ── 4. Llamada al modelo con system prompt separado
        response = client.chat.completions.create(
            model=modelo_elegido,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]
        )

        respuesta_ia = response.choices[0].message.content

        # ── 5. Extracción robusta del JSON
        # Primero intentar parsear directo
        try:
            ai_metadata_json = json.loads(respuesta_ia.strip())
        except json.JSONDecodeError:
            # Buscar bloque JSON dentro del texto
            match = re.search(r'\{.*\}', respuesta_ia, re.DOTALL)
            if match:
                try:
                    ai_metadata_json = json.loads(match.group(0))
                except json.JSONDecodeError:
                    ai_metadata_json = {
                        "error": "El modelo no generó un JSON válido",
                        "raw_response": respuesta_ia
                    }
            else:
                ai_metadata_json = {
                    "error": "No se encontró un JSON en la respuesta",
                    "raw_response": respuesta_ia
                }

        return {
            "status": "success",
            "projectId": data.projectId,
            "plan_usado": plan_normalizado,
            "plan_recibido": plan_raw,
            "ai_engine": modelo_elegido,
            "aiMetadata": ai_metadata_json
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error Crítico en la IA:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Error interno IA: {str(e)}")