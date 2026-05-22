# main.py — WLSuite AI Engine v3 (Compatible con backend Java + LandingViewer.jsx)
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

app = FastAPI(title="WLSuite AI Engine v3 — Full Stack Compatible")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ── Modelos por plan ──────────────────────────────────────────────────────────
MODELOS_IA = {
    "BASIC":        "google/gemini-2.5-flash-lite",
    "INTERMEDIATE": "openai/gpt-4o-mini",
    "PREMIUM":      "anthropic/claude-haiku-4.5",
}

MAX_TOKENS_BY_PLAN = {
    "BASIC":        3000,
    "INTERMEDIATE": 5500,
    "PREMIUM":      8000,
}

# ── Mapas semánticos ──────────────────────────────────────────────────────────
SECTOR_MAP = {
    "gastronomia":   "negocio gastronómico (restaurant, café, pastelería o similar)",
    "tecnologia":    "empresa o producto tecnológico (SaaS, app, servicio digital)",
    "salud":         "negocio del área de salud o bienestar (clínica, terapias, nutrición)",
    "educacion":     "plataforma o servicio educativo (cursos, tutorías, academia)",
    "moda":          "marca de moda, ropa, accesorios o estilo de vida",
    "fitness":       "negocio de fitness, deporte o entrenamiento físico",
    "legal":         "estudio jurídico o servicio legal profesional",
    "inmobiliaria":  "empresa o agente inmobiliario",
    "turismo":       "agencia de viajes, tours o alojamiento turístico",
    "ecommerce":     "tienda online o comercio electrónico generalista",
    "fintech":       "servicio financiero o fintech (pagos, inversión, crédito)",
    "consultoria":   "consultora o servicios de asesoría profesional",
    "construccion":  "empresa de construcción, arquitectura o remodelación",
    "belleza":       "salón de belleza, estética, cuidado personal o cosmética",
    "mascotas":      "negocio orientado al cuidado o productos para mascotas",
    "eventos":       "organización de eventos, bodas, celebraciones o entretenimiento",
    "arte":          "creador, artista, fotógrafo o agencia creativa",
    "automotriz":    "concesionario, taller mecánico o servicio automotriz",
    "ong":           "organización sin fines de lucro o causa social",
    "otro":          "negocio de propósito general",
}

LANDING_GOAL_MAP = {
    "leads":     "capturar datos de contacto de prospectos interesados (generación de leads)",
    "ventas":    "concretar ventas directas del producto o servicio",
    "reservas":  "recibir reservas o agendamientos de hora/mesa/sesión",
    "informar":  "informar y educar al visitante sobre el negocio o producto",
    "descargas": "promover la descarga de una app, ebook o recurso gratuito",
    "registro":  "lograr que el usuario se registre o cree una cuenta",
}

TARGET_AUDIENCE_MAP = {
    "jovenes":        "jóvenes de 18 a 28 años, nativos digitales",
    "adultos":        "adultos de 30 a 50 años con poder adquisitivo",
    "profesionales":  "profesionales y ejecutivos, tomadores de decisión",
    "empresas":       "empresas y negocios (mercado B2B)",
    "emprendedores":  "emprendedores y founders de startups",
    "padres":         "padres y madres con familia",
    "adultos-mayores":"adultos mayores de 55 años",
}

BRAND_POSITIONING_MAP = {
    "economico":      "precio accesible, propuesta de valor económica",
    "calidad-precio": "mejor relación calidad-precio del mercado",
    "premium":        "calidad superior, experiencia premium",
    "lujo":           "exclusividad, lujo y aspiracionalidad",
}

BRAND_STAGE_MAP = {
    "nueva-marca":   "marca nueva que se presenta por primera vez al mercado",
    "establecida":   "marca con trayectoria y reconocimiento en su sector",
    "relanzamiento": "marca en proceso de relanzamiento con nueva propuesta de valor",
}

TONE_MAP = {
    "profesional": "Tono profesional y confiable, lenguaje preciso y claro.",
    "cercano":     "Tono cercano y amigable, conversacional, como hablar con un amigo experto.",
    "elegante":    "Tono elegante y sofisticado, palabras cuidadas, sin coloquialismos.",
    "jovial":      "Tono jovial y con energía, frases cortas, entusiasmo auténtico.",
    "inspirador":  "Tono inspirador y motivacional, enfocado en el potencial del cliente.",
    "tecnico":     "Tono técnico y especializado, dirigido a un público experto en el área.",
}

FORMALITY_MAP = {
    "formal":      "Usa 'usted' en todo momento. Sin contracciones ni informalidades.",
    "semi-formal": "Tuteo respetuoso y amable. Natural pero profesional.",
    "informal":    "Totalmente coloquial y directo. Usa 'tú', contracciones chilenas naturales.",
}

BUTTON_STYLE_MAP = {
    "solido":    "botones con fondo sólido del color primario",
    "outline":   "botones con borde del color primario y fondo transparente",
    "gradiente": "botones con fondo degradado entre color primario y secundario",
}

ANIMATION_MAP = {
    "ninguna":   "sin ningún tipo de animación, página completamente estática",
    "sutil":     "animaciones muy suaves: fade-in simple en carga, sin distracciones",
    "moderada":  "animaciones moderadas: elementos aparecen al hacer scroll, hover suaves",
    "expresiva": "animaciones expresivas y vivas: transiciones fluidas, paralaje, microinteracciones",
}

CREATIVITY_MAP = {
    "conservadora": "sigue las convenciones de diseño web estándar",
    "equilibrada":  "mezcla creatividad con convención, propón algo fresco pero reconocible",
    "experimental": "sé creativo y audaz, propón estructuras inesperadas, jerarquías originales",
}

LAYOUT_MAP = {
    "centrado":   "layout centrado en columna única, contenido bien enmarcado al centro",
    "asimetrico": "layout asimétrico con texto a la izquierda e imágenes/gráficos a la derecha",
    "full-width": "secciones de ancho completo, elementos que sangran hasta los bordes",
    "tarjetas":   "contenido organizado en grillas de tarjetas para cada feature o beneficio",
}

# ── Mapa hex de colores ───────────────────────────────────────────────────────
COLOR_HEX_MAP = {
    "azul-marino":     "#1e3a5f",
    "azul-cielo":      "#3b82f6",
    "verde-bosque":    "#166534",
    "verde-menta":     "#10b981",
    "terracota":       "#b5541c",
    "rojo-vibrante":   "#dc2626",
    "morado":          "#7c3aed",
    "rosa":            "#db2777",
    "negro":           "#111827",
    "gris-oscuro":     "#374151",
    "gris-neutro":     "#9ca3af",
    "blanco":          "#ffffff",
    "crema":           "#fef9f0",
    "amarillo-dorado": "#d97706",
    "naranja":         "#ea580c",
    "cian":            "#0891b2",
}

LIGHT_COLORS = {"blanco", "crema", "amarillo-dorado", "gris-neutro"}

FONT_IMPORT_MAP = {
    "geometrica":     "https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800;900&display=swap",
    "sans-humanista": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
    "serif-clasico":  "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=Inter:wght@400;500;600&display=swap",
    "display":        "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap",
    "monospace":      "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap",
}

FONT_FAMILY_MAP = {
    "geometrica":     "'Poppins', sans-serif",
    "sans-humanista": "'Inter', sans-serif",
    "serif-clasico":  "'Playfair Display', serif",
    "display":        "'Space Grotesk', sans-serif",
    "monospace":      "'JetBrains Mono', monospace",
}

SECTIONS_LABELS = {
    "hero":         "Hero principal con headline, subtítulo y CTA destacado",
    "features":     "Sección de características o beneficios clave (mínimo 4)",
    "testimonials": "Testimonios o prueba social de clientes reales",
    "faq":          "Sección de preguntas frecuentes con respuestas concisas",
    "pricing":      "Tabla o sección de precios y planes",
    "urgency":      "Sección de urgencia, escasez o promoción con límite de tiempo",
}

class ProjectData(BaseModel):
    projectId: int
    userPlan: str
    projectName: str
    projectIdea: str
    callToAction: str

    # Plan Básico
    businessSector:   Optional[str] = None
    landingGoal:      Optional[str] = None
    targetAudience:   Optional[str] = None
    brandPositioning: Optional[str] = None
    brandStage:       Optional[str] = None
    valueProposition: Optional[str] = None

    # Intermedio
    communicationTone: Optional[str] = None
    formalityLevel:    Optional[str] = None

    # Intermedio+ (vienen de designPreferences en AiService.java)
    primaryColor:        Optional[str] = None
    secondaryColor:      Optional[str] = None
    baseMode:            Optional[str] = None
    contrastLevel:       Optional[str] = None
    visualStyle:         Optional[str] = None
    typographyHierarchy: Optional[str] = None
    visualDensity:       Optional[str] = None
    sectionDividers:     Optional[str] = None
    sections:            Optional[str] = None

    # Premium
    typographyStyle: Optional[str] = None
    buttonShape:     Optional[str] = None
    buttonStyle:     Optional[str] = None
    iconStyle:       Optional[str] = None
    layoutType:      Optional[str] = None
    creativityLevel: Optional[str] = None
    animationLevel:  Optional[str] = None
    scrollEffect:    Optional[str] = None
    heroEffect:      Optional[str] = None
    hoverIntensity:  Optional[str] = None
    contentDensity:  Optional[str] = None

    # Imágenes personalizadas
    heroImageUrl: Optional[str] = None
    logoImageUrl: Optional[str] = None


# ── Constructor del prompt ────────────────────────────────────────────────────
def build_prompt(data: ProjectData, plan: str) -> str:

    # ── CORRECCIÓN: truncar projectIdea a 800 chars para evitar overflow de tokens ──
    raw_idea     = data.projectIdea or ""
    project_idea = raw_idea[:800]
    if len(raw_idea) > 800:
        print(f"[WARN] projectIdea truncado de {len(raw_idea)} a 800 chars | projectId={data.projectId}")

    sector      = SECTOR_MAP.get(data.businessSector or "", "negocio")
    goal        = LANDING_GOAL_MAP.get(data.landingGoal or "", "informar al visitante")
    audience    = TARGET_AUDIENCE_MAP.get(data.targetAudience or "", "público general")
    positioning = BRAND_POSITIONING_MAP.get(data.brandPositioning or "", "relación calidad-precio")
    stage       = BRAND_STAGE_MAP.get(data.brandStage or "", "marca establecida")
    tone        = TONE_MAP.get(data.communicationTone or "", "Tono profesional y claro.")
    formality   = FORMALITY_MAP.get(data.formalityLevel or "", "Tuteo respetuoso.")

    prompt = f"""Eres el mejor copywriter de conversión de América Latina.
Crea el contenido completo para una landing page de {data.projectName}.

DESCRIPCIÓN DEL NEGOCIO: {project_idea}
SECTOR: {sector}
OBJETIVO DE LA LANDING: {goal}
AUDIENCIA OBJETIVO: {audience}
POSICIONAMIENTO DE MARCA: {positioning}
ETAPA DE MARCA: {stage}
PROPUESTA DE VALOR ADICIONAL: {data.valueProposition or 'No especificada'}
LLAMADO A LA ACCIÓN PRINCIPAL: {data.callToAction}
"""

    if plan in ("INTERMEDIATE", "PREMIUM"):
        creativity = CREATIVITY_MAP.get(data.creativityLevel or "", "equilibrada")
        layout     = LAYOUT_MAP.get(data.layoutType or "", "centrado")
        prompt += f"""
TONO DE COMUNICACIÓN: {tone}
FORMALIDAD: {formality}
ESTILO VISUAL: {data.visualStyle or 'moderno'}
DENSIDAD DE CONTENIDO: {data.visualDensity or 'equilibrado'}
CREATIVIDAD: {creativity}
LAYOUT: {layout}
"""

    if plan == "PREMIUM":
        btn_style  = BUTTON_STYLE_MAP.get(data.buttonStyle or "", "sólido")
        animation  = ANIMATION_MAP.get(data.animationLevel or "", "sutil")
        font_fam   = FONT_FAMILY_MAP.get(data.typographyStyle or "", "'Inter', sans-serif")
        prompt += f"""
TIPOGRAFÍA: {font_fam}
ESTILO DE BOTONES: {btn_style}
NIVEL DE ANIMACIÓN: {animation}
DENSIDAD DE CONTENIDO AVANZADA: {data.contentDensity or 'equilibrado'}
"""

    return prompt


# ── Estructura JSON esperada por plan ─────────────────────────────────────────
def get_json_structure(plan: str, sections_csv: str) -> str:
    sections = [s.strip() for s in sections_csv.split(",") if s.strip()]

    base = {
        "hero": {
            "badge":          "...",
            "headline":       "...",
            "subheadline":    "...",
            "ctaButton":      "...",
            "secondaryCta":   "...",
            "supportingText": "..."
        },
        "features": [
            {"icon": "⚡", "title": "...", "description": "..."},
            {"icon": "🎯", "title": "...", "description": "..."},
            {"icon": "✨", "title": "...", "description": "..."},
            {"icon": "🚀", "title": "...", "description": "..."},
        ],
        "footer": {
            "description": "...",
            "contact":     "contacto@empresa.cl",
            "phone":       "+56 9 XXXX XXXX",
            "legalText":   "Todos los derechos reservados.",
            "links": [
                {"label": "Inicio",    "href": "#hero"},
                {"label": "Servicios", "href": "#features"},
                {"label": "Contacto",  "href": "#contacto"},
            ]
        }
    }

    if plan == "BASIC":
        return json.dumps(base, ensure_ascii=False, indent=2)

    # ── INTERMEDIATE ──────────────────────────────────────────────────────────
    if plan == "INTERMEDIATE":
        if "testimonials" in sections:
            base["socialProof"] = {
                "title":    "...",
                "subtitle": "...",
                "stats": [
                    {"number": "...", "label": "...", "description": "..."},
                    {"number": "...", "label": "...", "description": "..."},
                    {"number": "...", "label": "...", "description": "..."},
                ],
                "testimonials": [
                    {"name": "...", "role": "...", "company": "...", "quote": "...", "rating": 5},
                    {"name": "...", "role": "...", "company": "...", "quote": "...", "rating": 5},
                    {"name": "...", "role": "...", "company": "...", "quote": "...", "rating": 4},
                ]
            }

        if "faq" in sections:
            base["faq"] = {
                "title":    "...",
                "subtitle": "...",
                "items": [
                    {"question": "...", "answer": "..."},
                    {"question": "...", "answer": "..."},
                    {"question": "...", "answer": "..."},
                    {"question": "...", "answer": "..."},
                ]
            }

        if "urgency" in sections:
            base["urgency"] = {
                "badge":          "...",
                "title":          "...",
                "subtitle":       "...",
                "countdown":      {"enabled": True, "label": "La oferta termina en:"},
                "benefitsList":   ["...", "...", "..."],
                "ctaButton":      "...",
                "supportingText": "..."
            }

        base["cta"] = {
            "title":        "...",
            "subtitle":     "...",
            "ctaButton":    "...",
            "secondaryCta": "...",
            "trustText":    "..."
        }

        base["footer"] = {
            "description": "...",
            "contact":     "contacto@empresa.cl",
            "phone":       "+56 9 XXXX XXXX",
            "legalText":   "Todos los derechos reservados.",
            "socialProof": "...",
            "links": [
                {"label": "Inicio",    "href": "#hero"},
                {"label": "Servicios", "href": "#features"},
                {"label": "Contacto",  "href": "#contacto"},
            ]
        }

        return json.dumps(base, ensure_ascii=False, indent=2)

    # ── PREMIUM ───────────────────────────────────────────────────────────────
    if "testimonials" in sections:
        base["socialProof"] = {
            "title":    "...",
            "subtitle": "...",
            "stats": [
                {"number": "...", "label": "...", "description": "..."},
                {"number": "...", "label": "...", "description": "..."},
                {"number": "...", "label": "...", "description": "..."},
            ],
            "testimonials": [
                {"name": "...", "role": "...", "company": "...", "quote": "...", "rating": 5},
                {"name": "...", "role": "...", "company": "...", "quote": "...", "rating": 5},
                {"name": "...", "role": "...", "company": "...", "quote": "...", "rating": 4},
            ]
        }

    if "pricing" in sections:
        base["pricing"] = {
            "badge":    "...",
            "title":    "...",
            "subtitle": "...",
            "plans": [
                {
                    "name":        "...",
                    "description": "...",
                    "price":       "...",
                    "period":      "...",
                    "featured":    False,
                    "benefits":    ["...", "...", "...", "...", "..."],
                    "notIncluded": [],
                    "ctaButton":   "..."
                },
                {
                    "name":        "...",
                    "description": "...",
                    "price":       "...",
                    "period":      "...",
                    "featured":    True,
                    "badge":       "Más popular",
                    "benefits":    ["...", "...", "...", "...", "..."],
                    "ctaButton":   "..."
                },
            ]
        }

    if "faq" in sections:
        base["faq"] = {
            "title":    "...",
            "subtitle": "...",
            "items": [
                {"question": "...", "answer": "..."},
                {"question": "...", "answer": "..."},
                {"question": "...", "answer": "..."},
                {"question": "...", "answer": "..."},
                {"question": "...", "answer": "..."},
            ]
        }

    if "urgency" in sections:
        base["urgency"] = {
            "badge":          "...",
            "title":          "...",
            "subtitle":       "...",
            "countdown":      {"enabled": True, "label": "La oferta termina en:"},
            "benefitsList":   ["...", "...", "...", "..."],
            "ctaButton":      "...",
            "supportingText": "..."
        }

    base["howItWorks"] = {
        "title":    "...",
        "subtitle": "...",
        "steps": [
            {"number": "01", "title": "...", "description": "..."},
            {"number": "02", "title": "...", "description": "..."},
            {"number": "03", "title": "...", "description": "..."},
        ]
    }

    base["cta"] = {
        "title":        "...",
        "subtitle":     "...",
        "ctaButton":    "...",
        "secondaryCta": "...",
        "trustText":    "..."
    }

    base["footer"] = {
        "description": "...",
        "contact":     "contacto@empresa.cl",
        "phone":       "+56 9 XXXX XXXX",
        "legalText":   "Todos los derechos reservados.",
        "socialProof": "...",
        "links": [
            {"label": "Inicio",    "href": "#hero"},
            {"label": "Servicios", "href": "#features"},
            {"label": "Contacto",  "href": "#contacto"},
        ]
    }

    return json.dumps(base, ensure_ascii=False, indent=2)


# ── Endpoint de salud ─────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "WLSuite AI Engine v3 — Full Stack Compatible activo"}


# ── Endpoint principal ────────────────────────────────────────────────────────
@app.post("/api/v1/ai/generate")
async def generate_landing(data: ProjectData):
    try:
        raw_plan = (data.userPlan or "").strip().upper()
        PLAN_MAP = {
            "BASICO":       "BASIC",
            "BÁSICO":       "BASIC",
            "BASIC":        "BASIC",
            "INTERMEDIO":   "INTERMEDIATE",
            "INTERMEDIATE": "INTERMEDIATE",
            "PREMIUM":      "PREMIUM",
        }
        plan = PLAN_MAP.get(raw_plan, "BASIC")

        if plan != raw_plan:
            print(f"[generate_landing] Plan normalizado: '{raw_plan}' → '{plan}'")

        model        = MODELOS_IA.get(plan, MODELOS_IA["BASIC"])
        max_tokens   = MAX_TOKENS_BY_PLAN.get(plan, 3000)
        sections_csv = data.sections or "hero,features,footer"

        print(f"[{plan}] Proyecto: {data.projectName} | Modelo: {model} | Secciones: {sections_csv}")

        # ── Construcción del prompt y la estructura esperada ──────────────────
        prompt         = build_prompt(data, plan)
        json_structure = get_json_structure(plan, sections_csv)

        full_prompt = f"""{prompt}

=== INSTRUCCIÓN DE SALIDA ===
Devuelve ÚNICAMENTE el objeto JSON.
Sin markdown, sin explicaciones, sin texto fuera del JSON.
Rellena cada "..." con copy real, específico y persuasivo para {data.projectName}.
Mercado objetivo: Chile. Usa español chileno natural (no neutro).

REGLAS DE CALIDAD:
1. Todos los textos deben ser específicos para el sector y la audiencia indicados
2. Los números en estadísticas deben ser creíbles y variados (no siempre "100%")
3. Los testimonios deben incluir nombres chilenos reales (Valentina, Diego, Catalina, etc.)
4. Las preguntas del FAQ deben ser objeciones reales de compra para este tipo de negocio
5. El headline del hero debe impactar en 8 palabras o menos

JSON esperado:
{json_structure}
"""
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                timeout=75.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres el mejor copywriter de conversión de América Latina, "
                            "especializado en landing pages para el mercado chileno. "
                            "Respondes SOLO con JSON válido. Nunca incluyes texto adicional, "
                            "nunca usas bloques markdown, nunca explicas tu respuesta."
                        )
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ]
            )
        except Exception as timeout_err:
            print(f"[ERROR] Timeout o fallo de red en modelo '{model}' | projectId={data.projectId}: {timeout_err}")
            raise HTTPException(
                status_code=504,
                detail=f"El modelo de IA tardó demasiado en responder. Intenta nuevamente. (plan={plan}, modelo={model})"
            )

        raw = response.choices[0].message.content.strip()

        # ── Limpieza defensiva del JSON ───────────────────────────────────────
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$",       "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        if not raw.startswith("{"):
            first_brace = raw.find("{")
            if first_brace != -1:
                print(f"[WARN] Texto antes del JSON — recortando desde posición {first_brace}")
                raw = raw[first_brace:]
            else:
                print(f"[ERROR] Sin JSON en respuesta (plan={plan}):\n{raw[:500]}")
                raise HTTPException(status_code=500, detail="El modelo no retornó un JSON válido.")

        parsed = json.loads(raw)
        print(f"[OK] JSON generado correctamente | projectId={data.projectId} | plan={plan} | claves={list(parsed.keys())}")

        return {"content": parsed}

    except HTTPException:
        raise
    except json.JSONDecodeError as je:
        print(f"[ERROR] JSON inválido del modelo | projectId={data.projectId}: {je}\nRaw: {raw[:300]}")
        raise HTTPException(status_code=500, detail="El modelo retornó un JSON malformado.")
    except Exception as e:
        print(f"[ERROR] Excepción no controlada en /api/v1/ai/generate | projectId={data.projectId}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")