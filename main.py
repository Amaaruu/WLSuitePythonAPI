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
    "PREMIUM":      "anthropic/claude-3.5-sonnet",
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
    "jovenes":         "jóvenes adultos de 18–28 años, nativos digitales, lenguaje informal y aspiracional",
    "adultos":         "adultos de 30–50 años, valoran la practicidad, lenguaje directo y claro",
    "adultos-mayores": "personas de 55+ años, prefieren claridad extrema, sin jerga técnica",
    "empresas":        "tomadores de decisión en empresas (B2B), lenguaje técnico y orientado a ROI",
    "profesionales":   "profesionales independientes o ejecutivos, lenguaje sofisticado y eficiente",
    "padres":          "padres y madres de familia, lenguaje cálido, enfocado en seguridad y confianza",
    "emprendedores":   "emprendedores y fundadores, lenguaje motivador, orientado a resultados",
}

BRAND_POSITIONING_MAP = {
    "economico":      "posicionamiento de precio bajo, accesible para todos, énfasis en ahorro",
    "calidad-precio": "mejor relación calidad-precio del mercado, equilibrio entre costo y valor",
    "premium":        "marca premium, calidad superior, dispuesta a pagar más por mejores resultados",
    "lujo":           "marca de lujo exclusiva, experiencia aspiracional, precio no es objeción",
}

BRAND_STAGE_MAP = {
    "nueva-marca":   "marca nueva que se presenta al mercado por primera vez",
    "establecida":   "marca con trayectoria y reconocimiento en su mercado",
    "relanzamiento": "marca que se renueva y vuelve con nueva propuesta de valor",
}

TONE_MAP = {
    "profesional": "Tono profesional, confiable, preciso. Sin informalidades.",
    "cercano":     "Tono cercano y conversacional, como hablarle a un amigo inteligente.",
    "elegante":    "Tono elegante y sofisticado. Palabras cuidadas, frases bien construidas.",
    "jovial":      "Tono jovial, divertido, con energía. Puede usar humor ligero.",
    "inspirador":  "Tono inspirador y motivacional, enfocado en el potencial del cliente.",
    "tecnico":     "Tono técnico y especializado, directo a expertos en el área.",
}

FORMALITY_MAP = {
    "formal":      "Usa usted, lenguaje formal, sin contracciones.",
    "semi-formal": "Tuteo respetuoso, amable pero sin exceso de informalidad.",
    "informal":    "Tuteando directamente, lenguaje coloquial y cercano.",
}

VISUAL_STYLE_MAP = {
    "minimalista": "estética minimalista: mucho espacio en blanco, tipografía limpia, pocos elementos",
    "moderno":     "diseño moderno y contemporáneo: líneas limpias, geometría plana, colores frescos",
    "corporativo": "diseño corporativo serio: estructura rígida, aspecto formal y confiable",
    "futurista":   "estética futurista y tecnológica: dark mode, acentos neón o cian, elementos angulares",
    "elegante":    "diseño de alta gama: paleta reducida, tipografía serif, mucho aire",
    "organico":    "estética orgánica y natural: colores de tierra, formas curvas, sensación cálida",
    "audaz":       "diseño audaz y llamativo: contrastes altos, tipografía display, impacto visual inmediato",
    "retro":       "estética retro o vintage: paletas desaturadas, tipografías con carácter",
}

TYPOGRAPHY_MAP = {
    "geometrica":     "tipografía geométrica sans-serif (Poppins, Montserrat): moderna, limpia, técnica",
    "sans-humanista": "tipografía sans-serif humanista (Inter, DM Sans): legible, amigable, versátil",
    "serif-clasico":  "tipografía serif clásica (Playfair, Lora): elegante, literaria, premium",
    "display":        "tipografía display de alto impacto (Space Grotesk, Clash Display): personalidad fuerte",
    "monospace":      "tipografía monoespaciada (JetBrains Mono): para marcas tech o dev",
}

BUTTON_SHAPE_MAP = {
    "cuadrado":   "botones con bordes rectos (border-radius 0–4px)",
    "redondeado": "botones con esquinas ligeramente redondeadas (border-radius 8–12px)",
    "pildora":    "botones en forma de píldora (border-radius 999px)",
}

BUTTON_STYLE_MAP = {
    "solido":    "botones de fondo sólido con color de relleno fuerte",
    "outline":   "botones tipo outline, solo borde visible sin relleno",
    "ghost":     "botones ghost, casi transparentes con hover de relleno",
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

    # Imágenes personalizadas (vienen dentro de designPreferences
    # y AiService.java las extrae como campos de primer nivel)
    heroImageUrl: Optional[str] = None
    logoImageUrl: Optional[str] = None


# ── Constructor del prompt ────────────────────────────────────────────────────

def build_prompt(data: ProjectData, plan: str) -> str:
    sector      = SECTOR_MAP.get(data.businessSector or "", "negocio")
    goal        = LANDING_GOAL_MAP.get(data.landingGoal or "", "informar al visitante")
    audience    = TARGET_AUDIENCE_MAP.get(data.targetAudience or "", "público general")
    positioning = BRAND_POSITIONING_MAP.get(data.brandPositioning or "", "relación calidad-precio")
    stage       = BRAND_STAGE_MAP.get(data.brandStage or "", "marca establecida")
    tone        = TONE_MAP.get(data.communicationTone or "", "Tono profesional y claro.")
    formality   = FORMALITY_MAP.get(data.formalityLevel or "", "Tuteo respetuoso.")
    v_style     = VISUAL_STYLE_MAP.get(data.visualStyle or "", "diseño moderno y limpio")
    typo        = TYPOGRAPHY_MAP.get(data.typographyStyle or "", "tipografía sans-serif humanista")
    btn_shape   = BUTTON_SHAPE_MAP.get(data.buttonShape or "", "botones redondeados")
    btn_style   = BUTTON_STYLE_MAP.get(data.buttonStyle or "", "botones sólidos")
    anim        = ANIMATION_MAP.get(data.animationLevel or "", "animaciones sutiles")
    creativity  = CREATIVITY_MAP.get(data.creativityLevel or "", "diseño equilibrado")
    layout      = LAYOUT_MAP.get(data.layoutType or "", "layout centrado")

    sections_csv  = data.sections or "hero,features,footer"
    sections_list = [s.strip() for s in sections_csv.split(",") if s.strip()]
    sections_desc = "\n".join(
        f"  - {SECTIONS_LABELS.get(s, s)}"
        for s in sections_list
    )

    # ── Bloque base — todos los planes ───────────────────────────────────────
    prompt = f"""Eres un copywriter experto en CRO (Conversion Rate Optimization) especializado
en landing pages de alta conversión para el mercado chileno.
Tu tarea es generar los textos persuasivos de una landing page real.

REGLA CRÍTICA: No uses frases genéricas como "soluciones innovadoras",
"calidad garantizada" o "expertos en el área". Cada palabra debe ser específica
para este negocio y resonar con su audiencia real.

=== CONTEXTO DEL NEGOCIO ===
Nombre: {data.projectName}
Tipo de negocio: {sector}
Propuesta de valor: {data.projectIdea}
Etapa de la marca: {stage}
Posicionamiento: {positioning}
Objetivo de la landing: {goal}
Público objetivo: {audience}
CTA principal: {data.callToAction}
"""

    if data.valueProposition:
        prompt += f"Diferenciador clave: {data.valueProposition}\n"

    # ── Bloque Intermedio+ ────────────────────────────────────────────────────
    if plan in ["INTERMEDIATE", "PREMIUM"]:
        prompt += f"""
=== IDENTIDAD DE COMUNICACIÓN ===
Tono: {tone}
Formalidad: {formality}
Color primario: {data.primaryColor or 'azul marino'}
Color secundario: {data.secondaryColor or 'blanco'}
Modo base: {data.baseMode or 'claro'}
Contraste: {data.contrastLevel or 'estándar'}
Estilo visual: {v_style}
Densidad visual: {data.visualDensity or 'equilibrado'}
Separación entre secciones: {data.sectionDividers or 'limpia'}

=== SECCIONES OBLIGATORIAS (en este orden) ===
{sections_desc}
"""

    # ── Bloque Premium ────────────────────────────────────────────────────────
    if plan == "PREMIUM":
        prompt += f"""
=== DIRECTRICES AVANZADAS ===
Tipografía: {typo}
Jerarquía tipográfica: {data.typographyHierarchy or 'equilibrada'}
Layout: {layout}
Forma de botones: {btn_shape}
Estilo de botones: {btn_style}
Íconos: {data.iconStyle or 'outline'}
Animaciones: {anim}
Efecto de scroll: {data.scrollEffect or 'fade-in'}
Efecto hero: {data.heroEffect or 'ninguno'}
Intensidad hover: {data.hoverIntensity or 'sutil'}
Densidad de contenido: {data.contentDensity or 'equilibrado'}
Creatividad: {creativity}
"""

    return prompt

def get_json_structure(plan: str, sections_csv: str) -> str:
    sections = [s.strip() for s in sections_csv.split(",") if s.strip()]

    # Hero — presente en todos los planes
    base = {
        "hero": {
            "badge":           "...",
            "headline":        "...",
            "subheadline":     "...",
            "ctaButton":       "...",
            "secondaryCta":    "...",
            "trustIndicators": ["...", "...", "..."],
            "supportingText":  "..."
        }
    }

    # Features — todos los planes (BASIC incluye si está en sections)
    if "features" in sections or plan in ["INTERMEDIATE", "PREMIUM"]:
        base["features"] = [
            {"icon": "🎯", "title": "...", "description": "...", "highlight": False},
            {"icon": "⚡", "title": "...", "description": "...", "highlight": True},
            {"icon": "🛡️", "title": "...", "description": "...", "highlight": False},
            {"icon": "✨", "title": "...", "description": "...", "highlight": False},
        ]

    # ── BASIC ─────────────────────────────────────────────────────────────────
    if plan == "BASIC":
        base["footer"] = {
            "description": "...",
            "contact":     "contacto@empresa.cl",
            "legalText":   "Todos los derechos reservados."
        }
        return json.dumps(base, ensure_ascii=False, indent=2)

    # ── INTERMEDIATE ──────────────────────────────────────────────────────────
    if plan == "INTERMEDIATE":
        # socialProof: LandingViewer lee d.socialProof.testimonials y d.socialProof.stats
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

        # urgency: LandingViewer lee countdown.enabled (booleano)
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

        # cta final — siempre presente desde INTERMEDIATE
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
            "socialProof": "..."
        }
        return json.dumps(base, ensure_ascii=False, indent=2)

    # ── PREMIUM ───────────────────────────────────────────────────────────────
    # socialProof: misma estructura que INTERMEDIATE
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

    # faq: misma estructura que INTERMEDIATE
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

    # urgency: misma estructura que INTERMEDIATE
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

    # howItWorks: exclusivo Premium — LandingViewer lee d.howItWorks.steps
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
Devuelve ÚNICAMENTE el objeto JSON. Sin markdown, sin explicaciones, sin texto fuera del JSON.
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

        # ── Llamada al modelo de IA ───────────────────────────────────────────
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
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

        raw = response.choices[0].message.content.strip()

        # ── Limpieza defensiva ────────────────────────────────────────────────
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

        # ── Parseo ────────────────────────────────────────────────────────────
        try:
            content_data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON inválido (plan={plan}): {e}\n{raw[:1000]}")
            raise HTTPException(status_code=500, detail=f"JSON inválido del modelo: {str(e)}")
        primary_key   = data.primaryColor   or "azul-marino"
        secondary_key = data.secondaryColor or "azul-cielo"
        typo_key      = data.typographyStyle or "sans-humanista"
        base_mode     = data.baseMode        or "claro"

        primary_hex   = COLOR_HEX_MAP.get(primary_key,   "#1e3a5f")
        secondary_hex = COLOR_HEX_MAP.get(secondary_key, "#3b82f6")

        primary_text   = "#111827" if primary_key   in LIGHT_COLORS else "#ffffff"
        secondary_text = "#111827" if secondary_key in LIGHT_COLORS else "#ffffff"

        if base_mode == "oscuro":
            bg_primary   = "#0a0a0f"
            bg_secondary = "#13131a"
            text_base    = "#f0f0f5"
            text_muted   = "#9090a0"
            card_bg      = "#1a1a24"
            card_border  = "#2a2a3a"
        else:
            bg_primary   = "#ffffff"
            bg_secondary = "#f8fafc"
            text_base    = "#0f172a"
            text_muted   = "#64748b"
            card_bg      = "#ffffff"
            card_border  = "#e2e8f0"

        theme_obj = {
            "primaryColor":   primary_hex,
            "secondaryColor": secondary_hex,
            "primaryText":    primary_text,
            "secondaryText":  secondary_text,
            "fontImport":     FONT_IMPORT_MAP.get(typo_key, FONT_IMPORT_MAP["sans-humanista"]),
            "fontFamily":     FONT_FAMILY_MAP.get(typo_key, FONT_FAMILY_MAP["sans-humanista"]),
            "baseMode":       base_mode,
            "bgPrimary":      bg_primary,
            "bgSecondary":    bg_secondary,
            "textBase":       text_base,
            "textMuted":      text_muted,
            "cardBg":         card_bg,
            "cardBorder":     card_border,
            "buttonShape":    data.buttonShape    or "redondeado",
            "buttonStyle":    data.buttonStyle    or "solido",
            "animationLevel": data.animationLevel or "sutil",
            "visualStyle":    data.visualStyle    or "moderno",
            "scrollEffect":   data.scrollEffect   or "fade-in",
        }

        if data.heroImageUrl:
            theme_obj["heroImageUrl"] = data.heroImageUrl
        if data.logoImageUrl:
            theme_obj["logoImageUrl"] = data.logoImageUrl

        content_data["_theme"] = theme_obj
        return {
            "projectId": data.projectId,
            "status":    "success",
            "content":   content_data
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Generación fallida (plan={data.userPlan}): {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))