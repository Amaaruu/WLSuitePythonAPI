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

app = FastAPI(title="WLSuite AI Engine v2 — Rich Context")

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

# ── Mapas semánticos: token del selector → instrucción en lenguaje natural ────
SECTOR_MAP = {
    "gastronomia":      "negocio gastronómico (restaurant, café, pastelería o similar)",
    "tecnologia":       "empresa o producto tecnológico (SaaS, app, servicio digital)",
    "salud":            "negocio del área de salud o bienestar (clínica, terapias, nutrición)",
    "educacion":        "plataforma o servicio educativo (cursos, tutorías, academia)",
    "moda":             "marca de moda, ropa, accesorios o estilo de vida",
    "fitness":          "negocio de fitness, deporte o entrenamiento físico",
    "legal":            "estudio jurídico o servicio legal profesional",
    "inmobiliaria":     "empresa o agente inmobiliario",
    "turismo":          "agencia de viajes, tours o alojamiento turístico",
    "ecommerce":        "tienda online o comercio electrónico generalista",
    "fintech":          "servicio financiero o fintech (pagos, inversión, crédito)",
    "consultoria":      "consultora o servicios de asesoría profesional",
    "construccion":     "empresa de construcción, arquitectura o remodelación",
    "belleza":          "salón de belleza, estética, cuidado personal o cosmética",
    "mascotas":         "negocio orientado al cuidado o productos para mascotas",
    "eventos":          "organización de eventos, bodas, celebraciones o entretenimiento",
    "arte":             "creador, artista, fotógrafo o agencia creativa",
    "automotriz":       "concesionario, taller mecánico o servicio automotriz",
    "ong":              "organización sin fines de lucro o causa social",
    "otro":             "negocio de propósito general",
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
    "economico":       "posicionamiento de precio bajo, accesible para todos, énfasis en ahorro",
    "calidad-precio":  "mejor relación calidad-precio del mercado, equilibrio entre costo y valor",
    "premium":         "marca premium, calidad superior, dispuesta a pagar más por mejores resultados",
    "lujo":            "marca de lujo exclusiva, experiencia aspiracional, precio no es objeción",
}

BRAND_STAGE_MAP = {
    "nueva-marca":    "marca nueva que se presenta al mercado por primera vez",
    "establecida":    "marca con trayectoria y reconocimiento en su mercado",
    "relanzamiento":  "marca que se renueva y vuelve con nueva propuesta de valor",
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
    "minimalista":  "estética minimalista: mucho espacio en blanco, tipografía limpia, pocos elementos, cada elemento con propósito claro",
    "moderno":      "diseño moderno y contemporáneo: líneas limpias, geometría plana, colores frescos",
    "corporativo":  "diseño corporativo serio: azules institucionales, estructura rígida, aspecto formal y confiable",
    "futurista":    "estética futurista y tecnológica: dark mode, acentos neón o cian, elementos geométricos angulares",
    "elegante":     "diseño de alta gama: paleta reducida de colores neutros o oscuros, tipografía serif, mucho aire",
    "organico":     "estética orgánica y natural: colores de tierra, formas curvas, sensación cálida y humana",
    "audaz":        "diseño audaz y llamativo: contrastes altos, tipografía display, impacto visual inmediato",
    "retro":        "estética retro o vintage: paletas desaturadas, tipografías con carácter, nostalgia controlada",
}

TYPOGRAPHY_MAP = {
    "geometrica":      "tipografía geométrica sans-serif (estilo Futura, Montserrat, Poppins): moderna, limpia, técnica",
    "sans-humanista":  "tipografía sans-serif humanista (estilo Inter, DM Sans): muy legible, amigable, versátil",
    "serif-clasico":   "tipografía serif clásica (estilo Playfair, Lora, Cormorant): elegante, literaria, premium",
    "display":         "tipografía display de alto impacto para títulos (estilo Space Grotesk, Clash Display): personalidad fuerte",
    "monospace":       "tipografía monoespaciada (estilo JetBrains Mono, Fira Code): para marcas tech o dev",
}

BUTTON_SHAPE_MAP = {
    "cuadrado":    "botones con bordes rectos (border-radius 0–4px)",
    "redondeado":  "botones con esquinas ligeramente redondeadas (border-radius 8–12px)",
    "pildora":     "botones en forma de píldora (border-radius 999px)",
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
    "expresiva": "animaciones expresivas y vivas: transiciones fluidas, efectos de paralaje, microinteracciones notorias",
}

CREATIVITY_MAP = {
    "conservadora":   "sigue las convenciones de diseño web estándar, no experimentes con estructuras inusuales",
    "equilibrada":    "mezcla creatividad con convención, propón algo fresco pero reconocible",
    "experimental":   "sé creativo y audaz, propón estructuras inesperadas, jerarquías visuales originales",
}

LAYOUT_MAP = {
    "centrado":    "layout centrado en columna única, contenido bien enmarcado al centro",
    "asimetrico":  "layout asimétrico con texto a la izquierda e imágenes/gráficos a la derecha (o viceversa)",
    "full-width":  "secciones de ancho completo, elementos que sangran hasta los bordes",
    "tarjetas":    "contenido organizado en grillas de tarjetas (cards) para cada feature o beneficio",
}

# ── Mapa semántico → hex real ─────────────────────────────────────────────────
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

# Colores claros: el texto sobre ellos debe ser oscuro
LIGHT_COLORS = {"blanco", "crema", "amarillo-dorado", "gris-neutro"}

# Tipografías reales de Google Fonts por clave semántica
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
    "features":     "Sección de características o beneficios clave (mínimo 3)",
    "testimonials": "Testimonios o prueba social de clientes reales",
    "faq":          "Sección de preguntas frecuentes con respuestas concisas",
    "pricing":      "Tabla o sección de precios y planes",
    "urgency":      "Sección de urgencia, escasez o promoción con límite de tiempo",
}


# ── Modelo Pydantic ───────────────────────────────────────────────────────────

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

    # Intermedio+ desde designPreferences
    primaryColor:        Optional[str] = None
    secondaryColor:      Optional[str] = None
    baseMode:            Optional[str] = None
    contrastLevel:       Optional[str] = None
    visualStyle:         Optional[str] = None
    typographyHierarchy: Optional[str] = None
    visualDensity:       Optional[str] = None
    sectionDividers:     Optional[str] = None
    sections:            Optional[str] = None   # CSV: "hero,features,testimonials"

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


# ── Constructor de prompts ────────────────────────────────────────────────────

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
    prompt = f"""Eres un Copywriter experto en CRO (Conversion Rate Optimization) y diseño web de alta conversión.
Tu tarea es estructurar los textos persuasivos de una landing page para el siguiente negocio.

=== CONTEXTO DEL NEGOCIO ===
Nombre del proyecto: {data.projectName}
Tipo de negocio: {sector}
Propuesta de valor: {data.projectIdea}
Etapa de la marca: {stage}
Posicionamiento de precio: {positioning}
Objetivo principal de la landing: {goal}
Público objetivo: {audience}
Llamado a la acción (CTA): {data.callToAction}
"""

    if data.valueProposition:
        prompt += f"Diferenciador clave del negocio: {data.valueProposition}\n"

    # ── Bloque Intermedio+ ────────────────────────────────────────────────────
    if plan in ["INTERMEDIATE", "PREMIUM"]:
        prompt += f"""
=== IDENTIDAD DE COMUNICACIÓN ===
Tono de la marca: {tone}
Nivel de formalidad: {formality}
Paleta de color primaria: {data.primaryColor or 'azul marino'}
Paleta de color secundaria: {data.secondaryColor or 'blanco'}
Modo base: {data.baseMode or 'claro'}
Nivel de contraste: {data.contrastLevel or 'estándar'}
Estilo visual: {v_style}
Densidad visual: {data.visualDensity or 'equilibrado'}
Separación de secciones: {data.sectionDividers or 'limpia'}

=== ESTRUCTURA DE SECCIONES SOLICITADA ===
La landing debe incluir OBLIGATORIAMENTE estas secciones en este orden:
{sections_desc}
"""

    # ── Bloque Premium ────────────────────────────────────────────────────────
    if plan == "PREMIUM":
        prompt += f"""
=== DIRECTRICES AVANZADAS DE DISEÑO ===
Tipografía: {typo}
Jerarquía tipográfica: {data.typographyHierarchy or 'equilibrada'}
Tipo de layout: {layout}
Forma de botones: {btn_shape}
Estilo de botones: {btn_style}
Estilo de íconos: {data.iconStyle or 'outline'}
Nivel de animaciones: {anim}
Efecto de scroll: {data.scrollEffect or 'fade-in'}
Efecto en hero: {data.heroEffect or 'ninguno'}
Intensidad de hover: {data.hoverIntensity or 'sutil'}
Densidad de contenido: {data.contentDensity or 'equilibrado'}
Nivel de creatividad permitido: {creativity}
"""

    return prompt


# ── Estructuras JSON por plan ─────────────────────────────────────────────────

def get_json_structure(plan: str, sections_csv: str) -> str:
    sections = [s.strip() for s in sections_csv.split(",") if s.strip()]

    base = {
        "hero": {
            "headline": "...",
            "subheadline": "...",
            "ctaButton": "..."
        }
    }

    if "features" in sections or plan in ["INTERMEDIATE", "PREMIUM"]:
        base["features"] = [
            {"title": "...", "description": "..."},
            {"title": "...", "description": "..."},
            {"title": "...", "description": "..."},
        ]

    if plan == "BASIC":
        base["footer"] = {"contact": "contacto@empresa.cl"}
        return json.dumps(base, ensure_ascii=False, indent=2)

    if plan == "INTERMEDIATE":
        if "testimonials" in sections:
            base["socialProof"] = {
                "urgencyText": "...",
                "shippingText": "..."
            }
        if "faq" in sections:
            base["faq"] = [
                {"question": "...", "answer": "..."},
                {"question": "...", "answer": "..."},
            ]
        base["footer"] = {"contact": "contacto@empresa.cl"}
        return json.dumps(base, ensure_ascii=False, indent=2)

    # PREMIUM
    if "testimonials" in sections:
        base["socialProof"] = {
            "urgencyText": "...",
            "shippingText": "...",
            "testimonials": [
                {"name": "...", "role": "...", "quote": "..."},
                {"name": "...", "role": "...", "quote": "..."},
            ]
        }
    if "pricing" in sections:
        base["pricing"] = [
            {"planName": "...", "price": "...", "benefits": ["...", "...", "..."]}
        ]
    if "faq" in sections:
        base["faq"] = [
            {"question": "...", "answer": "..."},
            {"question": "...", "answer": "..."},
            {"question": "...", "answer": "..."},
        ]
    if "urgency" in sections:
        base["urgency"] = {"title": "...", "countdown": "...", "ctaButton": "..."}

    base["footer"] = {"contact": "contacto@empresa.cl"}
    return json.dumps(base, ensure_ascii=False, indent=2)


# ── Endpoint principal ────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "WLSuite AI Engine v2 — Rich Context activo"}


@app.post("/api/v1/ai/generate")
async def generate_landing(data: ProjectData):
    try:
        plan = data.userPlan.upper()
        model = MODELOS_IA.get(plan, MODELOS_IA["BASIC"])
        sections_csv = data.sections or "hero,features,footer"

        print(f"[{plan}] Generando landing para: {data.projectName} | Modelo: {model}")

        # Construye el prompt semántico
        prompt = build_prompt(data, plan)

        # Obtiene la estructura JSON esperada
        json_structure = get_json_structure(plan, sections_csv)

        # Instrucción final al modelo
        full_prompt = f"""{prompt}

=== INSTRUCCIÓN DE SALIDA ===
Devuelve ÚNICAMENTE un objeto JSON válido con exactamente esta estructura.
No incluyas explicaciones, markdown, ni texto fuera del JSON.
Rellena cada campo "..." con el copy real y persuasivo.
Mercado objetivo: Chile. Usa el español chileno natural (no neutro).

Estructura esperada:
{json_structure}
"""

        response = client.chat.completions.create(
            model=model,
            max_tokens=3000,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en copywriting de conversión y diseño de landing pages. Respondes SOLO con JSON válido, nunca con texto adicional."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        )

        raw = response.choices[0].message.content.strip()

        # Limpieza defensiva: elimina bloques markdown si el modelo los agrega
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        content_data = json.loads(raw)

        # ── Inyectar tema visual en el contenido ──────────────────────────────
        primary_key   = data.primaryColor   or "azul-marino"
        secondary_key = data.secondaryColor or "azul-cielo"
        typo_key      = data.typographyStyle or "sans-humanista"
        base_mode     = data.baseMode       or "claro"

        primary_hex   = COLOR_HEX_MAP.get(primary_key,   "#1e3a5f")
        secondary_hex = COLOR_HEX_MAP.get(secondary_key, "#3b82f6")

        # Texto sobre color primario: oscuro si el color es claro, blanco si es oscuro
        primary_text   = "#111827" if primary_key   in LIGHT_COLORS else "#ffffff"
        secondary_text = "#111827" if secondary_key in LIGHT_COLORS else "#ffffff"

        # Fondos y textos base según modo claro u oscuro
        if base_mode == "oscuro":
            bg_primary   = "#0f172a"
            bg_secondary = "#1e293b"
            text_base    = "#f1f5f9"
            text_muted   = "#94a3b8"
            card_bg      = "#1e293b"
            card_border  = "#334155"
        else:
            bg_primary   = "#ffffff"
            bg_secondary = "#f8fafc"
            text_base    = "#0f172a"
            text_muted   = "#64748b"
            card_bg      = "#ffffff"
            card_border  = "#e2e8f0"

        content_data["_theme"] = {
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
        }
        # ─────────────────────────────────────────────────────────────────────

        return {
            "projectId": data.projectId,
            "status": "success",
            "content": content_data
        }

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON inválido del modelo: {e}")
        raise HTTPException(status_code=500, detail="El modelo retornó un JSON inválido.")
    except Exception as e:
        print(f"[ERROR] Generación fallida: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))