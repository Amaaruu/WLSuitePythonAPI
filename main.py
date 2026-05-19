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

app = FastAPI(title="WLSuite AI Engine v3 — Hyper-Personalized")

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

# ── Mapas semánticos mejorados ─────────────────────────────────────────────────

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
    "expresiva": "animaciones expresivas y vivas: transiciones fluidas, efectos de paralaje, microinteracciones",
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
    "features":     "Sección de características o beneficios clave (mínimo 3)",
    "testimonials": "Testimonios o prueba social de clientes reales",
    "faq":          "Sección de preguntas frecuentes con respuestas concisas",
    "pricing":      "Tabla o sección de precios y planes",
    "urgency":      "Sección de urgencia, escasez o promoción con límite de tiempo",
}

# ══════════════════════════════════════════════════════════════════════════════
# NUEVO: Biblioteca de técnicas de copywriting de alta conversión
# ══════════════════════════════════════════════════════════════════════════════

COPYWRITING_TECHNIQUES = {
    "ventas": """
TÉCNICAS DE COPYWRITING PARA VENTAS DIRECTAS:
- Usa la fórmula PAS: Problema → Agitación → Solución
- El headline debe tocar el dolor más grande del cliente, no describir el producto
- Incluye números específicos cuando sea posible (ej: "87% de nuestros clientes..." o "En menos de 3 días...")
- El CTA debe crear urgencia implícita sin sonar desesperado
- Los beneficios deben ser emocionales, no solo funcionales
- Incluye un "reason why" claro: ¿por qué debería creerte?
""",
    "leads": """
TÉCNICAS DE COPYWRITING PARA CAPTACIÓN DE LEADS:
- Usa la fórmula AIDA: Atención → Interés → Deseo → Acción
- El headline debe prometer una transformación específica
- Reduce la fricción del formulario mencionando lo que NO pedirás (ej: "sin spam, sin compromisos")
- El CTA debe enfatizar el valor que recibirán, no la acción que harán
- Usa social proof numérico ("Más de 2.400 personas ya...")
- Los features deben estar escritos como beneficios tangibles para el usuario
""",
    "reservas": """
TÉCNICAS DE COPYWRITING PARA RESERVAS:
- Genera urgencia real con disponibilidad limitada o fechas
- Usa el principio de reciprocidad: ofrece algo de valor antes de pedir la reserva
- El headline debe hacer imaginar la experiencia, no describirla
- Incluye garantías que eliminen el riesgo de la reserva
- Los testimonios deben mencionar la experiencia vivida específicamente
- El CTA debe sonar como el primer paso de algo emocionante, no como un compromiso
""",
    "informar": """
TÉCNICAS DE COPYWRITING PARA GENERAR AUTORIDAD:
- Usa la estructura "Problema conocido → Nueva perspectiva → Tu solución"
- El headline debe desafiar una creencia común del sector
- Incluye datos, estadísticas o hechos que sorprendan
- Los features son logros o metodologías, no características del servicio
- El CTA debe invitar a "aprender más" o "descubrir" antes de vender
- Los testimonios deben hablar de resultados concretos y medibles
""",
    "descargas": """
TÉCNICAS DE COPYWRITING PARA DESCARGAS:
- El headline debe hacer el recurso irresistible en 6 palabras o menos
- Usa la fórmula "Lo que obtendrás": lista de beneficios concretos del recurso
- Elimina toda fricción: menciona que es gratis, instantáneo, sin registro complejo
- El CTA debe ser activo y específico ("Descargar mi guía gratis" > "Descargar")
- Usa proof of value: "El mismo framework que usamos para conseguir X resultado"
""",
    "registro": """
TÉCNICAS DE COPYWRITING PARA REGISTRO:
- El headline debe vender la pertenencia, no el producto
- Usa el principio de comunidad: "Únete a X personas que ya..."
- Muestra claramente qué pasa DESPUÉS del registro
- Elimina objeciones proactivamente (privacidad, cancelación, precio)
- El CTA debe iniciar con un verbo de pertenencia ("Unirme", "Empezar mi cuenta")
- Los beneficios deben hablar del "estado futuro" del usuario registrado
""",
}

# Instrucciones de diseño específicas por estilo visual (como un designer real)
DESIGN_DIRECTION_MAP = {
    "minimalista": """
DIRECCIÓN DE DISEÑO — MINIMALISTA:
- El hero debe tener UN elemento visual dominante y mucho espacio en blanco
- Las features van en grilla de 3 columnas con iconos lineales muy simples
- Usa solo 2 pesos tipográficos: ultra light para cuerpo, black para títulos
- Los separadores de sección son SOLO espacio, nunca líneas ni formas
- Los botones tienen mucho padding horizontal, bordes sutiles
- La paleta es casi monocromática: 95% del color primario y secundario como acento único
""",
    "moderno": """
DIRECCIÓN DE DISEÑO — MODERNO:
- El hero usa una grilla asimétrica: texto a la izquierda, elemento visual flotante a la derecha
- Los cards de features tienen sombras suaves y border-radius generoso
- Los títulos combinan un peso regular con un acento de color en una palabra clave
- Usa un color de acento vibrante contra un fondo neutro para crear jerarquía
- Los separadores de sección son formas SVG suaves (ondas o diagonales)
- El espaciado es generoso: las secciones respiran
""",
    "corporativo": """
DIRECCIÓN DE DISEÑO — CORPORATIVO:
- El hero tiene una imagen o ilustración profesional a la derecha del texto
- La estructura es simétrica y predecible: el usuario nunca se pierde
- Los colores primarios son azul marino o verde oscuro, transmiten confianza
- Los features tienen íconos de negocios (gráficos, escudos, check marks) en cuadros
- Los títulos son directos, sin metáforas: beneficio claro en el headline
- Footer completo con información legal visible
""",
    "futurista": """
DIRECCIÓN DE DISEÑO — FUTURISTA:
- El hero usa dark mode con un elemento de brillo (glow) en el color primario
- Los títulos del hero son enormes, con un gradiente de color que va de primario a secundario
- Los cards de features tienen bordes con gradiente y fondo semi-transparente (glassmorphism)
- Los separadores son líneas con gradiente que se desvanecen
- Los íconos son geométricos y angulares
- El CTA principal tiene un efecto de brillo o pulsación sutil
""",
    "elegante": """
DIRECCIÓN DE DISEÑO — ELEGANTE:
- El hero es espartano: marca, headline grande en serif, CTA discreto
- Mucho espacio en blanco, el contenido no compite entre sí
- Los testimonios van en citas tipográficas grandes, sin cards
- La paleta es oscura con dorado, crema o blanco roto como acento
- Los separadores son líneas finas o simplemente espacio generoso
- El pricing card premium tiene fondo oscuro con texto en oro o crema
""",
    "organico": """
DIRECCIÓN DE DISEÑO — ORGÁNICO:
- El hero tiene formas curvas o blobs de color en el fondo
- La paleta usa terracota, verde bosque, crema, o tonos de tierra
- Los cards de features tienen bordes muy redondeados (24px+) 
- La tipografía mezcla serif suave con sans-serif cálida
- Los separadores son ondas SVG con los colores de la paleta
- El tono visual transmite calidez, cercanía, naturalidad
""",
    "audaz": """
DIRECCIÓN DE DISEÑO — AUDAZ:
- El hero tiene el headline en tipografía enorme que ocupa casi toda la pantalla
- Usa bloques de color sólido como fondos de sección (no solo blanco/gris)
- Los contrastes son máximos: texto blanco sobre colores saturados
- Los features son listas con números grandes y decorativos
- El CTA es un bloque de color completo, muy visible, imposible de ignorar
- El layout rompe la cuadrícula en al menos una sección
""",
    "retro": """
DIRECCIÓN DE DISEÑO — RETRO:
- La paleta usa colores desaturados o sepia: mostaza, terracota, verde oliva
- La tipografía tiene personalidad: sans-serif condensada o serif con serifa cuadrada
- Los cards tienen bordes gruesos y sombras de offset (sin blur: box-shadow: 4px 4px 0)
- Los badges y etiquetas tienen estilo "stamp" o "sticker"
- Los separadores son líneas gruesas o filas de puntos/guiones
- El hero tiene una estética de "afiche" o "cartel" con texto muy protagonista
""",
}

# ── Modelo Pydantic ───────────────────────────────────────────────────────────

class ProjectData(BaseModel):
    projectId: int
    userPlan: str
    projectName: str
    projectIdea: str
    callToAction: str

    businessSector:   Optional[str] = None
    landingGoal:      Optional[str] = None
    targetAudience:   Optional[str] = None
    brandPositioning: Optional[str] = None
    brandStage:       Optional[str] = None
    valueProposition: Optional[str] = None

    communicationTone: Optional[str] = None
    formalityLevel:    Optional[str] = None

    primaryColor:        Optional[str] = None
    secondaryColor:      Optional[str] = None
    baseMode:            Optional[str] = None
    contrastLevel:       Optional[str] = None
    visualStyle:         Optional[str] = None
    typographyHierarchy: Optional[str] = None
    visualDensity:       Optional[str] = None
    sectionDividers:     Optional[str] = None
    sections:            Optional[str] = None

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


# ══════════════════════════════════════════════════════════════════════════════
# NUEVO: Constructor de "personalidad de marca" antes del prompt
# ══════════════════════════════════════════════════════════════════════════════

def build_brand_personality(data: ProjectData) -> str:
    """
    Genera un párrafo de contexto de marca que el modelo usa para
    mantener coherencia en todo el copy generado.
    """
    sector      = SECTOR_MAP.get(data.businessSector or "", "negocio")
    positioning = BRAND_POSITIONING_MAP.get(data.brandPositioning or "", "relación calidad-precio")
    audience    = TARGET_AUDIENCE_MAP.get(data.targetAudience or "", "público general")
    stage       = BRAND_STAGE_MAP.get(data.brandStage or "", "marca establecida")
    tone        = TONE_MAP.get(data.communicationTone or "", "Tono profesional y claro.")
    formality   = FORMALITY_MAP.get(data.formalityLevel or "", "Tuteo respetuoso.")

    personality = f"""
=== PERSONALIDAD DE MARCA (úsala como filtro para TODO el copy) ===
Nombre: {data.projectName}
Qué hace: {data.projectIdea}
Sector: {sector}
Etapa: {stage}
Posicionamiento: {positioning}
Audiencia principal: {audience}
CTA central: {data.callToAction}
"""
    if data.valueProposition:
        personality += f"Diferenciador único: {data.valueProposition}\n"

    personality += f"""
Voz de la marca: {tone} {formality}

REGLA DE ORO: Todo el copy debe sentirse como si lo hubiera escrito alguien que conoce profundamente
este negocio y su audiencia. NUNCA uses frases genéricas como:
- "Soluciones innovadoras para tu negocio"
- "Calidad y profesionalismo garantizados"
- "Tu satisfacción es nuestra prioridad"
- "Contamos con años de experiencia"

En su lugar, sé ESPECÍFICO con el contexto del negocio: menciona detalles del sector,
usa el lenguaje que usa la audiencia objetivo, y conecta emocionalmente con el problema real.
"""
    return personality


def build_section_instructions(sections_csv: str, goal: str, plan: str) -> str:
    """
    Genera instrucciones específicas por sección para que el modelo
    sepa exactamente qué escribir en cada una.
    """
    sections = [s.strip() for s in sections_csv.split(",") if s.strip()]
    copy_techniques = COPYWRITING_TECHNIQUES.get(goal or "informar", COPYWRITING_TECHNIQUES["informar"])

    instructions = f"""
=== TÉCNICAS DE COPYWRITING PARA ESTE OBJETIVO ===
{copy_techniques}

=== INSTRUCCIONES ESPECÍFICAS POR SECCIÓN ===
"""
    if "hero" in sections:
        instructions += """
HERO:
- headline: Máximo 8 palabras. Debe golpear una emoción o tocar un dolor real. NO describas el producto.
  Ejemplos buenos: "Para de perder clientes por una web mediocre", "Tu próxima gran venta empieza aquí"
  Ejemplos malos: "Bienvenido a nuestro sitio web", "Somos líderes en el sector"
- subheadline: 1–2 oraciones que expanden el headline con el beneficio concreto.
  Menciona el nombre del negocio naturalmente. Máximo 20 palabras.
- ctaButton: Verbo de acción + resultado específico. Ej: "Empezar ahora", "Ver cómo funciona", "Reservar mi lugar"
- secondaryCta: Opción menos comprometida para los indecisos. Ej: "Ver ejemplos primero"
- badge: Pequeña etiqueta sobre el headline. Ej: "Nuevo ✦", "Solo por tiempo limitado", "Exclusivo para Santiago"
- trustIndicators: Lista de 3–4 pruebas sociales cortas. Ej: ["Sin tarjeta requerida", "Más de 500 clientes", "Respuesta en 24h"]
"""

    if "features" in sections:
        instructions += """
FEATURES (mínimo 4, máximo 6):
- Cada feature debe resolver una objeción específica o resaltar un beneficio único
- El título debe ser una afirmación de beneficio, no una descripción técnica
  Malo: "Tecnología avanzada" | Bueno: "Resultados en 48 horas"
- La descripción debe explicar el PORQUÉ importa ese beneficio para el usuario
- Al menos UN feature debe tener "highlight: true" para destacarlo visualmente
- Los iconos deben ser emojis relevantes al sector y al beneficio específico
"""

    if "testimonials" in sections or plan in ["INTERMEDIATE", "PREMIUM"]:
        instructions += """
TESTIMONIOS (mínimo 3):
- Las citas deben ser específicas, NO genéricas. Incluyen:
  * Un resultado concreto ("Aumenté mis ventas un 40% en el primer mes")
  * Una referencia al antes/después ("Antes tardaba horas, ahora minutos")
  * Una emoción genuina ("Por fin puedo dormir tranquila")
- Los nombres deben sonar reales para Chile (Valentina, Diego, Catalina, Matías, etc.)
- El rol debe ser creíble y específico para el sector del negocio
- Varía el rating: no todos 5 estrellas (usa 4 y 5, nunca menos)
"""

    if "faq" in sections:
        instructions += """
FAQ (mínimo 4 preguntas):
- Las preguntas deben ser las OBJECIONES REALES de compra, no preguntas técnicas
  Ejemplos: "¿Qué pasa si no quedo satisfecho?", "¿Cuánto tiempo toma ver resultados?",
            "¿Necesito experiencia previa?", "¿Hay contrato de permanencia?"
- Las respuestas eliminan la objeción y refuerzan la confianza
- Al menos una pregunta debe manejar la objeción de precio
"""

    if "pricing" in sections:
        instructions += """
PRICING (1–3 planes):
- Si hay un plan destacado, debe tener "featured: true"
- Los precios deben ser en formato chileno o usar "$XX.000 CLP" si el contexto lo pide
- Cada plan debe tener mínimo 5 beneficios específicos, no genéricos
- El plan featured debe tener un badge visible ("Más popular", "Mejor valor")
- Los beneficios usan el lenguaje del sector específico
"""

    if "urgency" in sections:
        instructions += """
URGENCIA:
- La razón de la urgencia debe ser CREÍBLE para el sector (no inventada)
  Ej: gastronomía → "Solo 12 mesas disponibles este fin de semana"
      tecnología  → "Precio de lanzamiento hasta el viernes"
      salud       → "Últimas 3 horas disponibles este mes"
- El countdown.enabled debe ser true para activar el contador visual
- Los benefitsList son los beneficios que perderán si no actúan ahora
"""

    return instructions


# ── Constructor principal del prompt ─────────────────────────────────────────

def build_prompt(data: ProjectData, plan: str) -> str:
    goal       = data.landingGoal or "informar"
    v_style    = VISUAL_STYLE_MAP.get(data.visualStyle or "", "diseño moderno y limpio")
    typo       = TYPOGRAPHY_MAP.get(data.typographyStyle or "", "tipografía sans-serif humanista")
    btn_shape  = BUTTON_SHAPE_MAP.get(data.buttonShape or "", "botones redondeados")
    btn_style  = BUTTON_STYLE_MAP.get(data.buttonStyle or "", "botones sólidos")
    anim       = ANIMATION_MAP.get(data.animationLevel or "", "animaciones sutiles")
    creativity = CREATIVITY_MAP.get(data.creativityLevel or "", "diseño equilibrado")
    layout     = LAYOUT_MAP.get(data.layoutType or "", "layout centrado")

    sections_csv  = data.sections or "hero,features,footer"
    sections_list = [s.strip() for s in sections_csv.split(",") if s.strip()]
    sections_desc = "\n".join(
        f"  - {SECTIONS_LABELS.get(s, s)}"
        for s in sections_list
    )

    # Bloque de personalidad de marca (todos los planes)
    prompt = build_brand_personality(data)

    # Instrucciones de secciones con técnicas de CRO (todos los planes)
    prompt += build_section_instructions(sections_csv, goal, plan)

    # Bloque de diseño — Intermedio+
    if plan in ["INTERMEDIATE", "PREMIUM"]:
        design_direction = DESIGN_DIRECTION_MAP.get(data.visualStyle or "", "")
        prompt += f"""
=== IDENTIDAD VISUAL Y DE COMUNICACIÓN ===
Estilo visual: {v_style}
Paleta primaria: {data.primaryColor or 'azul marino'}
Paleta secundaria: {data.secondaryColor or 'blanco'}
Modo base: {data.baseMode or 'claro'}
Contraste: {data.contrastLevel or 'estándar'}
Densidad visual: {data.visualDensity or 'equilibrado'}
Separación de secciones: {data.sectionDividers or 'limpia'}

{design_direction}

=== ESTRUCTURA OBLIGATORIA ===
La landing debe incluir EXACTAMENTE estas secciones en este orden:
{sections_desc}
"""

    # Bloque avanzado — Premium
    if plan == "PREMIUM":
        prompt += f"""
=== DIRECTRICES AVANZADAS DE DISEÑO ===
Tipografía: {typo}
Jerarquía tipográfica: {data.typographyHierarchy or 'equilibrada'}
Layout: {layout}
Forma de botones: {btn_shape}
Estilo de botones: {btn_style}
Íconos: {data.iconStyle or 'outline'}
Animaciones: {anim}
Efecto scroll: {data.scrollEffect or 'fade-in'}
Efecto hero: {data.heroEffect or 'ninguno'}
Creatividad: {creativity}

INSTRUCCIÓN PREMIUM: Este cliente pagó el plan más alto. El copy debe ser el mejor trabajo de
tu carrera como copywriter. Cada palabra debe estar justificada. La estructura debe ser memorable.
Los titulos de sección deben ser creativos, no genéricos ("¿Por qué nosotros?" está prohibido).
"""

    return prompt


# ── Estructuras JSON enriquecidas por plan ────────────────────────────────────

def get_json_structure(plan: str, sections_csv: str) -> str:
    sections = [s.strip() for s in sections_csv.split(",") if s.strip()]

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

    if "features" in sections or plan in ["INTERMEDIATE", "PREMIUM"]:
        base["features"] = [
            {"icon": "🎯", "title": "...", "description": "...", "highlight": False},
            {"icon": "⚡", "title": "...", "description": "...", "highlight": True},
            {"icon": "🛡️", "title": "...", "description": "...", "highlight": False},
            {"icon": "✨", "title": "...", "description": "...", "highlight": False},
        ]

    # BASIC
    if plan == "BASIC":
        base["footer"] = {
            "description": "...",
            "contact":     "contacto@empresa.cl",
            "legalText":   "Todos los derechos reservados."
        }
        return json.dumps(base, ensure_ascii=False, indent=2)

    # INTERMEDIATE
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
                "badge":         "...",
                "title":         "...",
                "subtitle":      "...",
                "countdown":     {"enabled": True, "label": "La oferta termina en:"},
                "benefitsList":  ["...", "...", "..."],
                "ctaButton":     "...",
                "supportingText":"..."
            }
        base["cta"] = {
            "title":       "...",
            "subtitle":    "...",
            "ctaButton":   "...",
            "secondaryCta":"...",
            "trustText":   "..."
        }
        base["footer"] = {
            "description": "...",
            "contact":     "contacto@empresa.cl",
            "phone":       "+56 9 XXXX XXXX",
            "legalText":   "Todos los derechos reservados.",
            "socialProof": "..."
        }
        return json.dumps(base, ensure_ascii=False, indent=2)

    # PREMIUM
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
            "badge":         "...",
            "title":         "...",
            "subtitle":      "...",
            "countdown":     {"enabled": True, "label": "La oferta termina en:"},
            "benefitsList":  ["...", "...", "...", "..."],
            "ctaButton":     "...",
            "supportingText":"..."
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
        "title":       "...",
        "subtitle":    "...",
        "ctaButton":   "...",
        "secondaryCta":"...",
        "trustText":   "..."
    }
    base["footer"] = {
        "description": "...",
        "contact":     "contacto@empresa.cl",
        "phone":       "+56 9 XXXX XXXX",
        "legalText":   "Todos los derechos reservados.",
        "socialProof": "...",
        "links": [
            {"label": "Inicio",   "href": "#hero"},
            {"label": "Servicios","href": "#features"},
            {"label": "Contacto", "href": "#contacto"},
        ]
    }
    return json.dumps(base, ensure_ascii=False, indent=2)


# ── Endpoint principal ────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "WLSuite AI Engine v3 — Hyper-Personalized activo"}


@app.post("/api/v1/ai/generate")
async def generate_landing(data: ProjectData):
    try:
        # Normalización del plan
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

        # Construcción del prompt
        prompt         = build_prompt(data, plan)
        json_structure = get_json_structure(plan, sections_csv)

        full_prompt = f"""{prompt}

=== INSTRUCCIÓN FINAL DE SALIDA ===
Devuelve ÚNICAMENTE el objeto JSON. Cero texto adicional, cero markdown, cero explicaciones.

REGLAS CRÍTICAS DE CALIDAD:
1. Cada campo "..." debe tener contenido REAL y específico para "{data.projectName}"
2. El copy debe ser diferente al 99% de las landing pages del mismo sector
3. Nunca uses frases plantilla como "Soluciones de calidad" o "Expertos en el área"
4. Menciona el nombre del negocio al menos en el hero y en el CTA final
5. Los números en estadísticas deben ser creíbles y específicos (no siempre "100%", "1000+" o "99%")
6. Las objeciones en el FAQ deben ser las que REALMENTE se hace la audiencia de este sector

Mercado objetivo: Chile. Español chileno natural.

JSON esperado:
{json_structure}
"""

        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres el mejor copywriter de CRO (Conversion Rate Optimization) de América Latina, "
                        "especializado en landing pages de alta conversión para el mercado chileno. "
                        "Tu trabajo es generar copy que realmente convierte: específico, emocional, sin clichés. "
                        "Respondes SOLO con JSON válido, nunca con texto adicional, nunca con markdown."
                    )
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        )

        raw = response.choices[0].message.content.strip()

        # Limpieza defensiva del JSON
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

        try:
            content_data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON inválido (plan={plan}): {e}\n{raw[:1000]}")
            raise HTTPException(status_code=500, detail=f"JSON inválido: {str(e)}")

        # Inyección del tema visual
        primary_key   = data.primaryColor   or "azul-marino"
        secondary_key = data.secondaryColor or "azul-cielo"
        typo_key      = data.typographyStyle or "sans-humanista"
        base_mode     = data.baseMode        or "claro"

        primary_hex   = COLOR_HEX_MAP.get(primary_key,   "#1e3a5f")
        secondary_hex = COLOR_HEX_MAP.get(secondary_key, "#3b82f6")

        primary_text   = "#111827" if primary_key   in LIGHT_COLORS else "#ffffff"
        secondary_text = "#111827" if secondary_key in LIGHT_COLORS else "#ffffff"

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
            "scrollEffect":   data.scrollEffect   or "fade-in",
        }

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