# test_main.py — WLSuite AI Engine v3 — Suite de tests completa
# Ejecutar con: pytest test_main.py -v --tb=short
# Con cobertura: pytest test_main.py -v --cov=main --cov-report=term-missing

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import (
    app,
    build_prompt,
    get_json_structure,
    ProjectData,
    SECTOR_MAP,
    LANDING_GOAL_MAP,
    TARGET_AUDIENCE_MAP,
    BRAND_POSITIONING_MAP,
    BRAND_STAGE_MAP,
    TONE_MAP,
    FORMALITY_MAP,
    BUTTON_STYLE_MAP,
    ANIMATION_MAP,
    CREATIVITY_MAP,
    LAYOUT_MAP,
    COLOR_HEX_MAP,
    FONT_FAMILY_MAP,
    FONT_IMPORT_MAP,
    SECTIONS_LABELS,
)

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def basic_project():
    return ProjectData(
        projectId=1,
        userPlan="BASIC",
        projectName="Mi Restaurante",
        projectIdea="Restaurante de comida chilena en Santiago",
        callToAction="Reserva tu mesa ahora",
        businessSector="gastronomia",
        landingGoal="reservas",
        targetAudience="adultos",
        brandPositioning="calidad-precio",
        brandStage="establecida",
        valueProposition="La mejor comida casera de Santiago",
    )

@pytest.fixture
def intermediate_project():
    return ProjectData(
        projectId=2,
        userPlan="INTERMEDIATE",
        projectName="TechSaaS",
        projectIdea="Plataforma SaaS de gestión empresarial",
        callToAction="Prueba gratis 14 días",
        businessSector="tecnologia",
        landingGoal="leads",
        targetAudience="empresas",
        brandPositioning="premium",
        brandStage="nueva-marca",
        valueProposition="Ahorra 10 horas semanales de trabajo",
        communicationTone="profesional",
        formalityLevel="semi-formal",
        visualStyle="moderno",
        visualDensity="equilibrado",
        creativityLevel="equilibrada",
        layoutType="asimetrico",
        sections="hero,features,testimonials,faq,footer",
    )

@pytest.fixture
def premium_project():
    return ProjectData(
        projectId=3,
        userPlan="PREMIUM",
        projectName="LuxurySpa",
        projectIdea="Spa de lujo con tratamientos exclusivos en Providencia",
        callToAction="Agenda tu experiencia",
        businessSector="belleza",
        landingGoal="reservas",
        targetAudience="adultos",
        brandPositioning="lujo",
        brandStage="establecida",
        valueProposition="Bienestar total en el corazón de Santiago",
        communicationTone="elegante",
        formalityLevel="formal",
        typographyStyle="serif-clasico",
        buttonShape="pill",
        buttonStyle="solido",
        iconStyle="lineal",
        layoutType="centrado",
        creativityLevel="experimental",
        animationLevel="expresiva",
        scrollEffect="parallax",
        heroEffect="video",
        hoverIntensity="alto",
        contentDensity="espacioso",
        sections="hero,features,testimonials,pricing,faq,urgency,footer",
    )

def make_mock_response(content: dict) -> MagicMock:
    """Construye un mock del objeto response de OpenAI."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(content, ensure_ascii=False)
    return mock_response


# ─────────────────────────────────────────────────────────────────────────────
# 1. HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_root_returns_200(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_message(self):
        r = client.get("/")
        assert "WLSuite AI Engine" in r.json()["message"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. MAPAS DE CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigMaps:
    """Valida que los mapas semánticos estén completos y no tengan valores vacíos."""

    def test_sector_map_no_empty_values(self):
        for k, v in SECTOR_MAP.items():
            assert v, f"SECTOR_MAP['{k}'] está vacío"

    def test_landing_goal_map_keys(self):
        expected = {"leads", "ventas", "reservas", "informar", "descargas", "registro"}
        assert set(LANDING_GOAL_MAP.keys()) == expected

    def test_tone_map_no_empty_values(self):
        for k, v in TONE_MAP.items():
            assert v, f"TONE_MAP['{k}'] está vacío"

    def test_font_import_and_family_maps_are_in_sync(self):
        assert set(FONT_IMPORT_MAP.keys()) == set(FONT_FAMILY_MAP.keys())

    def test_color_hex_map_values_are_valid_hex(self):
        import re
        hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
        for name, hex_val in COLOR_HEX_MAP.items():
            assert hex_pattern.match(hex_val), f"COLOR_HEX_MAP['{name}'] = '{hex_val}' no es un hex válido"

    def test_sections_labels_coverage(self):
        expected = {"hero", "features", "testimonials", "faq", "pricing", "urgency"}
        assert expected.issubset(set(SECTIONS_LABELS.keys()))

    def test_animation_map_has_all_levels(self):
        expected = {"ninguna", "sutil", "moderada", "expresiva"}
        assert set(ANIMATION_MAP.keys()) == expected

    def test_creativity_map_has_all_levels(self):
        expected = {"conservadora", "equilibrada", "experimental"}
        assert set(CREATIVITY_MAP.keys()) == expected


# ─────────────────────────────────────────────────────────────────────────────
# 3. build_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPrompt:

    def test_basic_prompt_contains_project_name(self, basic_project):
        prompt = build_prompt(basic_project, "BASIC")
        assert "Mi Restaurante" in prompt

    def test_basic_prompt_contains_cta(self, basic_project):
        prompt = build_prompt(basic_project, "BASIC")
        assert "Reserva tu mesa ahora" in prompt

    def test_basic_prompt_resolves_sector(self, basic_project):
        prompt = build_prompt(basic_project, "BASIC")
        assert "gastronómico" in prompt

    def test_basic_prompt_resolves_goal(self, basic_project):
        prompt = build_prompt(basic_project, "BASIC")
        assert "reservas" in prompt.lower()

    def test_intermediate_prompt_includes_tone(self, intermediate_project):
        prompt = build_prompt(intermediate_project, "INTERMEDIATE")
        assert "profesional" in prompt.lower()

    def test_intermediate_prompt_includes_creativity(self, intermediate_project):
        prompt = build_prompt(intermediate_project, "INTERMEDIATE")
        assert "equilibrada" in prompt.lower() or "mezcla creatividad" in prompt.lower()

    def test_premium_prompt_includes_typography(self, premium_project):
        prompt = build_prompt(premium_project, "PREMIUM")
        assert "Playfair" in prompt

    def test_premium_prompt_includes_animation(self, premium_project):
        prompt = build_prompt(premium_project, "PREMIUM")
        assert "expresiva" in prompt.lower() or "animaciones expresivas" in prompt.lower()

    def test_prompt_truncates_long_idea(self):
        long_idea = "A" * 1000
        data = ProjectData(
            projectId=99,
            userPlan="BASIC",
            projectName="Test",
            projectIdea=long_idea,
            callToAction="CTA",
        )
        prompt = build_prompt(data, "BASIC")
        # La idea truncada a 800 chars debe estar en el prompt
        assert "A" * 800 in prompt
        assert "A" * 801 not in prompt

    def test_prompt_handles_none_value_proposition(self):
        data = ProjectData(
            projectId=10,
            userPlan="BASIC",
            projectName="X",
            projectIdea="Idea",
            callToAction="CTA",
            valueProposition=None,
        )
        prompt = build_prompt(data, "BASIC")
        assert "No especificada" in prompt

    def test_basic_prompt_does_not_include_tone_block(self, basic_project):
        prompt = build_prompt(basic_project, "BASIC")
        assert "TONO DE COMUNICACIÓN:" not in prompt

    def test_intermediate_prompt_does_not_include_typography(self, intermediate_project):
        prompt = build_prompt(intermediate_project, "INTERMEDIATE")
        assert "TIPOGRAFÍA:" not in prompt

    def test_unknown_sector_falls_back_to_default(self):
        data = ProjectData(
            projectId=11,
            userPlan="BASIC",
            projectName="X",
            projectIdea="Idea",
            callToAction="CTA",
            businessSector="inexistente",
        )
        prompt = build_prompt(data, "BASIC")
        assert "negocio" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# 4. get_json_structure
# ─────────────────────────────────────────────────────────────────────────────

class TestGetJsonStructure:

    def test_basic_structure_has_hero(self):
        result = json.loads(get_json_structure("BASIC", "hero,features,footer"))
        assert "hero" in result

    def test_basic_structure_has_features(self):
        result = json.loads(get_json_structure("BASIC", "hero,features,footer"))
        assert "features" in result
        assert len(result["features"]) == 4

    def test_basic_structure_has_footer(self):
        result = json.loads(get_json_structure("BASIC", "hero,features,footer"))
        assert "footer" in result

    def test_basic_structure_does_not_have_cta(self):
        result = json.loads(get_json_structure("BASIC", "hero,features,footer"))
        assert "cta" not in result

    def test_intermediate_with_testimonials_adds_social_proof(self):
        result = json.loads(get_json_structure("INTERMEDIATE", "hero,features,testimonials,footer"))
        assert "socialProof" in result
        assert "testimonials" in result["socialProof"]
        assert len(result["socialProof"]["testimonials"]) == 3

    def test_intermediate_with_faq_adds_faq(self):
        result = json.loads(get_json_structure("INTERMEDIATE", "hero,features,faq,footer"))
        assert "faq" in result
        assert len(result["faq"]["items"]) == 4

    def test_intermediate_with_urgency_adds_urgency(self):
        result = json.loads(get_json_structure("INTERMEDIATE", "hero,features,urgency,footer"))
        assert "urgency" in result
        assert result["urgency"]["countdown"]["enabled"] is True

    def test_intermediate_always_has_cta(self):
        result = json.loads(get_json_structure("INTERMEDIATE", "hero,features,footer"))
        assert "cta" in result

    def test_premium_with_pricing_adds_pricing(self):
        result = json.loads(get_json_structure("PREMIUM", "hero,features,pricing,footer"))
        assert "pricing" in result
        assert len(result["pricing"]["plans"]) == 2

    def test_premium_pricing_has_featured_plan(self):
        result = json.loads(get_json_structure("PREMIUM", "hero,features,pricing,footer"))
        featured = [p for p in result["pricing"]["plans"] if p.get("featured")]
        assert len(featured) == 1

    def test_premium_always_has_how_it_works(self):
        result = json.loads(get_json_structure("PREMIUM", "hero,features,footer"))
        assert "howItWorks" in result
        assert len(result["howItWorks"]["steps"]) == 3

    def test_premium_faq_has_5_items(self):
        result = json.loads(get_json_structure("PREMIUM", "hero,features,faq,footer"))
        assert len(result["faq"]["items"]) == 5

    def test_premium_without_testimonials_excludes_social_proof(self):
        result = json.loads(get_json_structure("PREMIUM", "hero,features,footer"))
        assert "socialProof" not in result

    def test_structure_is_valid_json(self):
        for plan in ["BASIC", "INTERMEDIATE", "PREMIUM"]:
            raw = get_json_structure(plan, "hero,features,testimonials,faq,pricing,urgency,footer")
            parsed = json.loads(raw)
            assert isinstance(parsed, dict)

    def test_hero_has_required_keys(self):
        result = json.loads(get_json_structure("BASIC", "hero,features,footer"))
        hero_keys = {"badge", "headline", "subheadline", "ctaButton", "secondaryCta", "supportingText"}
        assert hero_keys.issubset(set(result["hero"].keys()))

    def test_footer_has_links(self):
        result = json.loads(get_json_structure("BASIC", "hero,features,footer"))
        assert isinstance(result["footer"]["links"], list)
        assert len(result["footer"]["links"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. ProjectData — validación de modelo Pydantic
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectDataModel:

    def test_minimal_valid_data(self):
        data = ProjectData(
            projectId=1,
            userPlan="BASIC",
            projectName="Test",
            projectIdea="Una idea",
            callToAction="Actúa ya",
        )
        assert data.projectId == 1

    def test_optional_fields_default_to_none(self):
        data = ProjectData(
            projectId=1,
            userPlan="BASIC",
            projectName="Test",
            projectIdea="Idea",
            callToAction="CTA",
        )
        assert data.heroImageUrl is None
        assert data.logoImageUrl is None
        assert data.typographyStyle is None

    def test_all_optional_fields_accepted(self, premium_project):
        assert premium_project.animationLevel == "expresiva"
        assert premium_project.heroEffect == "video"


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENDPOINT /api/v1/ai/generate — happy paths
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateEndpointHappyPath:

    MOCK_CONTENT = {
        "hero": {
            "badge": "Nuevo",
            "headline": "La mejor comida de Santiago",
            "subheadline": "Sabor auténtico",
            "ctaButton": "Reserva ahora",
            "secondaryCta": "Ver menú",
            "supportingText": "Más de 500 clientes satisfechos"
        },
        "features": [
            {"icon": "🍽️", "title": "Frescura", "description": "Ingredientes del día"},
            {"icon": "👨‍🍳", "title": "Chef experto", "description": "15 años de experiencia"},
            {"icon": "📍", "title": "Ubicación", "description": "En el centro de Santiago"},
            {"icon": "💲", "title": "Precio justo", "description": "Menú desde $5.000"},
        ],
        "footer": {
            "description": "El mejor restaurante de Chile",
            "contact": "contacto@mirestaurante.cl",
            "phone": "+56 9 1234 5678",
            "legalText": "Todos los derechos reservados.",
            "links": [{"label": "Inicio", "href": "#hero"}]
        }
    }

    @patch("main.client.chat.completions.create")
    def test_basic_plan_returns_200(self, mock_create):
        mock_create.return_value = make_mock_response(self.MOCK_CONTENT)
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Mi Restaurante",
            "projectIdea": "Restaurante chileno",
            "callToAction": "Reserva ahora"
        })
        assert r.status_code == 200

    @patch("main.client.chat.completions.create")
    def test_response_has_content_key(self, mock_create):
        mock_create.return_value = make_mock_response(self.MOCK_CONTENT)
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        assert "content" in r.json()

    @patch("main.client.chat.completions.create")
    def test_response_content_has_hero(self, mock_create):
        mock_create.return_value = make_mock_response(self.MOCK_CONTENT)
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        assert "hero" in r.json()["content"]

    @patch("main.client.chat.completions.create")
    def test_plan_normalization_basico(self, mock_create):
        """Acepta 'BASICO' (sin tilde) y lo normaliza a BASIC."""
        mock_create.return_value = make_mock_response(self.MOCK_CONTENT)
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASICO",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        assert r.status_code == 200

    @patch("main.client.chat.completions.create")
    def test_plan_normalization_basico_with_tilde(self, mock_create):
        """Acepta 'BÁSICO' (con tilde) y lo normaliza a BASIC."""
        mock_create.return_value = make_mock_response(self.MOCK_CONTENT)
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BÁSICO",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        assert r.status_code == 200

    @patch("main.client.chat.completions.create")
    def test_plan_normalization_intermedio(self, mock_create):
        mock_create.return_value = make_mock_response({**self.MOCK_CONTENT, "cta": {}})
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 2, "userPlan": "INTERMEDIO",
            "projectName": "Tech", "projectIdea": "SaaS", "callToAction": "Prueba gratis"
        })
        assert r.status_code == 200

    @patch("main.client.chat.completions.create")
    def test_unknown_plan_falls_back_to_basic(self, mock_create):
        mock_create.return_value = make_mock_response(self.MOCK_CONTENT)
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "DESCONOCIDO",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        assert r.status_code == 200

    @patch("main.client.chat.completions.create")
    def test_model_called_once(self, mock_create):
        mock_create.return_value = make_mock_response(self.MOCK_CONTENT)
        client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        mock_create.assert_called_once()

    @patch("main.client.chat.completions.create")
    def test_json_with_markdown_fences_is_cleaned(self, mock_create):
        """El endpoint debe limpiar bloques ```json ... ``` del output del modelo."""
        fenced = "```json\n" + json.dumps(self.MOCK_CONTENT) + "\n```"
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = fenced
        mock_create.return_value = mock_resp
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        assert r.status_code == 200
        assert "hero" in r.json()["content"]

    @patch("main.client.chat.completions.create")
    def test_json_with_leading_text_is_cleaned(self, mock_create):
        """El endpoint debe ignorar texto antes del JSON."""
        leading = "Claro, aquí tienes:\n" + json.dumps(self.MOCK_CONTENT)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = leading
        mock_create.return_value = mock_resp
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        assert r.status_code == 200
        assert "hero" in r.json()["content"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENDPOINT /api/v1/ai/generate — manejo de errores
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateEndpointErrors:

    @patch("main.client.chat.completions.create")
    def test_invalid_json_returns_500(self, mock_create):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "{ invalid json ::: }"
        mock_create.return_value = mock_resp
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        assert r.status_code == 500
        assert "malformado" in r.json()["detail"].lower()

    @patch("main.client.chat.completions.create")
    def test_model_returns_no_json_returns_500(self, mock_create):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Lo siento, no puedo responder eso."
        mock_create.return_value = mock_resp
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        assert r.status_code == 500
        assert "JSON válido" in r.json()["detail"]

    @patch("main.client.chat.completions.create")
    def test_timeout_returns_504(self, mock_create):
        import httpx
        mock_create.side_effect = httpx.TimeoutException("timeout")
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        assert r.status_code == 504

    @patch("main.client.chat.completions.create")
    def test_generic_exception_inside_ai_call_returns_504(self, mock_create):
        """RuntimeError dentro del bloque de llamada a la IA cae en el handler
        interno (timeout/network), que re-lanza como 504."""
        mock_create.side_effect = RuntimeError("Fallo inesperado")
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        # El código captura toda Exception en el try interno → HTTPException 504
        assert r.status_code == 504

    def test_missing_required_field_returns_422(self):
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            # falta projectName, projectIdea, callToAction
        })
        assert r.status_code == 422

    def test_empty_body_returns_422(self):
        r = client.post("/api/v1/ai/generate", json={})
        assert r.status_code == 422

    def test_wrong_content_type_returns_422(self):
        r = client.post("/api/v1/ai/generate", content="not json", headers={"Content-Type": "text/plain"})
        assert r.status_code in (422, 415)


# ─────────────────────────────────────────────────────────────────────────────
# 8. INTEGRACIÓN DE SECCIONES — verifica que el modelo correcto se use por plan
# ─────────────────────────────────────────────────────────────────────────────

class TestModelSelection:

    @patch("main.client.chat.completions.create")
    def test_basic_plan_uses_gemini(self, mock_create):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({
            "hero": {"badge": "", "headline": "", "subheadline": "", "ctaButton": "", "secondaryCta": "", "supportingText": ""},
            "features": [], "footer": {"description": "", "contact": "", "phone": "", "legalText": "", "links": []}
        })
        mock_create.return_value = mock_resp
        client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        call_kwargs = mock_create.call_args
        assert call_kwargs[1]["model"] == "google/gemini-2.5-flash-lite"

    @patch("main.client.chat.completions.create")
    def test_intermediate_plan_uses_gpt4o_mini(self, mock_create):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({
            "hero": {"badge": "", "headline": "", "subheadline": "", "ctaButton": "", "secondaryCta": "", "supportingText": ""},
            "features": [], "footer": {"description": "", "contact": "", "phone": "", "legalText": "", "links": []},
            "cta": {}
        })
        mock_create.return_value = mock_resp
        client.post("/api/v1/ai/generate", json={
            "projectId": 2, "userPlan": "INTERMEDIATE",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        call_kwargs = mock_create.call_args
        assert call_kwargs[1]["model"] == "openai/gpt-4o-mini"

    @patch("main.client.chat.completions.create")
    def test_premium_plan_uses_claude_haiku(self, mock_create):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({
            "hero": {"badge": "", "headline": "", "subheadline": "", "ctaButton": "", "secondaryCta": "", "supportingText": ""},
            "features": [], "footer": {"description": "", "contact": "", "phone": "", "legalText": "", "links": []},
            "howItWorks": {"title": "", "subtitle": "", "steps": []}, "cta": {}
        })
        mock_create.return_value = mock_resp
        client.post("/api/v1/ai/generate", json={
            "projectId": 3, "userPlan": "PREMIUM",
            "projectName": "Test", "projectIdea": "Idea", "callToAction": "CTA"
        })
        call_kwargs = mock_create.call_args
        assert call_kwargs[1]["model"] == "anthropic/claude-haiku-4.5"


# ─────────────────────────────────────────────────────────────────────────────
# 9. EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    @patch("main.client.chat.completions.create")
    def test_empty_sections_string_defaults_gracefully(self, mock_create):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({
            "hero": {"badge": "", "headline": "", "subheadline": "", "ctaButton": "", "secondaryCta": "", "supportingText": ""},
            "features": [], "footer": {"description": "", "contact": "", "phone": "", "legalText": "", "links": []}
        })
        mock_create.return_value = mock_resp
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Test", "projectIdea": "Idea",
            "callToAction": "CTA", "sections": ""
        })
        assert r.status_code == 200

    def test_get_json_structure_ignores_unknown_sections(self):
        result = json.loads(get_json_structure("PREMIUM", "hero,features,unknown_section,footer"))
        assert "unknown_section" not in result

    @patch("main.client.chat.completions.create")
    def test_unicode_project_name_handled(self, mock_create):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({
            "hero": {"badge": "", "headline": "", "subheadline": "", "ctaButton": "", "secondaryCta": "", "supportingText": ""},
            "features": [], "footer": {"description": "", "contact": "", "phone": "", "legalText": "", "links": []}
        })
        mock_create.return_value = mock_resp
        r = client.post("/api/v1/ai/generate", json={
            "projectId": 1, "userPlan": "BASIC",
            "projectName": "Café Ñoño & Co. 🇨🇱",
            "projectIdea": "Cafetería con detalles especiales",
            "callToAction": "¡Visítanos!"
        })
        assert r.status_code == 200

    def test_build_prompt_with_all_maps_missing_falls_back(self):
        data = ProjectData(
            projectId=50,
            userPlan="PREMIUM",
            projectName="FallbackTest",
            projectIdea="Idea",
            callToAction="CTA",
            communicationTone="desconocido",
            formalityLevel="desconocido",
            typographyStyle="desconocido",
            buttonStyle="desconocido",
            animationLevel="desconocido",
            creativityLevel="desconocido",
            layoutType="desconocido",
        )
        # No debe lanzar excepción; los valores desconocidos usan defaults vacíos
        prompt = build_prompt(data, "PREMIUM")
        assert "FallbackTest" in prompt