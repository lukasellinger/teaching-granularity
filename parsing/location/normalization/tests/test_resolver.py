import pytest
from parsing.location.normalization.resolver import (
    AuditableWikidataNormalizer, is_generic_place, contextual_component,
    explicit_component_level, is_place_expression, source_alias_targets,
    source_component_hint, _cache_identity,
)
from parsing.location.normalization.values import LocationLevel


class FakeNormalizer(AuditableWikidataNormalizer):
    def __init__(self, candidates, ancestors, metadata, context_candidates=None):
        self.candidates = candidates
        self.ancestors = ancestors
        self.metadata = metadata
        self.context_candidates = context_candidates or {}

    def _search(self, component):
        return self.candidates.get(component, [])

    def _context_search(self, component, context):
        return self.context_candidates.get((component, context), [])

    def _search_place_candidates(self, component):
        return self.candidates.get(component, [])

    def _ancestor_qids(self, qid):
        return self.ancestors.get(qid, set())

    def _metadata(self, qid):
        return self.metadata.get(
            qid, {"city": None, "region": None, "country": None, "continent": None}
        )

def candidate(qid, label, level, historical=False):
    return {"qid": qid, "label": label, "level": level, "historical": historical}

def test_explicit_overseas_territory_preserves_physical_continent_not_sovereign_continent():
    normalizer = FakeNormalizer(
        {"marquesas islands": [candidate("QM", "Marquesas Islands", LocationLevel.REGION)],
         "french polynesia": [candidate("QFP", "French Polynesia", LocationLevel.COUNTRY)]},
        {"QM": {"QFP", "QFR"}, "QFP": {"QFR"}},
        {"QM": {"region": ["Marquesas Islands"], "country": "France", "continent": "Europe"},
         "QFP": {"country": "France", "continent": "Insular Oceania"}},
    )
    result = normalizer.resolve(["Marquesas Islands", "French Polynesia"])
    assert result.value.country == "France"
    assert result.value.continent == "Oceania"
    assert result.consensus_value.continent == "Oceania"
    assert normalizer.metadata["QFP"]["continent"] == "Insular Oceania"

def test_explicit_sovereign_country_does_not_override_head_physical_continent():
    normalizer = FakeNormalizer(
        {"marquesas islands": [candidate("QM", "Marquesas Islands", LocationLevel.REGION)],
         "france": [candidate("QFR", "France", LocationLevel.COUNTRY)]},
        {"QM": {"QFR"}},
        {"QM": {"region": ["Marquesas Islands"], "country": "France", "continent": "Oceania"},
         "QFR": {"country": "France", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Marquesas Islands", "France"])
    assert result.value.continent == "Oceania"

def test_constituent_country_continent_repair_does_not_change_country_naming():
    normalizer = FakeNormalizer(
        {"leiden": [candidate("QL", "Leiden", LocationLevel.CITY)],
         "netherlands": [candidate("Q55", "Netherlands", LocationLevel.COUNTRY)]},
        {"QL": {"Q55", "Q29999"}, "Q55": {"Q29999"}},
        {"QL": {"city": "Leiden", "country": "Netherlands", "continent": "Europe"},
         "Q55": {"country": "Kingdom of the Netherlands", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Leiden", "Netherlands"])
    assert result.value.country == "Netherlands"

def test_frankfurt_city_alias_beats_cemetery_search_match():
    city = candidate("Q1794", "Frankfurt", LocationLevel.CITY)
    city["matched_name"] = "Frankfurt am Main, Germany"
    cemetery = candidate("Q881481", "Frankfurt Main Cemetery", LocationLevel.PLACE)
    cemetery["matched_name"] = "Frankfurt am Main"
    normalizer = FakeNormalizer(
        {"frankfurt am main": [city, cemetery],
         "germany": [candidate("Q183", "Germany", LocationLevel.COUNTRY)]},
        {"Q1794": {"Q183"}, "Q881481": {"Q1794", "Q183"}},
        {"Q1794": {"city": "Frankfurt", "region": ["Hesse"],
                   "country": "Germany", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Frankfurt am Main", "Germany"])
    assert result.head_qid == "Q1794"
    assert result.value.level is LocationLevel.CITY
    assert result.value.place is None

def test_station_ancestry_disambiguates_new_york_as_state():
    normalizer = FakeNormalizer(
        {"montauk": [candidate("Q4044493", "Montauk", LocationLevel.PLACE)],
         "new york": [candidate("Q1507987", "New York", LocationLevel.CITY),
                      candidate("Q1384", "New York", LocationLevel.REGION)]},
        {"Q4044493": {"Q1384", "Q30"}, "Q1384": {"Q30"},
         "Q1507987": set()},
        {"Q1384": {"region": ["New York"], "country": "United States",
                   "continent": "North America"},
         "Q4044493": {"region": ["New York"], "country": "United States",
                       "continent": "North America"}},
    )
    result = normalizer.resolve(["Montauk", "New York"])
    assert result.head_qid == "Q1384"
    assert result.value.city != "New York"
    assert "New York" in result.value.region

@pytest.mark.parametrize("component,text,level,historical", [
    ("Anjou", "He was born in the region of **Anjou**.", LocationLevel.REGION, False),
    ("Dalmatia", "Salona was in the Roman province of Dalmatia.", LocationLevel.REGION, True),
    ("York", "York, a city in England.", LocationLevel.CITY, False),
    ("Anjou", "Anjou, a region. Later Anjou, a village.", None, False),
    ("Paris", "A region of France. Later he lived in Paris.", None, False),
])
def test_direct_source_type_hint_not_remote_or_conflicting(component, text, level, historical):
    hint = source_component_hint(component, text)
    assert hint.get("level") == level
    if level is not None:
        assert hint["historical"] == historical
        assert hint["evidence"]

def test_source_type_does_not_leak_from_compound_name_to_indonesia():
    answer = "He was born in Kemusuk, Yogyakarta Special Region of Indonesia."
    assert source_component_hint("Indonesia", answer) == {}

def test_explicit_alias_relation_maps_golden_house_to_domus_aurea():
    answer = 'He died at Domus Aurea (also known as the "Golden House") in Rome.'
    assert source_alias_targets(["Domus Aurea", "Golden House", "Rome"], answer) == {1: 0}

def test_anjou_source_region_hint_rejects_context_compatible_commune():
    normalizer = FakeNormalizer(
        {"anjou": [candidate("QCOMMUNE", "Anjou", LocationLevel.CITY),
                   candidate("QREGION", "Anjou", LocationLevel.REGION)],
         "france": [candidate("QFR", "France", LocationLevel.COUNTRY)]},
        {"QCOMMUNE": {"QFR"}, "QREGION": {"QFR"}},
        {"QREGION": {"country": "France", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Anjou", "France"], answer="Born in the region of Anjou, France.")
    assert result.head_qid == "QREGION"
    assert result.value.level is LocationLevel.REGION
    assert result.traces[0].source_hint["level"] is LocationLevel.REGION
    assert [row.qid for row in result.traces[0].candidates] == ["QREGION"]

def test_extracted_indonesia_outranks_remote_region_word():
    normalizer = FakeNormalizer(
        {
            "kemusuk": [candidate("QK", "Kemusuk", LocationLevel.CITY)],
            "yogyakarta special region": [
                candidate("QY", "Yogyakarta Special Region", LocationLevel.REGION)
            ],
            "indonesia": [
                candidate("Q252", "Indonesia", LocationLevel.COUNTRY),
                candidate("Q49332", "Nusantara", LocationLevel.REGION),
            ],
        },
        {"QK": {"QY", "Q252"}, "QY": {"Q252"}},
        {
            "QK": {"city": "Kemusuk", "region": ["Special Region of Yogyakarta"],
                   "country": "Indonesia", "continent": "Asia"},
            "QY": {"region": ["Special Region of Yogyakarta"],
                   "country": "Indonesia", "continent": "Asia"},
            "Q252": {"country": "Indonesia", "continent": "Asia"},
        },
    )
    result = normalizer.resolve(
        ["Kemusuk", "Yogyakarta Special Region", "Indonesia"],
        answer="He was born in Kemusuk, Yogyakarta Special Region of Indonesia.",
    )
    assert result.status == "normalized"
    assert result.value.city == "Kemusuk"
    assert result.value.country == "Indonesia"
    assert result.consensus_value == result.value
    assert result.traces[2].selected.qid == "Q252"

def test_explicit_source_alias_reuses_target_entity_not_unrelated_homonym():
    normalizer = FakeNormalizer(
        {
            "domus aurea": [candidate("QDOM", "Domus Aurea", LocationLevel.PLACE)],
            "golden house": [candidate("QHOTEL", "Golden House", LocationLevel.PLACE)],
            "rome": [candidate("QROME", "Rome", LocationLevel.CITY)],
        },
        {"QDOM": {"QROME", "QIT"}, "QHOTEL": {"QCAM"}, "QROME": {"QIT"}},
        {
            "QDOM": {"city": "Rome", "country": "Italy", "continent": "Europe"},
            "QROME": {"city": "Rome", "country": "Italy", "continent": "Europe"},
        },
    )
    result = normalizer.resolve(
        ["Domus Aurea", "Golden House", "Rome"],
        answer='He died at Domus Aurea (also known as the "Golden House") in Rome.',
    )
    assert result.status == "normalized"
    assert result.consensus_value == result.value
    assert result.traces[1].selected.qid == "QDOM"
    assert result.traces[1].source_hint["alias_of"] == "Domus Aurea"

def test_explicit_source_alias_keeps_alias_candidate_when_old_name_is_unresolved():
    normalizer = FakeNormalizer(
        {
            "hurley medical center": [
                candidate("QHURLEY", "Hurley Medical Center", LocationLevel.PLACE)
            ],
            "detroit": [candidate("QDET", "Detroit", LocationLevel.CITY)],
            "michigan": [candidate("QMI", "Michigan", LocationLevel.REGION)],
        },
        {
            "QHURLEY": {"QFLINT", "QMI", "QUS"},
            "QDET": {"QMI", "QUS"},
            "QMI": {"QUS"},
        },
        {
            "QHURLEY": {"place": "Hurley Medical Center", "city": "Flint",
                        "region": ["Michigan"], "country": "United States",
                        "continent": "North America"},
            "QDET": {"city": "Detroit", "region": ["Michigan"],
                     "country": "United States", "continent": "North America"},
            "QMI": {"region": ["Michigan"], "country": "United States",
                    "continent": "North America"},
        },
    )
    result = normalizer.resolve(
        ["caveats Hospital", "Hurley Medical Center", "Detroit", "Michigan"],
        answer=("She died at caveats Hospital (now known as Hurley Medical Center) "
                "in Detroit, Michigan."),
    )
    assert result.status == "normalized"
    assert result.traces[1].selected is None
    assert result.traces[1].candidate_qids == ["QHURLEY"]
    assert result.traces[1].place_case == "explicit"
    assert "incompatible Wikidata candidates rejected" in result.traces[1].note
    assert result.consensus_value == result.value

def test_unresolved_component_does_not_flatten_single_resolved_region_hierarchy():
    normalizer = FakeNormalizer(
        {
            "fuyang province": [candidate("QFY", "Fuyang", LocationLevel.REGION)],
            "anhui province": [candidate("QAH", "Anhui", LocationLevel.REGION)],
            "china": [candidate("QCN", "People's Republic of China", LocationLevel.COUNTRY)],
        },
        {"QFY": {"QAH", "QCN"}, "QAH": {"QCN"}},
        {
            "QFY": {"region": ["Fuyang", "Anhui", "China"],
                    "country": "People's Republic of China", "continent": "Asia"},
            "QAH": {"region": ["Anhui", "China"],
                    "country": "People's Republic of China", "continent": "Asia"},
            "QCN": {"country": "People's Republic of China", "continent": "Asia"},
        },
    )
    result = normalizer.resolve(
        ["Jinghui County", "Fuyang Province", "Anhui Province", "China"]
    )
    assert result.status == "partial"
    assert result.consensus_value.region == ["Fuyang", "Anhui", "China"]
    assert result.consensus_value.country == "People's Republic of China"

def test_historical_component_ignores_incompatible_modern_homonym():
    normalizer = FakeNormalizer(
        {
            "kaliningrad oblast": [candidate("QK", "Kaliningrad Oblast", LocationLevel.REGION)],
            "prussia": [
                candidate("QH", "Prussia", LocationLevel.REGION, historical=True),
                candidate("QITALY", "Prussia", LocationLevel.CITY),
            ],
            "russia": [candidate("QR", "Russia", LocationLevel.COUNTRY)],
        },
        {"QK": {"QR"}, "QITALY": {"QIT"}},
        {
            "QK": {"region": ["Kaliningrad Oblast"], "country": "Russia",
                   "continent": "Europe"},
            "QR": {"country": "Russia", "continent": "Europe"},
            "QITALY": {"city": "Prussia", "country": "Italy", "continent": "Europe"},
        },
    )
    result = normalizer.resolve(
        ["Krasnoe Podgory", "Kaliningrad Oblast", "Eastern Prussia", "Russia"]
    )
    assert result.status == "partial"
    assert result.value.country == "Russia"
    assert result.consensus_value.country == "Russia"
    assert result.consensus_value.region == ["Kaliningrad Oblast"]
    assert result.traces[2].status == "omitted"
    assert "incompatible modern homonyms" in result.traces[2].note
    assert result.traces[2].consensus_candidate_qids == []

@pytest.mark.parametrize("description,level", [
    ("valley in Central Asia", LocationLevel.REGION),
    ("low mountain range in northern Germany", LocationLevel.REGION),
    ("peninsula in Western Asia", LocationLevel.REGION),
    ("subdivision of Mönchengladbach, Germany", LocationLevel.SUBCITY),
    ("district of Wilhelmshaven, Germany", LocationLevel.SUBCITY),
    ("hamlet in Yogyakarta, Indonesia", LocationLevel.CITY),
    ("exclave of Russia bounded by Poland and Lithuania", LocationLevel.REGION),
    ("oblast and federal subject of Russia", LocationLevel.REGION),
    ("Yvelines departmental fire and rescue service", None),
    ("archival records of Rheydt in a state archive", None),
])
def test_audited_geographic_and_institution_types(description, level):
    assert AuditableWikidataNormalizer._level_from_description(description) is level

@pytest.mark.parametrize("description", [
    "kingdom in Southeastern Europe between 1881 and 1947",
    "Roman province in the Balkans",
    "former federation of East Slavic tribes",
])
def test_historical_kingdom_and_province_descriptions(description):
    assert AuditableWikidataNormalizer._historical_description(description)

def test_dated_historical_polity_label_recovers_terse_wikidata_description():
    assert AuditableWikidataNormalizer._historical_candidate(
        "Kingdom of Yugoslavia",
        "official name of Yugoslavia during the period of 1929–1941 (de facto)/1945",
    )
    assert not AuditableWikidataNormalizer._historical_candidate(
        "Kingdom of the Netherlands",
        "sovereign state and constitutional monarchy in Western Europe",
    )

@pytest.mark.parametrize("label,description", [
    ("Habsburg monarchy", "monarchy in Europe (1282–1918)"),
    ("Kievan Rus'", "medieval state in Eastern and Northern Europe"),
])
def test_dated_or_medieval_polities_are_historical(label, description):
    assert AuditableWikidataNormalizer._historical_candidate(label, description)

@pytest.mark.parametrize("description", [
    "thoroughfare in Naples, Italy",
    "court prison in Nuremberg, Germany",
])
def test_streets_and_prisons_are_places(description):
    assert AuditableWikidataNormalizer._level_from_description(description) is LocationLevel.PLACE

def test_generic_place_does_not_turn_following_city_into_a_venue():
    normalizer = FakeNormalizer(
        {
            "saint-cloud": [
                candidate("QCITY", "Saint-Cloud", LocationLevel.CITY),
                candidate("QPLACE", "Saint-Cloud", LocationLevel.PLACE),
            ],
        },
        {"QCITY": {"QFR"}, "QPLACE": {"QCITY", "QFR"}},
        {
            "QCITY": {"city": "Saint-Cloud", "region": ["Île-de-France"],
                      "country": "France", "continent": "Europe"},
            "QPLACE": {"place": "Saint-Cloud", "city": "Saint-Cloud",
                       "country": "France", "continent": "Europe"},
        },
    )
    result = normalizer.resolve(
        ["his home", "Saint-Cloud"],
        answer="He died at his home in Saint-Cloud, France.",
    )
    assert result.value.level is LocationLevel.PLACE
    assert result.value.place == ["his home"]
    assert result.value.city == "Saint-Cloud"
    assert result.traces[1].selected.level is LocationLevel.CITY

def test_context_corroboration_survives_duplicate_independent_search_hit():
    normalizer = FakeNormalizer(
        {
            "fengyang": [
                candidate("QWRONG", "Fengyang", LocationLevel.CITY),
                candidate("QRIGHT", "Fengyang County", LocationLevel.REGION),
            ],
            "anhui province": [candidate("QANHUI", "Anhui", LocationLevel.REGION)],
            "china": [candidate("QCHINA", "China", LocationLevel.COUNTRY)],
        },
        {
            "QWRONG": {"QCHINA"},
            "QRIGHT": {"QANHUI", "QCHINA"},
            "QANHUI": {"QCHINA"},
        },
        {
            "QWRONG": {"city": "Fengyang", "region": ["Guangxi"],
                       "country": "China", "continent": "Asia"},
            "QRIGHT": {"region": ["Fengyang County", "Anhui"],
                       "country": "China", "continent": "Asia"},
            "QANHUI": {"region": ["Anhui"], "country": "China",
                       "continent": "Asia"},
            "QCHINA": {"country": "China", "continent": "Asia"},
        },
        {
            ("Fengyang", "Anhui Province"): [
                candidate("QRIGHT", "Fengyang County", LocationLevel.REGION)
            ],
        },
    )

    result = normalizer.resolve(
        ["Fengyang", "Anhui Province", "China"],
        answer="He was born in Fengyang, Anhui Province, China.",
    )

    assert result.head_qid == "QRIGHT"
    assert set(result.value.region) == {"Anhui", "Fengyang County"}
    assert result.value.country == "China"
    assert result.traces[0].selected.qid == "QRIGHT"

def test_cell_and_prison_are_place_evidence():
    assert is_generic_place("his cell")
    assert is_generic_place("the town's center")
    assert is_place_expression("Nuremberg Prison")

def test_historical_only_component_becomes_partial_operational_fallback():
    normalizer = FakeNormalizer(
        {"seleucid empire": [
            candidate("QSE", "Seleucid Empire", LocationLevel.REGION, historical=True)
        ]},
        {},
        {},
    )
    result = normalizer.resolve(["Seleucid Empire", "royal administrative centers"])
    assert result.status == "partial"
    assert result.value.level is LocationLevel.REGION
    assert result.value.region == ["Seleucid Empire"]
    assert result.consensus_value == result.value
    assert result.head_qid == "QSE"
    assert result.traces[0].status == "selected"
    assert result.traces[0].selected.historical is True
    assert result.traces[0].candidate_qids == ["QSE"]
    assert result.traces[1].status == "unresolved"

def test_historical_polity_context_does_not_make_modern_hierarchy_partial():
    normalizer = FakeNormalizer(
        {
            "požarevac": [candidate("QPOZ", "Požarevac", LocationLevel.CITY)],
            "serbia": [candidate("QRS", "Serbia", LocationLevel.COUNTRY)],
            "kingdom of yugoslavia": [candidate(
                "QYUG", "Kingdom of Yugoslavia", LocationLevel.COUNTRY, historical=True
            )],
        },
        {"QPOZ": {"QRS"}, "QRS": set()},
        {
            "QPOZ": {"city": "Požarevac", "country": "Serbia", "continent": "Europe"},
            "QRS": {"country": "Serbia", "continent": "Europe"},
        },
    )
    result = normalizer.resolve(["Požarevac", "Serbia", "Kingdom of Yugoslavia"])
    assert result.status == "normalized"
    assert result.value.city == "Požarevac"
    assert result.value.country == "Serbia"
    assert result.traces[2].status == "omitted"
    assert result.traces[2].candidate_qids == ["QYUG"]

def test_rheydt_exact_subcity_cannot_become_station_to_fit_context():
    normalizer = FakeNormalizer(
        {"rheydt": [candidate("QSTATION", "Rheydt Central Station", LocationLevel.PLACE),
                    candidate("Q702411", "Rheydt", LocationLevel.SUBCITY)],
         "germany": [candidate("QDE", "Germany", LocationLevel.COUNTRY)],
         "düsseldorf": [candidate("QGOV", "Düsseldorf Government Region", LocationLevel.REGION),
                        candidate("QCITY", "Düsseldorf", LocationLevel.CITY)]},
        {"Q702411": {"QDE"}, "QSTATION": {"QGOV", "QDE"}, "QCITY": {"QGOV", "QDE"}},
        {"Q702411": {"country": "Germany", "continent": "Europe"},
         "QCITY": {"country": "Germany", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Rheydt", "Germany", "Düsseldorf"])
    assert result.value.level is not LocationLevel.PLACE
    assert "QSTATION" not in result.traces[0].candidate_qids
    assert result.traces[2].candidate_qids == ["QCITY"]
    assert result.status == "partial"
    assert result.consensus_value.level is LocationLevel.COUNTRY

def test_distance_phrase_is_context_not_address_and_west_hollywood_is_name():
    assert not is_place_expression("about 25 kilometers north of Paris")
    assert contextual_component("about 25 kilometers north of Paris") == ("near", "Paris")
    assert is_place_expression("10 Market Street")
    normalizer = FakeNormalizer(
        {"west hollywood": [candidate("QWH", "West Hollywood", LocationLevel.CITY)],
         "france": [candidate("QFR", "France", LocationLevel.COUNTRY)]}, {}, {},
    )
    assert normalizer.resolve(["West Hollywood"]).head_qid == "QWH"
    assert normalizer.resolve(["western France"]).value.country == "France"
    near = normalizer.resolve(["near France"])
    assert near.value is None

def test_compound_directional_area_supplies_broad_context_without_becoming_head():
    normalizer = FakeNormalizer(
        {
            "bonaventura": [candidate("QBON", "Bonaventura", LocationLevel.PLACE)],
            "brest": [candidate("QBREST", "Brest", LocationLevel.CITY)],
            "france": [candidate("QFR", "France", LocationLevel.COUNTRY)],
        },
        {"QBON": {"QFR"}, "QBREST": {"QFR"}, "QFR": set()},
        {
            "QBON": {"place": "Bonaventura", "country": "France",
                     "continent": "Europe"},
            "QBREST": {"city": "Brest", "country": "France", "continent": "Europe"},
            "QFR": {"country": "France", "continent": "Europe"},
        },
    )
    result = normalizer.resolve(["Bonaventura", "near Brest", "northwest France"])
    assert result.status == "normalized"
    assert result.value.level is LocationLevel.PLACE
    assert result.value.country == "France"
    assert result.traces[1].status == "context"
    assert result.traces[1].selected.qid == "QBREST"
    assert result.traces[2].selected.qid == "QFR"

def test_literal_department_does_not_retype_a_service_as_region():
    normalizer = FakeNormalizer(
        {"yvelines department": [candidate("QSERVICE", "Yvelines departmental fire and rescue service", LocationLevel.PLACE)],
         "france": [candidate("QFR", "France", LocationLevel.COUNTRY)]}, {}, {},
    )
    result = normalizer.resolve(["Yvelines department", "France"])
    assert result.traces[0].status == "unresolved"
    assert result.value.level is LocationLevel.COUNTRY

def test_roman_province_source_hint_does_not_select_modern_region():
    normalizer = FakeNormalizer(
        {"salona": [candidate("QS", "Salona", LocationLevel.CITY)],
         "dalmatia": [candidate("QM", "Dalmatia", LocationLevel.REGION),
                      candidate("QH", "Dalmatia", LocationLevel.REGION, historical=True)]},
        {}, {"QS": {"country": "Croatia", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Salona", "Dalmatia"],
                                answer="Born in Salona in the Roman province of Dalmatia.")
    assert result.status == "normalized"
    assert result.consensus_value.city == "Salona"
    assert result.traces[1].candidate_qids == ["QH"]
    assert result.traces[1].candidates[0].historical

def test_cache_distinguishes_source_type_hints():
    components = ["Anjou"]
    assert _cache_identity(components, answer="Born in the region of Anjou.") != _cache_identity(
        components, answer="Born in the commune of Anjou.")

def test_resolved_natural_region_conflict_coarsens_consensus():
    normalizer = FakeNormalizer(
        {"breselenz": [candidate("QB", "Breselenz", LocationLevel.CITY)],
         "harz mountains": [candidate("Q4186", "Harz", LocationLevel.REGION)],
         "germany": [candidate("QDE", "Germany", LocationLevel.COUNTRY)]},
        {"QB": {"QDE"}, "Q4186": {"QDE"}},
        {"QB": {"country": "Germany", "continent": "Europe"},
         "Q4186": {"country": "Germany", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Breselenz", "Harz Mountains", "Germany"])
    assert result.status == "partial"
    assert result.consensus_value.level is LocationLevel.COUNTRY
    assert result.traces[1].consensus_candidate_qids == ["Q4186"]

@pytest.mark.parametrize("supported,country", [
    (["Uzbekistan", "Tajikistan", "Kyrgyzstan"], "Uzbekistan"),
    (["Tajikistan", "Kyrgyzstan"], None),
])
def test_transnational_valley_uses_only_supported_explicit_country(supported, country):
    normalizer = FakeNormalizer(
        {"andijan": [candidate("QA", "Andijan", LocationLevel.CITY)],
         "fergana valley": [candidate("QV", "Fergana Valley", LocationLevel.REGION)],
         "uzbekistan": [candidate("QU", "Uzbekistan", LocationLevel.COUNTRY)]},
        {"QA": {"QU"}},
        {"QA": {"country": "Uzbekistan", "continent": "Asia"},
         "QV": {"country": None, "country_candidates": supported, "continent": "Asia"},
         "QU": {"continent": "Asia"}},
    )
    result = normalizer.resolve(["Andijan", "Fergana Valley", "Uzbekistan"])
    assert result.status == "partial"  # country overlap is NOT proof of city containment
    assert result.consensus_value.country == country
    if country:
        assert result.consensus_value.level is LocationLevel.COUNTRY
        assert "explicit answer context" in result.traces[1].candidate_hierarchies[0].note

def test_physical_feature_containment_query_uses_p706_not_proximity():
    calls = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"results": {"bindings": [{"ancestor": {"value": "http://www.wikidata.org/entity/QVALLEY"}}]}}
    class Session:
        def get(self, url, **kwargs):
            calls.append(kwargs["params"]["query"])
            return Response()
    normalizer = AuditableWikidataNormalizer()
    normalizer._session = lambda: Session()
    assert normalizer._ancestor_qids("Q1") == {"QVALLEY"}
    assert "wdt:P706" in calls[0]
    assert "wdt:P17" not in calls[0]  # sharing a country does not imply containment

def test_department_search_retries_geographic_name():
    searches = []
    class Response:
        def __init__(self, rows):
            self.rows = rows
        def raise_for_status(self):
            pass
        def json(self):
            return {"search": self.rows}
    class Session:
        def get(self, url, **kwargs):
            query = kwargs["params"]["search"]
            searches.append(query)
            if query == "yvelines department":
                return Response([{"id": "QSERVICE", "label": "Yvelines departmental fire and rescue service",
                                  "description": "departmental fire and rescue service in France"}])
            return Response([{"id": "Q12820", "label": "Yvelines", "description": "department in France"}])
    normalizer = AuditableWikidataNormalizer()
    normalizer._session = lambda: Session()
    result = normalizer._search("yvelines department")
    assert searches == ["yvelines department", "yvelines"]
    assert result[0]["qid"] == "Q12820"

def test_historical_context_does_not_leak_into_completed_present_day_country():
    metadata = AuditableWikidataNormalizer._metadata_from_bindings([
        {"adminLabel": {"value": "Cluj-Napoca"},
         "adminDescription": {"value": "city in Romania"},
         "directCountryLabel": {"value": "Romania"},
         "directCountryDescription": {"value": "country in Europe"}},
        {"adminLabel": {"value": "Kingdom of Romania"},
         "adminDescription": {"value": "kingdom in Southeastern Europe between 1881 and 1947"},
         "countryLabel": {"value": "Kingdom of Romania"},
         "countryDescription": {"value": "kingdom in Southeastern Europe between 1881 and 1947"}},
    ])
    assert metadata["country"] == "Romania"
    assert metadata["region"] is None

@pytest.mark.parametrize("components", [["Victoria", "Hong Kong"], ["Hong Kong", "Victoria"]])
def test_dual_city_region_is_region_when_containing_a_selected_city(components):
    hong_kong = candidate("QHK", "Hong Kong", LocationLevel.CITY)
    hong_kong["administrative_region_role"] = True
    normalizer = FakeNormalizer(
        {"victoria": [candidate("QV", "Victoria", LocationLevel.CITY)],
         "hong kong": [hong_kong]},
        {"QV": {"QHK", "QCN"}, "QHK": {"QCN"}},
        {"QV": {"country": "China", "continent": "Asia"},
         "QHK": {"country": "China", "continent": "Asia"}},
    )
    result = normalizer.resolve(components)
    assert result.value.city == "Victoria"
    assert result.value.region == ["Hong Kong"]
    assert result.value.explicit_levels == [LocationLevel.CITY, LocationLevel.REGION]
    assert result.consensus_value == result.value
    trace = next(trace for trace in result.traces if trace.component == "Hong Kong")
    assert trace.selected.level is LocationLevel.REGION
    assert trace.candidates[0].level is LocationLevel.CITY
    assert trace.candidates[0].administrative_region_role
    assert "containing a selected city" in trace.note

@pytest.mark.parametrize("components", [["Hong Kong"], ["Hong Kong", "China"]])
def test_dual_city_region_standalone_remains_city(components):
    hong_kong = candidate("QHK", "Hong Kong", LocationLevel.CITY)
    hong_kong["administrative_region_role"] = True
    normalizer = FakeNormalizer(
        {"hong kong": [hong_kong], "china": [candidate("QCN", "China", LocationLevel.COUNTRY)]},
        {"QHK": {"QCN"}},
        {"QHK": {"country": "China", "continent": "Asia"}},
    )
    result = normalizer.resolve(components)
    assert result.value.level is LocationLevel.CITY
    assert result.value.city == "Hong Kong"
    assert result.value.region is None
    assert result.value.country == "China"

def test_ordinary_city_container_is_not_reclassified_as_region():
    normalizer = FakeNormalizer(
        {"manhattan": [candidate("QM", "Manhattan", LocationLevel.SUBCITY)],
         "new york city": [candidate("QNY", "New York City", LocationLevel.CITY)]},
        {"QM": {"QNY"}},
        {"QM": {"city": "New York City", "country": "United States"}},
    )
    result = normalizer.resolve(["Manhattan", "New York City"])
    assert result.value.city == "New York City"
    assert result.value.region is None
    assert result.traces[1].selected.level is LocationLevel.CITY

def test_ordinary_city_inside_city_does_not_create_region_role():
    normalizer = FakeNormalizer(
        {"child": [candidate("QC", "Child", LocationLevel.CITY)],
         "parent": [candidate("QP", "Parent", LocationLevel.CITY)]},
        {"QC": {"QP"}}, {},
    )
    result = normalizer.resolve(["Child", "Parent"])
    assert result.value.region is None
    assert result.traces[1].selected.level is LocationLevel.CITY

def test_administrative_role_is_entity_type_not_surrounding_context():
    detect = AuditableWikidataNormalizer._has_administrative_region_role
    assert detect("city and special administrative region of China")
    assert detect("city and federal state of Austria")
    assert not detect("city in a special administrative region of China")
    assert not detect("city in China")

def test_hierarchy_completion_uses_dual_region_only_above_child_city():
    hong_kong = {"adminDescription": {"value": "city and special administrative region of China"},
                 "adminLabel": {"value": "Hong Kong"},
                 "countryLabel": {"value": "China"}}
    victoria = {"adminDescription": {"value": "historical city in Hong Kong"},
                "adminLabel": {"value": "Victoria"}}
    alone = AuditableWikidataNormalizer._metadata_from_bindings([hong_kong])
    contained = AuditableWikidataNormalizer._metadata_from_bindings([victoria, hong_kong])
    assert alone["city"] == "Hong Kong"
    assert alone["region"] is None
    assert contained["city"] == "Victoria"
    assert contained["region"] == ["Hong Kong"]

@pytest.mark.parametrize("first,first_country,second,second_country,continent", [
    ("London", "United Kingdom", "Paris", "France", "Europe"),
    ("Madrid", "Spain", "Lisbon", "Portugal", "Europe"),
])
def test_alternative_branch_uses_its_own_country_not_all_name_matches(
    first, first_country, second, second_country, continent,
):
    normalizer = FakeNormalizer(
        {first.lower(): [candidate("QA", first, LocationLevel.CITY)],
         first_country.lower(): [candidate("QCA", first_country, LocationLevel.COUNTRY)],
         second.lower(): [candidate("QUSCITY", second, LocationLevel.CITY),
                          candidate("QB", second, LocationLevel.CITY)],
         second_country.lower(): [candidate("QCB", second_country, LocationLevel.COUNTRY)]},
        {"QA": {"QCA"}, "QB": {"QCB"}, "QUSCITY": {"QUS"}},
        {"QA": {"country": first_country, "continent": continent},
         "QB": {"country": second_country, "continent": continent},
         "QCA": {"continent": continent}, "QCB": {"continent": continent},
         "QUSCITY": {"country": "United States", "continent": "North America"}},
    )
    result = normalizer.resolve([first, first_country, second, second_country])
    assert result.status == "partial"
    assert result.consensus_value.level is LocationLevel.CONTINENT
    assert result.consensus_value.continent == continent
    assert result.traces[2].consensus_candidate_qids == ["QB"]

def test_conflicting_country_uses_sensible_search_rank_not_homonymous_town():
    normalizer = FakeNormalizer(
        {"panama city": [candidate("QP", "Panama City", LocationLevel.CITY)],
         "costa rica": [candidate("QCR", "Costa Rica", LocationLevel.COUNTRY),
                       candidate("QTOWN", "Costa Rica", LocationLevel.CITY)]},
        {"QP": {"QPAN"}, "QCR": set(), "QTOWN": {"QBR"}},
        {"QP": {"country": "Panama", "continent": "North America"},
         "QCR": {"continent": "North America"},
         "QTOWN": {"country": "Brazil", "continent": "South America"}},
    )
    result = normalizer.resolve(["Panama City", "Costa Rica"])
    assert result.consensus_value.continent == "North America"
    assert result.traces[1].consensus_candidate_qids == ["QCR"]

def test_separate_country_search_cannot_replace_selected_city_country():
    normalizer = FakeNormalizer(
        {"leiden": [candidate("QL", "Leiden", LocationLevel.CITY)],
         "netherlands": [candidate("QK", "Kingdom of the Netherlands", LocationLevel.COUNTRY)]},
        {"QL": {"QNL", "QK"}},
        {"QL": {"country": "Netherlands", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Leiden", "Netherlands"])
    assert result.value.country == "Netherlands"
    assert result.traces[1].selected.qid == "QK"

def test_modern_container_repairs_historical_head_completion():
    normalizer = FakeNormalizer(
        {"victoria": [candidate("QV", "Victoria", LocationLevel.CITY)],
         "hong kong": [candidate("QHK", "Hong Kong", LocationLevel.CITY)]},
        {"QV": {"QHK", "QOLD"}, "QHK": {"QCN"}},
        {"QV": {"country": "British Hong Kong", "continent": "Asia"},
         "QHK": {"country": "China", "continent": "Asia"}},
    )
    result = normalizer.resolve(["Victoria", "Hong Kong"])
    assert result.value.country == "China"

def test_literal_country_in_city_chain_beats_search_ranked_country_alias():
    kingdom = candidate("QK", "Kingdom of the Netherlands", LocationLevel.COUNTRY)
    kingdom["matched_name"] = "Netherlands"
    normalizer = FakeNormalizer(
        {"leiden": [candidate("QL", "Leiden", LocationLevel.CITY)],
         "netherlands": [kingdom, candidate("QNL", "Netherlands", LocationLevel.COUNTRY)]},
        {"QL": {"QNL", "QK"}},
        {"QL": {"country": "Netherlands", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Leiden", "Netherlands"])
    assert result.value.country == "Netherlands"
    assert result.traces[1].selected.qid == "QNL"

def test_autonomous_country_description_is_retained_as_country():
    assert AuditableWikidataNormalizer._level_from_description(
        "one of the four autonomous countries of the Kingdom of the Netherlands; with territories in the Caribbean"
    ) is LocationLevel.COUNTRY
    assert AuditableWikidataNormalizer._level_from_description(
        "nationality and autonomous community of Spain"
    ) is LocationLevel.REGION
    assert AuditableWikidataNormalizer._level_from_description(
        "The official association football team of the Spanish autonomous community of Catalonia"
    ) is None

def test_completed_metadata_excludes_historical_admin_and_country():
    result = AuditableWikidataNormalizer._metadata_from_bindings([
        {"adminLabel": {"value": "British Hong Kong"},
         "adminDescription": {"value": "former colony in East Asia"},
         "countryLabel": {"value": "British Hong Kong"},
         "countryDescription": {"value": "former colony in East Asia"},
         "directCountryLabel": {"value": "British Hong Kong"},
         "directCountryDescription": {"value": "former colony in East Asia"}},
        {"adminLabel": {"value": "Hong Kong"},
         "countryLabel": {"value": "China"}},
    ])
    assert result["country"] == "China"
    assert result["region"] is None

def test_venue_description_and_name_remain_explicit_place_evidence():
    normalizer = FakeNormalizer(
        {"pecoli homestead": [], "nyack": [candidate("Q1", "Nyack", LocationLevel.CITY)],
         "new york": [candidate("Q2", "New York", LocationLevel.REGION)]},
        {"Q1": {"Q2"}, "Q2": set()},
        {"Q1": {"region": ["New York"], "country": "United States", "continent": "North America"}},
    )
    result = normalizer.resolve(["his family's home", "Pecoli Homestead", "Nyack", "New York"])
    assert result.status == "normalized"
    assert result.value.level is LocationLevel.PLACE
    assert result.value.place == ["his family's home", "Pecoli Homestead"]
    assert result.value.city == "Nyack"
    assert [trace.status for trace in result.traces] == [
        "place_expression", "place_expression", "selected", "selected"
    ]
    assert [trace.place_case for trace in result.traces[:2]] == ["generic", "explicit"]

def test_bare_wikidata_place_is_verified_separately_from_geographic_head():
    normalizer = FakeNormalizer(
         {"st. vincent's": [
             candidate("QDUB", "St. Vincent's University Hospital", LocationLevel.PLACE),
             candidate("QBR", "St. Vincent's Medical Center", LocationLevel.PLACE),
             candidate("QSETTLE", "St. Vincent's Bay", LocationLevel.CITY),
             candidate("QH", "St. Vincent's Hospital", LocationLevel.PLACE),
         ],
         "manhattan": [candidate("QM", "Manhattan", LocationLevel.CITY)],
         "united states": [candidate("QUS", "United States", LocationLevel.COUNTRY)]},
        {"QDUB": {"QDUBLIN", "QIE"}, "QBR": {"QBRIDGEPORT", "QUS"},
         "QSETTLE": {"QCANADA"},
         "QH": {"QM", "QUS"}, "QM": {"QUS"}, "QUS": set()},
        {"QM": {"region": ["New York"], "country": "United States",
                 "continent": "North America"},
         "QH": {"region": ["New York"], "country": "United States",
                 "continent": "North America", "city": "Manhattan"},
         "QBR": {"region": ["Connecticut"], "country": "United States",
                  "continent": "North America", "city": "Bridgeport"},
         "QDUB": {"region": None, "country": "Ireland",
                   "continent": "Europe", "city": "Dublin"}},
    )
    result = normalizer.resolve(["St. Vincent's", "Manhattan", "United States"])
    assert result.status == "normalized"
    assert result.head_qid == "QM"
    assert result.value.place == ["St. Vincent's"]
    assert result.value.city == "Manhattan"
    assert result.traces[0].place_case == "wikidata_compatible"
    assert result.traces[0].selected.qid == "QH"
    assert result.traces[0].selected.role == "place"

def test_incompatible_wikidata_place_is_rejected_without_penalizing_written_hierarchy():
    normalizer = FakeNormalizer(
        {"st. vincent's": [candidate("QH", "St. Vincent's", LocationLevel.PLACE)],
         "berlin": [candidate("QB", "Berlin", LocationLevel.CITY)],
         "germany": [candidate("QDE", "Germany", LocationLevel.COUNTRY)]},
        {"QH": {"QUS"}, "QB": {"QDE"}, "QDE": set(), "QUS": set()},
        {"QB": {"region": None, "country": "Germany", "continent": "Europe"},
         "QH": {"region": ["New York"], "country": "United States",
                 "continent": "North America", "city": "Bridgeport"}},
    )
    result = normalizer.resolve(["St. Vincent's", "Berlin", "Germany"])
    assert result.status == "normalized"
    assert result.head_qid == "QB"
    assert result.value.place == ["St. Vincent's"]
    assert result.value.city == "Berlin"
    assert result.traces[0].selected is None
    assert result.traces[0].candidate_qids == ["QH"]
    assert result.traces[0].place_case == "explicit"
    assert "incompatible Wikidata candidates rejected" in result.traces[0].note
    assert result.consensus_value == result.value

def test_stone_bridge_homonym_does_not_make_lancaster_new_york_partial():
    normalizer = FakeNormalizer(
        {
            "stone bridge": [candidate("QRU", "Stone Bridge", LocationLevel.PLACE)],
            "lancaster": [candidate("QLAN", "Lancaster", LocationLevel.CITY)],
            "new york": [candidate("QNY", "New York", LocationLevel.REGION)],
        },
        {"QRU": {"QRUSSIA"}, "QLAN": {"QNY", "QUS"}, "QNY": {"QUS"}},
        {
            "QRU": {"region": ["Vologda Oblast"], "country": "Russia"},
            "QLAN": {"city": "Lancaster", "region": ["Erie County", "New York"],
                     "country": "United States", "continent": "North America"},
            "QNY": {"region": ["New York"], "country": "United States",
                    "continent": "North America"},
        },
    )
    result = normalizer.resolve(["Stone Bridge", "Lancaster", "New York"])
    assert result.status == "normalized"
    assert result.value.place == ["Stone Bridge"]
    assert result.value.city == "Lancaster"
    assert result.value.region == ["New York", "Erie County"]
    assert result.traces[0].selected is None
    assert result.traces[0].candidate_qids == ["QRU"]
    assert result.traces[0].place_case == "explicit"
    assert "incompatible Wikidata candidates rejected" in result.traces[0].note

def test_wikidata_place_without_geography_remains_a_place_only_value():
    normalizer = FakeNormalizer(
        {"st. vincent's": [candidate("QH", "St. Vincent's", LocationLevel.PLACE)]},
        {"QH": {"QM", "QUS"}},
        {"QH": {"region": ["New York"], "country": "United States",
                 "continent": "North America"}},
    )
    result = normalizer.resolve(["St. Vincent's"])
    assert result.status == "normalized"
    assert result.head_qid is None
    assert result.value.level is LocationLevel.PLACE
    assert result.value.place == ["St. Vincent's"]
    assert result.value.country == "United States"
    assert result.traces[0].place_case == "wikidata_recognized"

def test_source_order_is_not_assumed_to_be_fine_to_broad():
    normalizer = FakeNormalizer(
        {"valencia": [candidate("QREG", "Valencian Community", LocationLevel.REGION)],
         "spain": [candidate("QES", "Spain", LocationLevel.COUNTRY)],
         "alcazalegre": [],
         "province of valencia": [candidate("QPROV", "Province of Valencia", LocationLevel.REGION)],
         "autonomous community of valencia": [candidate("QREG", "Valencian Community", LocationLevel.REGION)]},
        {"QPROV": {"QREG", "QES"}, "QREG": {"QES"}, "QES": set()},
        {"QPROV": {"region": ["Valencian Community"], "country": "Spain", "continent": "Europe"}},
    )
    result = normalizer.resolve([
        "Valencia", "Spain", "Alcazalegre", "province of Valencia", "autonomous community of Valencia"
    ])
    assert result.status == "partial"
    assert result.value.level is LocationLevel.REGION
    assert result.value.country == "Spain"
    assert result.traces[2].status == "unresolved"
    assert result.head_qid == "QPROV"

def test_aliases_and_incompatible_context_are_visible_in_trace():
    normalizer = FakeNormalizer(
        {"lydda": [candidate("QLOD", "Lod", LocationLevel.CITY)],
         "lod": [candidate("QLOD", "Lod", LocationLevel.CITY)],
         "palestine": [candidate("QPAL", "State of Palestine", LocationLevel.COUNTRY)],
         "israel": [candidate("QISR", "Israel", LocationLevel.COUNTRY)]},
        {"QLOD": {"QISR"}, "QPAL": set(), "QISR": set()},
        {"QLOD": {"region": ["Central District"], "country": "Israel", "continent": "Asia"}},
    )
    result = normalizer.resolve(["Lydda", "Lod", "Palestine", "Israel"])
    assert result.status == "partial"
    assert result.head_qid == "QLOD"
    assert result.value.city == "Lod"
    assert [trace.status for trace in result.traces] == ["selected", "selected", "omitted", "selected"]
    assert result.traces[0].selected.role in {"head", "alias"}
    assert result.traces[1].selected.role in {"head", "alias"}

def test_exact_country_beats_more_specific_but_inexact_search_hit():
    normalizer = FakeNormalizer(
        {"denmark": [candidate("Q35", "Denmark", LocationLevel.COUNTRY),
                     candidate("QART", "Denmark Pavilion", LocationLevel.PLACE)]},
        {"Q35": set(), "QART": {"Q35"}},
        {"Q35": {"region": None, "country": "Denmark", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Denmark"])
    assert result.head_qid == "Q35"
    assert result.value.level is LocationLevel.COUNTRY

def test_exact_region_component_beats_reusing_city_head_as_alias():
    normalizer = FakeNormalizer(
        {"new york city": [candidate("Q60", "New York City", LocationLevel.CITY)],
         "new york": [candidate("Q1384", "New York", LocationLevel.REGION),
                      candidate("Q60", "New York City", LocationLevel.CITY)]},
        {"Q60": {"Q1384", "Q30"}, "Q1384": {"Q30"}, "Q30": set()},
        {"Q60": {"city": "New York City", "region": ["New York"],
                  "country": "United States", "continent": "North America"}},
    )
    result = normalizer.resolve(["New York City", "New York"])
    assert result.status == "normalized"
    assert result.head_qid == "Q60"
    assert result.value.city == "New York City"
    assert result.value.region == ["New York"]
    assert result.value.explicit_levels == [LocationLevel.CITY, LocationLevel.REGION]
    assert result.consensus_value == result.value
    assert result.traces[1].selected.qid == "Q1384"
    assert result.traces[1].selected.role == "container"

def test_descriptive_area_suffix_is_removed_without_forcing_a_level():
    assert AuditableWikidataNormalizer._normalize_query("century city area") == "century city"
    assert explicit_component_level("Century City area") is None

def test_ambiguous_area_name_is_not_promoted_to_incompatible_place():
    normalizer = FakeNormalizer(
        {"west hollywood": [candidate("QWH", "West Hollywood", LocationLevel.CITY)],
         "california": [candidate("QCA", "California", LocationLevel.REGION)],
         "united states": [candidate("QUS", "United States", LocationLevel.COUNTRY)],
         "century city area": [candidate("QCC", "Century City", LocationLevel.REGION),
                               candidate("QSA", "Century City", LocationLevel.PLACE)]},
        {"QWH": {"QCA", "QUS"}, "QCA": {"QUS"}, "QUS": set(),
         "QCC": {"QLA", "QCA", "QUS"}, "QSA": {"QCAPE", "QZA"}},
        {"QWH": {"city": "West Hollywood", "region": ["California"],
                  "country": "United States", "continent": "North America"},
         "QCC": {"city": "Los Angeles", "region": ["California", "Century City"],
                 "country": "United States", "continent": "North America"},
         "QSA": {"city": "City of Cape Town", "region": ["Western Cape"],
                  "country": "South Africa", "continent": "Africa"}},
    )
    result = normalizer.resolve([
        "West Hollywood", "California", "United States", "Century City area"
    ])
    assert result.status == "partial"
    assert result.head_qid == "QWH"
    assert result.traces[-1].status == "omitted"
    assert result.traces[-1].place_case is None
    assert result.traces[-1].consensus_candidate_qids == ["QCC"]
    assert "used to derive consensus_value" in result.traces[-1].consensus_note
    assert result.consensus_value.level is LocationLevel.REGION
    assert result.consensus_value.region == ["California"]
    assert result.consensus_value.country == "United States"

def test_common_country_alias_is_selected_not_silently_unresolved():
    normalizer = FakeNormalizer(
        {"united states": [candidate("Q30", "United States", LocationLevel.COUNTRY)]},
        {"Q30": set()},
        {"Q30": {"region": None, "country": "United States", "continent": "North America"}},
    )
    result = normalizer.resolve(["USA"])
    assert result.status == "normalized"
    assert result.value.country == "United States"

def test_unrelated_search_result_is_rejected():
    normalizer = FakeNormalizer(
        {"rhanda": [candidate("QBAD", "large panel system building", LocationLevel.PLACE)]},
        {"QBAD": set()}, {},
    )
    result = normalizer.resolve(["Rhanda"])
    assert result.status == "unresolved"

def test_one_historical_search_hit_does_not_make_modern_conflict_safe():
    normalizer = FakeNormalizer(
        {"vienna": [candidate("QV", "Vienna", LocationLevel.CITY)],
         "austria": [candidate("QA", "Austria", LocationLevel.COUNTRY)],
         "czech republic": [candidate("QCZ", "Czech Republic", LocationLevel.COUNTRY),
                            candidate("QOLD", "Czech lands", LocationLevel.COUNTRY, historical=True)]},
        {"QV": {"QA"}, "QA": set(), "QCZ": set(), "QOLD": set()},
        {"QV": {"region": None, "country": "Austria", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Vienna", "Austria", "Czech Republic"])
    assert result.status == "partial"
    assert result.traces[-1].status == "omitted"

def test_gregory_audit_explicit_iran_is_not_erased_by_qajar_alias():
    # Real audited failure: historical Ctesiphon/Mesopotamia were discarded,
    # an Argentine Mesopotamia won search rank, and Iran was then hidden behind
    # its historical Qajar alias. Keep the contradiction visible for consensus.
    qajar=candidate("Q63158027", "Qajar Iran", LocationLevel.COUNTRY, historical=True)
    qajar['matched_name']='Iran'
    normalizer=FakeNormalizer(
        {'ctesiphon':[candidate('Q192541','Ctesiphon',LocationLevel.CITY,historical=True)],
         'mesopotamia':[candidate('Q11767','Mesopotamia',LocationLevel.REGION,historical=True),
                        candidate('Q1056116','Mesopotamia',LocationLevel.REGION)],
         'iran':[candidate('Q794','Iran',LocationLevel.COUNTRY),qajar],
         'sassanid empire':[candidate('Q83891','Sasanian Empire',LocationLevel.COUNTRY,historical=True)]},
        {'Q1056116':{'Q414'}},
        {'Q1056116':{'region':['Mesopotamia'],'country':'Argentina','continent':'South America'},
         'Q794':{'country':'Iran','continent':'Asia'}},
    )
    result=normalizer.resolve(['Ctesiphon','Mesopotamia','Iran','Sassanid Empire'],
        answer='Gregory of Nyssa died in Ctesiphon (modern-day Mesopotamia, now part of Iran), the capital of the Sassanid Empire.')
    iran=result.traces[2]
    assert result.status=='partial'
    assert iran.status=='omitted'
    assert iran.note=='no candidate belongs to the selected hierarchy'
    assert result.consensus_value is None

def test_entirely_historical_context_may_be_omitted_from_modern_hierarchy():
    normalizer = FakeNormalizer(
        {"berlin": [candidate("QB", "Berlin", LocationLevel.CITY)],
         "kingdom of prussia": [candidate("QP", "Kingdom of Prussia", LocationLevel.COUNTRY, historical=True)],
         "germany": [candidate("QG", "Germany", LocationLevel.COUNTRY)]},
        {"QB": {"QG"}, "QP": set(), "QG": set()},
        {"QB": {"region": None, "country": "Germany", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Berlin", "Kingdom of Prussia", "Germany"])
    assert result.status == "normalized"
    historical = result.traces[1].candidate_hierarchies
    assert len(historical) == 1
    assert historical[0].qid == "QP"
    assert historical[0].historical
    assert historical[0].value is None
    assert "excluded from present-day hierarchy" in historical[0].note

def test_unresolved_component_does_not_invent_a_conflicting_consensus_branch():
    normalizer = FakeNormalizer(
        {"stockholm": [candidate("QSTO", "Stockholm", LocationLevel.CITY)],
         "hillereds slott": []},
        {"QSTO": {"QSE"}},
        {"QSTO": {"region": ["Stockholm County"], "country": "Sweden",
                    "continent": "Europe"}},
    )
    result = normalizer.resolve(["Stockholm", "Hillereds slott"])
    assert result.status == "partial"
    assert result.value.city == "Stockholm"
    assert result.consensus_value.level is LocationLevel.CITY
    assert result.consensus_value.country == "Sweden"
    assert result.consensus_value.city == "Stockholm"
    assert "not evidence of geographic conflict" in result.traces[1].consensus_note

def test_trace_exposes_search_candidates_and_plausible_hierarchies():
    normalizer = FakeNormalizer(
        {"springfield": [candidate("QIL", "Springfield", LocationLevel.CITY),
                          candidate("QMA", "Springfield", LocationLevel.CITY)]},
        {"QIL": {"QUS"}, "QMA": {"QUS"}},
        {"QIL": {"region": ["Illinois"], "country": "United States",
                   "continent": "North America"},
         "QMA": {"region": ["Massachusetts"], "country": "United States",
                   "continent": "North America"}},
    )
    result = normalizer.resolve(["Springfield"])
    trace = result.traces[0]
    assert [item.qid for item in trace.candidates] == ["QIL", "QMA"]
    assert {item.qid for item in trace.candidate_hierarchies} == {"QIL", "QMA"}
    assert {item.value.region[0] for item in trace.candidate_hierarchies} == {
        "Illinois", "Massachusetts"
    }
    assert result.status == "normalized"
    assert result.value.region == ["Illinois"]
    assert result.disambiguation.status == "search_rank_fallback"
    assert result.disambiguation.ambiguous
    assert result.disambiguation.selected_qid == "QIL"
    assert result.disambiguation.alternative_qids == ["QMA"]
    assert result.consensus_value.level is LocationLevel.CITY
    assert result.consensus_value.city == "Springfield"
    assert result.consensus_value.region == ["Illinois"]
    assert result.consensus_value.country == "United States"

def test_other_component_resolves_same_name_candidate_without_rank_fallback():
    normalizer = FakeNormalizer(
        {"springfield": [candidate("QIL", "Springfield", LocationLevel.CITY),
                          candidate("QMA", "Springfield", LocationLevel.CITY)],
         "massachusetts": [candidate("QMAREG", "Massachusetts", LocationLevel.REGION)]},
        {"QIL": {"QUS"}, "QMA": {"QMAREG", "QUS"}, "QMAREG": {"QUS"}},
        {"QMA": {"region": ["Massachusetts"], "country": "United States",
                   "continent": "North America"}},
    )
    result = normalizer.resolve(["Springfield", "Massachusetts"])
    assert result.status == "normalized"
    assert result.head_qid == "QMA"
    assert result.value.region == ["Massachusetts"]
    assert result.disambiguation.status == "context_resolved"
    assert not result.disambiguation.ambiguous
    assert result.disambiguation.alternative_qids == []

def test_no_context_uses_search_rank_before_candidate_precision():
    normalizer = FakeNormalizer(
        {"example": [candidate("QREG", "Example", LocationLevel.REGION),
                      candidate("QCITY", "Example", LocationLevel.CITY)]},
        {"QREG": {"QCOUNTRY"}, "QCITY": {"QCOUNTRY"}},
        {"QREG": {"region": ["Example"], "country": "Exampleland",
                    "continent": "Europe"},
         "QCITY": {"region": ["Other Region"], "country": "Exampleland",
                     "continent": "Europe"}},
    )
    result = normalizer.resolve(["Example"])
    assert result.head_qid == "QREG"
    assert result.value.level is LocationLevel.REGION
    assert result.disambiguation.status == "search_rank_fallback"
    assert result.disambiguation.alternative_qids == ["QCITY"]

def test_direct_p131_chain_resolves_hampstead_london_hierarchy():
    normalizer = FakeNormalizer(
        {"hampstead": [candidate("QH", "Hampstead", LocationLevel.REGION)],
         "london": [candidate("QL", "London", LocationLevel.CITY)],
         "england": [candidate("QE", "England", LocationLevel.COUNTRY)]},
        {"QH": {"QL", "QE"}, "QL": {"QE"}, "QE": set()},
        {"QH": {"region": ["Hampstead", "Greater London"],
                  "country": "United Kingdom", "continent": "Europe"},
         "QL": {"region": ["Greater London"], "country": "United Kingdom",
                  "continent": "Europe"},
         "QE": {"region": None, "country": "United Kingdom", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Hampstead", "London", "England"])
    assert result.status == "normalized"
    assert [trace.status for trace in result.traces] == ["selected", "selected", "selected"]
    assert result.traces[0].selected.qid == "QH"
    assert result.value.city == "London"
    assert "Hampstead" in result.value.region

def test_exact_historical_region_wins_over_modern_name_extensions_for_omission():
    normalizer = FakeNormalizer(
        {"kafiristan": [candidate("QK", "Kafiristan", LocationLevel.REGION)],
         "afghanistan": [candidate("QAF", "Afghanistan", LocationLevel.COUNTRY)],
         "khorasan": [candidate("QH", "Khorasan", LocationLevel.REGION, historical=True),
                      candidate("QMOD", "Razavi Khorasan Province", LocationLevel.REGION)]},
        {"QK": {"QAF"}, "QAF": set(), "QH": set(), "QMOD": set()},
        {"QK": {"region": ["Kafiristan"], "country": "Afghanistan", "continent": "Asia"}},
    )
    result = normalizer.resolve(["Kafiristan", "Afghanistan", "Khorasan"])
    assert result.status == "normalized"
    assert result.traces[-1].status == "omitted"

def test_constituent_country_completes_to_sovereign_country():
    normalizer = FakeNormalizer(
        {"england": [candidate("QE", "England", LocationLevel.COUNTRY)]},
        {"QE": set()},
        {"QE": {"region": None, "country": "United Kingdom", "continent": "Europe"}},
    )
    result = normalizer.resolve(["England"])
    assert result.status == "normalized"
    assert result.value.level is LocationLevel.COUNTRY
    assert result.value.country == "United Kingdom"
    assert result.traces[0].selected.label == "England"

def test_region_inherits_country_from_typed_admin_ancestor_without_p17():
    rows = [
        {"adminLabel": {"value": "Styria"},
         "adminDescription": {"value": "state of Austria"},
         "continentLabel": {"value": "Europe"}},
        {"adminLabel": {"value": "Austria"},
         "adminDescription": {"value": "country in Central Europe"},
         "continentLabel": {"value": "Europe"}},
    ]
    metadata = AuditableWikidataNormalizer._metadata_from_bindings(rows)
    assert metadata == {"city": None, "subcity": None, "region": ["Styria"],
                        "country": "Austria", "continent": "Europe", "country_candidates": ["Austria"]}

def test_place_classification_is_conservative():
    assert is_generic_place("his family's home")
    assert is_place_expression("Pecoli Homestead")
    assert is_place_expression("10 Brattle Street")
    assert not is_place_expression("New York")

def test_semantic_type_prefix_is_removed_only_for_wikidata_lookup():
    assert AuditableWikidataNormalizer._normalize_query(
        "borough of Marylebone"
    ) == "Marylebone"
    assert AuditableWikidataNormalizer._normalize_query(
        "province of Valencia"
    ) == "Valencia"
    assert explicit_component_level("borough of Marylebone") is LocationLevel.SUBCITY
    assert explicit_component_level("province of Valencia") is LocationLevel.REGION

def test_wikidata_university_description_is_place_level():
    assert AuditableWikidataNormalizer._level_from_description(
        "state public university in Kolkata, West Bengal, India"
    ) is LocationLevel.PLACE

def test_wikidata_region_description_wins_over_later_city_mention():
    assert AuditableWikidataNormalizer._level_from_description(
        "region of England containing the City of London and the rest of Greater London"
    ) is LocationLevel.REGION

def test_wikidata_county_context_does_not_downgrade_city_entity():
    assert AuditableWikidataNormalizer._level_from_description(
        "seat of Los Angeles County, and largest city in California, United States"
    ) is LocationLevel.CITY

def test_wikidata_neighborhood_description_is_subcity():
    assert AuditableWikidataNormalizer._level_from_description(
        "neighborhood in Los Angeles, California, United States"
    ) is LocationLevel.SUBCITY

def test_wikidata_directional_city_area_description_is_subcity():
    assert AuditableWikidataNormalizer._level_from_description(
        "area of north London, England"
    ) is LocationLevel.SUBCITY

def test_wikidata_london_borough_description_is_subcity():
    assert AuditableWikidataNormalizer._level_from_description(
        "borough in the London Region in England"
    ) is LocationLevel.SUBCITY

def test_wikidata_generic_area_description_remains_region_level():
    assert AuditableWikidataNormalizer._level_from_description(
        "area in Northern California"
    ) is LocationLevel.REGION

def test_federal_district_description_does_not_become_its_contained_city():
    rows = [
        {"adminLabel": {"value": "Washington, D.C."},
         "adminDescription": {"value": "capital city of the United States of America"}},
        {"adminLabel": {"value": "District of Columbia"},
         "adminDescription": {"value": "federal district of the United States of America, containing the capital city of the United States, Washington"},
         "countryLabel": {"value": "United States"},
         "countryContinentLabel": {"value": "North America"}},
    ]
    metadata = AuditableWikidataNormalizer._metadata_from_bindings(rows)
    assert metadata["city"] == "Washington, D.C."
    assert metadata["region"] == ["District of Columbia"]

def test_direct_country_continent_wins_over_unrelated_ancestor_continents():
    rows = [
        {"directCountryLabel": {"value": "Republic of the Congo"},
         "directCountryContinentLabel": {"value": "Africa"},
         "countryContinentLabel": {"value": "Europe"}},
        {"directCountryLabel": {"value": "Republic of the Congo"},
         "directCountryContinentLabel": {"value": "Africa"},
         "countryContinentLabel": {"value": "Africa"}},
    ]
    metadata = AuditableWikidataNormalizer._metadata_from_bindings(rows)
    assert metadata["continent"] == "Africa"

def test_manhattan_and_new_york_city_are_preserved_at_distinct_levels():
    normalizer = FakeNormalizer(
        {"manhattan": [candidate("QM", "Manhattan", LocationLevel.SUBCITY)],
         "new york city": [candidate("QNY", "New York City", LocationLevel.CITY)]},
        {"QM": {"QNY", "QUS"}, "QNY": {"QUS"}},
        {"QM": {"city": "New York City", "region": ["New York"],
                "country": "United States", "continent": "North America"}},
    )
    result = normalizer.resolve(["New York City", "Manhattan"])
    assert result.value.level is LocationLevel.SUBCITY
    assert result.value.subcity == ["Manhattan"]
    assert result.value.city == "New York City"
    assert result.value.to_value().to_level(LocationLevel.CITY).city == "New York City"
    assert result.consensus_value.subcity == ["Manhattan"]

def test_context_country_cannot_replace_a_different_city_as_an_alias():
    class ContextNormalizer(FakeNormalizer):
        def _context_search(self, component, context):
            return [candidate("QCG", "Republic of the Congo", LocationLevel.COUNTRY)]
    normalizer = ContextNormalizer(
        {"brazzaville": [candidate("QB", "Brazzaville", LocationLevel.CITY)],
         "republic of the congo": [candidate("QCG", "Republic of the Congo", LocationLevel.COUNTRY)]},
        {"QB": {"QCG"}},
        {"QB": {"country": "Republic of the Congo", "continent": "Africa"}},
    )
    result = normalizer.resolve(["Brazzaville", "Republic of the Congo"])
    assert result.value.city == "Brazzaville"
    assert result.traces[0].selected.qid == "QB"

def test_same_named_venue_does_not_absorb_selected_city():
    normalizer = FakeNormalizer(
        {"san francisco general hospital": [candidate("QH", "San Francisco General Hospital", LocationLevel.PLACE)],
         "san francisco": [candidate("QC", "San Francisco", LocationLevel.CITY),
                           candidate("QCHURCH", "San Francisco", LocationLevel.PLACE)],
         "california": [candidate("QCA", "California", LocationLevel.REGION)]},
        {"QH": {"QC", "QCA"}, "QCHURCH": {"QC", "QCA"}, "QC": {"QCA"}},
        {"QC": {"region": ["California"], "country": "United States", "continent": "North America"}},
    )
    result = normalizer.resolve(["San Francisco General Hospital", "San Francisco", "California"])
    assert result.value.place == ["San Francisco General Hospital"]
    assert result.value.city == "San Francisco"
    assert result.traces[1].status == "selected"

def test_historical_country_never_becomes_present_day_head():
    normalizer = FakeNormalizer(
        {"kingdom of valencia": [candidate("QOLD", "Kingdom of Valencia", LocationLevel.COUNTRY, historical=True)],
         "spain": [candidate("QES", "Spain", LocationLevel.COUNTRY)]},
        {"QOLD": set(), "QES": set()},
        {"QES": {"country": "Spain", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Unknown village", "Kingdom of Valencia", "Spain"])
    assert result.status == "partial"
    assert result.head_qid == "QES"
    assert result.value.country == "Spain"
    assert result.consensus_value.country == "Spain"
    assert result.traces[1].status == "omitted"

def test_city_country_conflict_retains_city_branch_and_only_shared_continent_for_consensus():
    normalizer = FakeNormalizer(
        {"tus": [candidate("QTUS", "Tus", LocationLevel.CITY)],
         "iraq": [candidate("QIQ", "Iraq", LocationLevel.COUNTRY)]},
        {"QTUS": {"QIR"}, "QIQ": set()},
        {"QTUS": {"country": "Iran", "continent": "Asia"},
         "QIQ": {"country": "Iraq", "continent": "Asia"}},
    )
    result = normalizer.resolve(["Tus", "Iraq"])
    assert result.status == "partial"
    assert result.value.city == "Tus"
    assert result.value.country == "Iran"
    assert result.traces[1].status == "omitted"
    assert result.consensus_value.level is LocationLevel.CONTINENT
    assert result.consensus_value.continent == "Asia"

def test_generic_home_name_is_not_selected_as_same_named_village():
    normalizer = FakeNormalizer(
        {"windlesham": [candidate("QV", "Windlesham", LocationLevel.CITY),
                        candidate("QHOUSE", "Windlesham", LocationLevel.PLACE)],
         "hindhead": [candidate("QH", "Hindhead", LocationLevel.CITY)]},
        {"QV": set(), "QHOUSE": {"QH"}, "QH": set()},
        {"QH": {"country": "United Kingdom", "continent": "Europe"}},
    )
    result = normalizer.resolve(["his home", "Windlesham", "Hindhead"])
    assert result.value.city == "Hindhead"
    assert result.traces[1].selected.qid == "QHOUSE"

def test_constituent_country_is_region_with_completed_sovereign_country():
    normalizer = FakeNormalizer(
        {"england": [candidate("QE", "England", LocationLevel.REGION)]},
        {"QE": {"QUK"}},
        {"QE": {"country": "United Kingdom", "continent": "Europe"}},
    )
    result = normalizer.resolve(["England"])
    assert result.value.level is LocationLevel.REGION
    assert result.value.region == ["England"]
    assert result.value.country == "United Kingdom"

def test_square_description_is_place_even_when_it_mentions_a_city_district():
    assert AuditableWikidataNormalizer._level_from_description(
        "square in the 8th arrondissement of Paris, France"
    ) is LocationLevel.PLACE

def test_borough_and_village_are_not_subcity_without_an_urban_parent():
    classify = AuditableWikidataNormalizer._level_from_description
    assert classify("borough in Pennsylvania, United States") is LocationLevel.CITY
    assert classify("village in the borough of Surrey Heath in Surrey, England") is LocationLevel.CITY
    assert classify("borough of New York City, New York, United States") is LocationLevel.SUBCITY

def test_exact_city_label_beats_alias_for_different_context_entity():
    normalizer = FakeNormalizer(
        {"brazzaville": [candidate("QB", "Brazzaville", LocationLevel.CITY),
                         {**candidate("QCG", "Republic of the Congo", LocationLevel.COUNTRY),
                          "matched_name": "Brazzaville"}],
         "republic of the congo": [candidate("QCG", "Republic of the Congo", LocationLevel.COUNTRY)]},
        {"QB": {"QCG"}},
        {"QB": {"country": "Republic of the Congo", "continent": "Africa"}},
    )
    result = normalizer.resolve(["Brazzaville", "Republic of the Congo"])
    assert result.value.city == "Brazzaville"

def test_exact_country_does_not_become_same_named_foreign_town():
    normalizer = FakeNormalizer(
        {"turkey": [candidate("QT", "Turkey", LocationLevel.COUNTRY),
                    candidate("QTOWN", "Turkey", LocationLevel.CITY)]},
        {"QT": set(), "QTOWN": {"QUS"}},
        {"QT": {"country": "Turkey", "continent": "Asia"},
         "QTOWN": {"country": "United States", "continent": "North America"}},
    )
    result = normalizer.resolve(["Unresolved region", "Turkey"])
    assert result.value.country == "Turkey"
    assert result.value.city is None

def test_city_context_recovers_the_compatible_naval_hospital():
    class ContextNormalizer(FakeNormalizer):
        def _context_search(self, component, context):
            if component == "Naval Hospital" and context == "Washington, D.C.":
                return [candidate("Q113016429", "U.S. Naval Hospital", LocationLevel.PLACE)]
            return []

    normalizer = ContextNormalizer(
        {"naval hospital": [candidate("QWRONG", "Naval Hospital", LocationLevel.PLACE)],
         "washington, d.c.": [candidate("Q61", "Washington, D.C.", LocationLevel.CITY)]},
        {"QWRONG": {"QUK"}, "Q113016429": {"Q61", "QDC", "QUS"},
         "Q61": {"QDC", "QUS"}},
        {"Q61": {"region": ["District of Columbia"], "country": "United States",
                 "continent": "North America"},
         "QWRONG": {"country": "United Kingdom", "continent": "Europe"}},
    )
    result = normalizer.resolve(["Naval Hospital", "Washington, D.C."])
    assert result.value.region == ["District of Columbia"]
    assert result.traces[0].selected.qid == "Q113016429"
    assert result.traces[0].place_case == "wikidata_compatible"

def test_compatible_place_hierarchy_supplies_missing_city():
    normalizer = FakeNormalizer(
        {
            "hatfield house": [candidate("QHOUSE", "Hatfield House", LocationLevel.PLACE)],
            "hertfordshire": [candidate("QHERTS", "Hertfordshire", LocationLevel.REGION)],
            "england": [candidate("QENG", "England", LocationLevel.REGION)],
        },
        {
            "QHOUSE": {"QHATFIELD", "QHERTS", "QENG", "QUK"},
            "QHERTS": {"QENG", "QUK"},
            "QENG": {"QUK"},
        },
        {
            "QHOUSE": {
                "city": "Hatfield", "region": ["Hertfordshire", "Welwyn Hatfield"],
                "country": "United Kingdom", "continent": "Europe",
            },
            "QHERTS": {
                "region": ["Hertfordshire", "England"],
                "country": "United Kingdom", "continent": "Europe",
            },
            "QENG": {
                "region": ["England"], "country": "United Kingdom",
                "continent": "Europe",
            },
        },
    )
    result = normalizer.resolve(["Hatfield House", "Hertfordshire", "England"])
    assert result.status == "normalized"
    assert result.value.level is LocationLevel.PLACE
    assert result.value.place == ["Hatfield House"]
    assert result.value.city == "Hatfield"
    assert "Welwyn Hatfield" in result.value.region
    assert result.traces[0].place_case == "wikidata_compatible"

def test_oblast_is_an_explicit_region_type():
    assert AuditableWikidataNormalizer._level_from_description(
        "oblast in central Ukraine"
    ) is LocationLevel.REGION
    assert explicit_component_level("Cherkasy Oblast") is LocationLevel.REGION

def test_context_qualifier_parser_preserves_relation_but_cleans_lookup():
    assert contextual_component("near modern-day Xi'an") == ("near", "Xi'an")
    assert contextual_component("outside Paris") == ("outside", "Paris")
    assert contextual_component("New York") is None

def test_nearby_city_is_context_not_a_second_head_or_explicit_level():
    normalizer = FakeNormalizer(
        {"chang'an": [candidate("QCH", "Chang'an", LocationLevel.CITY)],
         "xi'an": [candidate("QXI", "Xi'an", LocationLevel.CITY)],
         # Wikidata's description mentions the provincial capital and can be
         # classified as CITY by the generic description parser.  The literal
         # "Province" suffix must override that ambiguity.
         "shaanxi province": [candidate("QSH", "Shaanxi", LocationLevel.CITY)],
         "china": [candidate("QCN", "People's Republic of China", LocationLevel.COUNTRY)]},
        {"QCH": {"QSH", "QCN"}, "QXI": {"QSH", "QCN"},
         "QSH": {"QCN"}, "QCN": set()},
        {"QCH": {"region": ["Shaanxi"], "country": "People's Republic of China",
                  "continent": "Asia"}},
    )
    result = normalizer.resolve(["Chang'an", "near Xi'an", "Shaanxi Province", "China"])
    assert result.status == "normalized"
    assert result.value.city == "Chang'an"
    assert result.value.region == ["Shaanxi"]
    assert result.value.explicit_levels == [LocationLevel.CITY, LocationLevel.REGION, LocationLevel.COUNTRY]
    assert [trace.status for trace in result.traces] == ["selected", "context", "selected", "selected"]
    assert result.traces[1].relation == "near"
    assert result.traces[1].selected.role == "context"
    assert result.traces[1].selected.label == "Xi'an"

def test_nearby_city_without_requested_head_is_partial_not_the_city():
    normalizer = FakeNormalizer(
        {"xi'an": [candidate("QXI", "Xi'an", LocationLevel.CITY)],
         "shaanxi province": [candidate("QSH", "Shaanxi", LocationLevel.REGION)],
         "china": [candidate("QCN", "People's Republic of China", LocationLevel.COUNTRY)]},
        {"QXI": {"QSH", "QCN"}, "QSH": {"QCN"}, "QCN": set()},
        {"QSH": {"region": ["Shaanxi"], "country": "People's Republic of China",
                  "continent": "Asia"}},
    )
    result = normalizer.resolve(["near Xi'an", "Shaanxi Province", "China"])
    assert result.status == "partial"
    assert result.value.level is LocationLevel.REGION
    assert result.value.city is None
