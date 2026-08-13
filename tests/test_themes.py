"""Network-free tests for thematic auto-tagging (live-view markets)."""

from stockscan import themes as T


def test_tag_themes_hits_named_themes():
    assert "AI" in T.tag_themes("a platform powered by artificial intelligence and machine learning")
    assert "SaaS" in T.tag_themes("a subscription-based software-as-a-service product")
    assert "Electric Vehicles" in T.tag_themes("manufactures battery electric vehicles and EV charging")
    assert "Cybersecurity" in T.tag_themes("endpoint security and threat detection for enterprises")
    assert "Clean Energy" in T.tag_themes("develops utility-scale solar and wind power projects")


def test_tag_themes_avoids_obvious_false_positives():
    # bare ambiguous tokens must not trip a theme
    assert T.tag_themes("the aircraft maintains altitude over the airfield") == []   # no 'AI'
    assert T.tag_themes("a chain of family restaurants") == []                        # no 'AI'
    assert "Electric Vehicles" not in T.tag_themes("every product we ship is durable")  # no bare 'ev'
    assert T.tag_themes("") == [] and T.tag_themes(None) == []


def test_v2_themes_catch_groups_the_sic_industries_cannot_express():
    """Each of these is filed under an industry that hides what the company does:
    bitcoin miners under Finance, quantum hardware under Services, nuclear split
    across Mining/Manufacturing/Utilities."""
    assert "Bitcoin Mining" in T.tag_themes("develops, owns, and operates bitcoin mining facility sites")
    assert "Quantum Computing" in T.tag_themes("development of general-purpose quantum computing systems")
    assert "Uranium & Nuclear" in T.tag_themes("exploration and operation of uranium mineral properties")
    assert "Hydrogen & Fuel Cells" in T.tag_themes("provides hydrogen fuel cell turnkey solutions")
    assert "Robotics & Automation" in T.tag_themes("machine vision for assembly and robotic guidance")
    assert "Data Centers" in T.tag_themes("manufactures transceivers for data centers and telecom")
    assert "Genomics & Gene Therapy" in T.tag_themes("an AAV-based gene therapy for Sanfilippo syndrome")
    assert "Digital Health" in T.tag_themes("operates a multi-specialty telehealth platform")
    assert "Drones & Autonomy" in T.tag_themes("manufactures unmanned aerial and aircraft systems")
    assert "Rare Earths & Critical Minerals" in T.tag_themes("owns rare earth mining and processing facilities")
    assert "Semiconductor Equipment" in T.tag_themes("subsystems for semiconductor capital equipment")
    assert "3D Printing" in T.tag_themes("provides 3D printing and digital manufacturing solutions")
    assert "Energy Storage & Grid" in T.tag_themes("sells power generation equipment and energy storage systems")


def test_v2_themes_hold_the_precision_line():
    """The false positives found in the description corpus while writing these rules."""
    # seismic dampers, not batteries — the noun after "energy storage" carries the sense
    assert "Energy Storage & Grid" not in T.tag_themes(
        "markets shock absorption, rate control, and energy storage devices")
    # a boot maker that happens to sell online is not a thematic e-commerce name;
    # the theme was dropped rather than shipped imprecise
    assert "E-commerce" not in T.tag_themes("sells its products through its e-commerce websites")
    # hydrogen peroxide is a chemical, not the hydrogen economy
    assert "Hydrogen & Fuel Cells" not in T.tag_themes("produces hydrogen peroxide and other chemicals")
    # a bank is an industry, not a theme — no rule should have been written for it
    assert T.tag_themes("provides commercial banking and consumer loans") == []


def test_theme_rules_do_not_restate_the_sic_industries():
    """A theme that names an industry the scan filter already offers is a second
    name for the same set — the v2 rules were scoped to avoid exactly that."""
    from stockscan.sector import sic_industry

    industry_labels = {sic_industry(s) for s in range(100, 9000)}
    overlap = {t for t in T.THEME_RULES if t in industry_labels}
    assert not overlap, f"these themes duplicate a sic_industry bucket: {sorted(overlap)}"


def test_tag_themes_multi_membership():
    hits = T.tag_themes("a cloud computing platform using artificial intelligence")
    assert set(hits) == {"AI", "Cloud"}


def test_store_round_trip_and_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "THEMES_DB_PATH", tmp_path / "themes.sqlite")
    with T.ThemeStore() as store:
        store.put(320193, ["AI", "Cloud"])
        store.put(2, ["SaaS"])
        store.commit()
    assert T.load_theme_tags() == {320193: ["AI", "Cloud"], 2: ["SaaS"]}
    with T.ThemeStore() as store:
        store.clear(2)
        store.commit()
    assert T.load_theme_tags() == {320193: ["AI", "Cloud"]}


def test_refresh_tags_builds_from_descriptions(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "THEMES_DB_PATH", tmp_path / "themes.sqlite")
    descs = {
        1: "artificial intelligence analytics",           # AI
        2: "software-as-a-service for accounting",        # SaaS
        3: "a regional bank holding company",             # nothing
    }
    stats = T.refresh_theme_tags([1, 2, 3], get_desc=descs.get, max_workers=2)
    assert stats["scanned"] == 3 and stats["tagged"] == 2
    assert stats["by_theme"] == {"AI": 1, "SaaS": 1}
    tags = T.load_theme_tags()
    assert tags == {1: ["AI"], 2: ["SaaS"]}               # untagged cik 3 not stored
