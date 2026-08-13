"""Tests for the SIC -> sector division (model) + fine industry (display) mappings."""

from stockscan.sector import sic_division, sic_industry


def test_sic_division_buckets():
    assert sic_division(3571) == "Manufacturing"
    assert sic_division(6021) == "Finance"
    assert sic_division(7372) == "Services"
    assert sic_division(1311) == "Mining"
    assert sic_division(5411) == "Retail"


def test_sic_division_handles_missing():
    assert sic_division(None) == "Unknown"
    assert sic_division(float("nan")) == "Unknown"


def test_sic_industry_maps_named_industries():
    cases = {
        3674: "Semiconductors",
        1311: "Oil & Gas E&P",
        2911: "Oil Refining",
        1381: "Oil & Gas Services",
        4923: "Pipelines & Gas Utilities",
        4911: "Electric & Multi Utilities",
        7372: "Software",
        7373: "IT Services & Internet",
        2834: "Pharmaceuticals",
        2836: "Biotech",
        6022: "Banks",
        6798: "REITs",
        6311: "Insurance",
        5812: "Restaurants",
        3711: "Autos & Parts",
    }
    for sic, label in cases.items():
        assert sic_industry(sic) == label, f"SIC {sic} -> {sic_industry(sic)!r}, want {label!r}"


def test_sic_industry_covers_the_blocks_that_fell_through_to_a_bare_division():
    """Codes that used to render as a bare 'Manufacturing'/'Services' in the scan
    table — the whole point of the fine label is that these read as something."""
    cases = {
        3826: "Instruments & Measurement",          # Agilent, Keysight, Mettler-Toledo
        3823: "Instruments & Measurement",
        3829: "Instruments & Measurement",
        7011: "Hotels, Casinos & Leisure",          # Las Vegas Sands, Marriott
        8742: "Consulting & Management Services",   # Booz Allen
        8200: "Education",                          # Stride, Adtalem
        7363: "Staffing & HR Services",             # Robert Half, Insperity
        7359: "Equipment Rental & Leasing",         # United Rentals
        3089: "Plastics & Rubber",
        7200: "Consumer Services",                  # H&R Block
    }
    for sic, label in cases.items():
        got = sic_industry(sic)
        assert got == label, f"SIC {sic} -> {got!r}, want {label!r}"
        assert got != sic_division(sic), f"SIC {sic} still reads as its coarse division"


def test_new_industry_rules_do_not_steal_from_the_existing_ones():
    """Every rule added sits in an ordered if-chain, so a too-wide range silently
    captures codes an earlier-intended rule should own."""
    assert sic_industry(8731) == "Biotech"                  # not the 874x consulting block
    assert sic_industry(7812) == "Media & Entertainment"    # 78xx-79xx keeps amusement
    assert sic_industry(7371) == "IT Services & Internet"   # not 736x staffing
    assert sic_industry(7372) == "Software"
    assert sic_industry(3841) == "Medical Devices"          # not 382x instruments
    assert sic_industry(3812) == "Aerospace & Defense"      # not 382x instruments


def test_sic_industry_precedence_and_fallback():
    # specific codes must win over the broad range they sit inside
    assert sic_industry(2834) == "Pharmaceuticals"   # not the 2800-2899 Chemicals range
    assert sic_industry(2082) == "Beverages"         # not the 2000-2099 Food range
    # an unmapped-but-valid code falls back to the coarse division, never empty
    assert sic_industry(9995) == sic_division(9995)
    assert sic_industry(None) == "Unknown"
