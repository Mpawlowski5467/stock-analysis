"""Coarse sector buckets from SIC codes (SIC divisions).

Used for within-sector cross-sectional normalization. This is the standard SIC
division split (~9 buckets); a finer Fama-French 12/48 mapping is a refinement
once the full universe makes sector buckets reliably large (see DESIGN.md).
"""

from __future__ import annotations


def sic_division(sic) -> str:
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    if s <= 999:
        return "Agriculture"
    if s <= 1499:
        return "Mining"
    if s <= 1799:
        return "Construction"
    if s <= 3999:
        return "Manufacturing"
    if s <= 4999:
        return "Transport/Utilities"
    if s <= 5199:
        return "Wholesale"
    if s <= 5999:
        return "Retail"
    if s <= 6799:
        return "Finance"
    if s <= 8999:
        return "Services"
    return "Public/Other"


def sic_industry(sic) -> str:
    """A finer, recognizable industry label from a 4-digit SIC (DISPLAY ONLY).

    Refines sic_division into ~45 named buckets for the markets overview
    ('Semiconductors', 'Oil & Gas E&P', 'Software', 'Banks', 'Biotech', …). This
    is NOT a model input — the model still normalizes within the coarse
    sic_division sector; this only regroups names for the markets page. Unmapped
    codes fall back to the coarse division so nothing is ever dropped.
    """
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"

    def rng(a, b) -> bool:
        return a <= s <= b

    # --- Energy & utilities ---
    if rng(1310, 1329):
        return "Oil & Gas E&P"
    if rng(1380, 1389):
        return "Oil & Gas Services"
    if rng(2910, 2919):
        return "Oil Refining"
    if rng(1200, 1299):
        return "Coal"
    if s in (4922, 4923, 4924, 4925) or rng(4610, 4619):
        return "Pipelines & Gas Utilities"
    if rng(4900, 4911) or rng(4930, 4939):
        return "Electric & Multi Utilities"
    if rng(4940, 4949):
        return "Water Utilities"

    # --- Healthcare ---
    if s == 2834:
        return "Pharmaceuticals"
    if s in (2836, 8731):
        return "Biotech"
    if rng(3840, 3851):
        return "Medical Devices"
    if rng(8000, 8099):
        return "Healthcare Services"
    # lab / measuring instruments (Agilent, Keysight, Mettler-Toledo, Rockwell): the
    # 382x block sits between the medical-device and aerospace rules and fell through
    # to a bare 'Manufacturing' for ~40 names
    if rng(3820, 3839):
        return "Instruments & Measurement"

    # --- Materials & chemicals ---
    if rng(1000, 1199) or rng(1400, 1499):
        return "Metals & Mining"
    if rng(3300, 3399):
        return "Steel & Metals"
    if rng(2800, 2899):
        return "Chemicals"
    if rng(2600, 2699):
        return "Paper & Packaging"
    if rng(3200, 3299) or rng(2400, 2499):
        return "Building Materials"

    # --- Technology, media & telecom ---
    if s == 3674:
        return "Semiconductors"
    if rng(3670, 3679) or rng(3600, 3629) or rng(3690, 3699):
        return "Electronic Components"
    if rng(3570, 3579) or rng(3680, 3689):
        return "Computer Hardware"
    if rng(3660, 3669):
        return "Communications Equipment"
    if s == 7372:
        return "Software"
    if rng(7370, 7379):
        return "IT Services & Internet"
    if s == 7389:
        return "Business Services"
    if rng(7360, 7369):
        return "Staffing & HR Services"
    if rng(7350, 7359):
        return "Equipment Rental & Leasing"
    # 8731 (commercial research) is already Biotech above; 874x is the consulting block
    if rng(8740, 8748):
        return "Consulting & Management Services"
    if rng(8200, 8299):
        return "Education"
    if rng(4800, 4899):
        return "Telecom"
    if rng(2700, 2799):
        return "Publishing"
    if rng(7800, 7999) or rng(4830, 4833):
        return "Media & Entertainment"

    # --- Financials ---
    if rng(6000, 6036):
        return "Banks"
    if rng(6100, 6199):
        return "Consumer & Specialty Finance"
    if rng(6200, 6299):
        return "Capital Markets & Asset Mgmt"
    if rng(6300, 6399) or s == 6411:
        return "Insurance"
    if s == 6798:
        return "REITs"
    if rng(6500, 6599):
        return "Real Estate"
    if rng(6700, 6799):
        return "Holding & Investment"

    # --- Consumer ---
    if rng(5800, 5813):
        return "Restaurants"
    # hotels/casinos only: 799x (amusement & recreation) is already claimed by the
    # Media & Entertainment rule above, and that ordering is deliberate
    if rng(7000, 7021):
        return "Hotels, Casinos & Leisure"
    if rng(7200, 7299):
        return "Consumer Services"
    if rng(2080, 2085):
        return "Beverages"
    if rng(2000, 2099):
        return "Food Products"
    if rng(2100, 2199):
        return "Tobacco"
    if rng(2200, 2399) or rng(3100, 3170):
        return "Apparel & Textiles"
    if rng(3710, 3719):
        return "Autos & Parts"
    if rng(3720, 3728) or rng(3760, 3769) or s == 3812:
        return "Aerospace & Defense"
    if rng(5200, 5799) or rng(5900, 5999):
        return "Retail"
    if rng(5000, 5199):
        return "Wholesale"
    if rng(2500, 2599) or rng(2840, 2844) or rng(3630, 3639):
        return "Consumer Products"
    if rng(3050, 3099):
        return "Plastics & Rubber"

    # --- Industrials ---
    if rng(3500, 3569):
        return "Machinery & Equipment"
    if rng(3400, 3499):
        return "Fabricated Products"
    if rng(1500, 1799):
        return "Construction & Homebuilding"
    if rng(4000, 4599) or rng(4700, 4799):
        return "Transportation & Logistics"

    return sic_division(s)
