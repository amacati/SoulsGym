from soulsgym.core.static import address_base_patterns, address_bases, addresses


def test_elden_ring_addresses_reference_configured_bases():
    configured_bases = set(address_bases["EldenRing"] or {}) | set(
        address_base_patterns["EldenRing"]
    )
    referenced_bases = {
        record["base"] for record in addresses["EldenRing"].values() if "base" in record
    }

    assert referenced_bases <= configured_bases