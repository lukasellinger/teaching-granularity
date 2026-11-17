"""Location label-set projection shared by case loading and scoring."""
from parsing.location.normalization.values import LocationLevel

FULL = "full"


FOUR_LEVEL = "four-level"


FOUR_LEVELS = (LocationLevel.CITY, LocationLevel.REGION,
               LocationLevel.COUNTRY, LocationLevel.CONTINENT)


def project_location(value, taxonomy=FULL):
    if taxonomy not in (FULL, FOUR_LEVEL):
        raise ValueError(f"Unknown location taxonomy: {taxonomy}")
    if value is None or taxonomy == FULL:
        return value
    return project_four_level_location(value)


def project_four_level_location(value):
    if value is not None:
        for level in FOUR_LEVELS:
            try:
                return value.to_level(level)
            except ValueError:
                pass
    return None

