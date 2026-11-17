"""Reproduce the original city-label candidate estimator on reviewed gold.

Original: uncertainty-granularity-calibration/scripts/calc_location_informativeness.py.
Preserves its distinct-city-label convention and event-weighted geometric means.
Regions are first averaged within each event, then across events with region data.
"""
import argparse
from shared.io import file_hash, write
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_METADATA = Path(__file__).resolve().parents[2] / "data_preparation/vital_persons/metadata/gold_adapted.jsonl"
PARAMETERS_PATH = Path(__file__).with_name("location_informativeness.json")


def geometric_mean(values: list[float]) -> float:
    positive = [value for value in values if value > 0]

    if not positive:
        raise ValueError("Cannot compute geometric mean without positive values.")

    return math.exp(
        sum(math.log(value) for value in positive) / len(positive)
    )


def load_location_records(path: Path) -> list[dict]:
    """
    Extract one record for every available birth/death city.

    Each record contains:
      - city
      - all intermediate regions
      - country
      - continent
    """
    records = []

    with path.open() as f:
        for line in f:
            row = json.loads(line)

            for prefix in ("birth", "death"):
                city = row.get(f"{prefix}_city")
                if city is None:
                    continue

                records.append(
                    {
                        "city": city,
                        "regions": row.get(f"{prefix}_region") or [],
                        "country": row.get(f"{prefix}_country"),
                        "continent": row.get(f"{prefix}_continent"),
                    }
                )

    return records


def build_city_sets(records: list[dict]):
    """
    Map each hierarchy entity to the set of dataset cities compatible with it.
    """
    cities_by_continent: dict[str, set[str]] = defaultdict(set)
    cities_by_country: dict[str, set[str]] = defaultdict(set)
    cities_by_region: dict[str, set[str]] = defaultdict(set)

    for record in records:
        city = record["city"]

        continent = record["continent"]
        if continent is not None:
            cities_by_continent[continent].add(city)

        country = record["country"]
        if country is not None:
            cities_by_country[country].add(city)

        for region in record["regions"]:
            cities_by_region[region].add(city)

    return cities_by_continent, cities_by_country, cities_by_region


def estimate_remaining_candidates(
    records: list[dict],
) -> tuple[dict[str, float], int]:
    (
        cities_by_continent,
        cities_by_country,
        cities_by_region,
    ) = build_city_sets(records)

    all_cities = {
        record["city"]
        for record in records
    }

    continent_counts = []
    country_counts = []
    region_counts = []

    for record in records:
        continent = record["continent"]
        if continent is not None:
            continent_counts.append(
                len(cities_by_continent[continent])
            )

        country = record["country"]
        if country is not None:
            country_counts.append(
                len(cities_by_country[country])
            )

        region_candidate_counts = [
            len(cities_by_region[region])
            for region in record["regions"]
            if region in cities_by_region
        ]

        if region_candidate_counts:
            region_counts.append(
                geometric_mean(region_candidate_counts)
            )

    remaining = {
        "continent": geometric_mean(continent_counts),
        "country": geometric_mean(country_counts),
        "region": geometric_mean(region_counts),
        "city": 1.0,
        "place": 1.0,
    }

    return remaining, len(all_cities)


def compute_informativeness(
    remaining: dict[str, float],
    prior_candidates: int,
) -> dict[str, float]:
    """
    Normalized information gain relative to not knowing any location.

    The prior search space consists of all unique city-level locations
    represented in the dataset.
    """
    if prior_candidates <= 1:
        raise ValueError(
            "The dataset must contain more than one unique city."
        )

    max_information = math.log(prior_candidates)

    return {
        level: (
            math.log(prior_candidates / candidates)
            / max_information
        )
        for level, candidates in remaining.items()
    }


def derive(path):
    records = load_location_records(path)
    remaining, prior = estimate_remaining_candidates(records)
    return {"metadata_sha256": file_hash(path), "location_records": len(records),
            "prior_candidates": prior, "remaining_candidates": remaining,
            "informativeness": compute_informativeness(remaining, prior),
            "definition": "Distinct city labels across birth/death gold locations. Event-weighted geometric means of compatible distinct-city counts; regions use within-event then across-event geometric means.",
            "original_script": "uncertainty-granularity-calibration/scripts/calc_location_informativeness.py"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--output', type=Path, default=PARAMETERS_PATH)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = derive(args.metadata)
    if args.check:
        if json.loads(args.output.read_text()) != result:
            raise ValueError('Geographic informativeness parameters are stale; regenerate and rescore utilities')
    else:
        write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
