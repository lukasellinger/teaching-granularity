"""Offline date-matching and event-selection regression tests."""
import unittest
from . import (TemporalLevel, TemporalValue, Relation, Season, TemporalModifier,
               parse_temporal_matches, parse_temporal_text, strip_brackets_with_mapping,
               parse_temporal_response, is_idk_response, to_five_level, remap_stripped_span)


class TemporalTests(unittest.TestCase):
    def test_response_does_not_salvage_year_from_invalid_date(self):
        parsed = parse_temporal_response('She was born on March 99, 1962.', 'birth')
        self.assertEqual(parsed.status, 'unparsed')
        self.assertIsNone(parsed.value)
        self.assertEqual(parsed.matched_values, ())

    def test_production_date_shape_regressions(self):
        for text in [
            'He was born on November 41, 1683.',
            'He died on December 61, 1958. This was **December 30, 1958**.',
        ]:
            relation = 'birth' if 'born' in text else 'death'
            self.assertEqual(parse_temporal_response(text, relation).status, 'unparsed')
        text = ('He was born on **November 19 (October 9, according to the old-style '
                'Julian calendar)** **1711**.')
        result = parse_temporal_response(text, 'birth')
        self.assertEqual(result.status, 'parsed')
        self.assertEqual(result.value.year, 1711)
        self.assertEqual(result.value.level, TemporalLevel.YEAR)

    def test_response_accepts_relation_strings_and_rejects_unknown_relations(self):
        self.assertEqual(parse_temporal_response('1900', 'birth').value.year, 1900)
        with self.assertRaises(ValueError):
            parse_temporal_response('1900', 'marriage')

    def test_five_level_projection_keeps_the_value_and_removes_intermediate_detail(self):
        for value, level in [
            (TemporalValue(TemporalLevel.SEASON, year=1900, season=Season.SPRING), TemporalLevel.YEAR),
            (TemporalValue(TemporalLevel.EARLYMIDYEAR, year=1900, year_modifier=TemporalModifier.EARLY), TemporalLevel.YEAR),
            (TemporalValue(TemporalLevel.EARLYMIDDECADE, decade=1960, decade_modifier=TemporalModifier.MID), TemporalLevel.DECADE),
            (TemporalValue(TemporalLevel.EARLYMIDCENTURY, century=20, century_modifier=TemporalModifier.LATE), TemporalLevel.CENTURY),
        ]:
            self.assertEqual(to_five_level(value).level, level)
            self.assertTrue(value.consistent_with(to_five_level(value)))

    def test_projection_cannot_invent_missing_details(self):
        value = TemporalValue(TemporalLevel.DAY, year=1900, month=3, day=24)
        self.assertEqual(value.to_level(TemporalLevel.YEAR).to_text(), '1900')
        with self.assertRaises(ValueError):
            value.to_level(TemporalLevel.YEAR).to_level(TemporalLevel.DAY)

    def test_century_boundaries_and_bce_rendering(self):
        for year, century in [(1900,19),(1901,20),(-100,-1),(-101,-2)]:
            value = TemporalValue(TemporalLevel.YEAR, year=year)
            self.assertEqual(value.century, century)
        self.assertEqual(TemporalValue(TemporalLevel.YEAR, year=-469).to_text(), '469 BCE')

    def test_remapping_checks_bounds(self):
        text = 'Born (possibly) in 1900.'
        stripped, _ = strip_brackets_with_mapping(text)
        start = stripped.index('1900')
        a,b = remap_stripped_span(text,start,start+4)
        self.assertEqual(text[a:b], '1900')
        with self.assertRaises(ValueError):
            remap_stripped_span(text,-1,3)

    def test_empty_and_non_temporal_answers_are_not_explicit_abstentions(self):
        for answer in ['', 'She was born in Paris.', 'I cannot give an exact date.']:
            self.assertEqual(parse_temporal_response(answer,'birth').status,'unparsed')
        self.assertEqual(parse_temporal_response('**IDK.**','birth').status,'idk')

    def test_parser_reports_invalid_year_without_correcting_or_inventing_a_value(self):
        text = "Born on March 24, 19621, in Madras."
        parsed = parse_temporal_text(text)
        assert parsed.matches == ()
        assert len(parsed.issues) == 1
        issue = parsed.issues[0]
        assert text[issue.start:issue.end] == "March 24, 19621"
        assert issue.status == "invalid"
        assert issue.reason == "year_exceeds_supported_width"
        assert parsed.has_temporal_content

    def test_parser_distinguishes_incomplete_date_from_no_date(self):
        parsed = parse_temporal_text("Born on March 24.")
        assert not parsed.matches and parsed.issues[0].status == "incomplete"
        assert not parse_temporal_text("Born in Madras.").has_temporal_content

    def test_diagnostics_do_not_change_valid_date_matching(self):
        text = "Born on March 24, 1962, in Madras."
        parsed = parse_temporal_text(text)
        assert parsed.matches == tuple(parse_temporal_matches(text))
        assert not parsed.issues
        for text in ("Born on March 24 in 1962.", "Born on 24 March in 1962.", "Born in March 24 BCE."):
            parsed = parse_temporal_text(text)
            assert parsed.matches == tuple(parse_temporal_matches(text))
            assert not parsed.issues

    def test_invalid_textual_date_does_not_leave_a_valid_bare_year(self):
        parsed = parse_temporal_text("Born on March 99, 1962.")
        assert not parsed.matches
        assert parsed.issues[0].reason == "day_out_of_range"

    def test_numeric_date_diagnostics_and_optional_bracket_scope(self):
        assert parse_temporal_text("3/4/1935").issues[0].status == "ambiguous"
        assert parse_temporal_text("36/34/384").issues[0].status == "invalid"
        assert not parse_temporal_text("Born (March 24, 19621).").has_temporal_content
        assert parse_temporal_text("Born (March 24, 19621).", include_brackets=True).issues

    def test_parser_returns_original_coordinates_after_removing_parentheses(self):
        answer = "Rodin (1840–1917) was born on June 1, 1900 in Paris."

        matches = parse_temporal_matches(answer)

        assert len(matches) == 1
        match = matches[0]
        assert answer[match.start:match.end] == "June 1, 1900"
        assert match.value.level is TemporalLevel.DAY

    def test_parser_does_not_treat_two_digit_street_number_as_year(self):
        answer = "John Donne was born at 10 Broad Street in Canterbury."

        assert parse_temporal_matches(answer) == []

    def test_parser_keeps_multiple_real_dates_explicit(self):
        answer = "She was born in 1900 and moved in 1920."

        matches = parse_temporal_matches(answer)

        assert [match.value.year for match in matches] == [1900, 1920]

    def test_case_insensitive_matching_does_not_change_unicode_offsets(self):
        answer = "İstanbul records say she was born on MAY 1, 1900."

        matches = parse_temporal_matches(answer)

        assert len(matches) == 1
        assert answer[matches[0].start:matches[0].end] == "MAY 1, 1900"

    def test_bracket_stripping_normalizes_text_and_preserves_mapping(self):
        source = "  Born (approximately 1899)\t in  1900 , in Paris.  "

        stripped, mapping = strip_brackets_with_mapping(source)

        assert stripped == "Born in 1900, in Paris."
        year_start = stripped.index("1900")
        original_start = mapping[year_start][0]
        original_end = mapping[year_start + 3][1]
        assert source[original_start:original_end] == "1900"

    def test_month_of_the_year_is_one_temporal_span(self):
        text = "Zheng He died in December of the year 1433."

        matches = parse_temporal_matches(text)

        assert len(matches) == 1
        assert text[matches[0].start:matches[0].end] == (
            "December of the year 1433"
        )
        assert matches[0].value.level is TemporalLevel.MONTH
        assert matches[0].value.year == 1433
        assert matches[0].value.month == 12

    def test_season_of_the_year_is_one_temporal_span(self):
        text = "She was born in winter of the year 1822."

        matches = parse_temporal_matches(text)

        assert len(matches) == 1
        assert text[matches[0].start:matches[0].end] == (
            "winter of the year 1822"
        )
        assert matches[0].value.level is TemporalLevel.SEASON

    def test_modifier_in_the_year_is_one_temporal_span(self):
        text = "The record places the event late in the year 149 BC."

        matches = parse_temporal_matches(text)

        assert len(matches) == 1
        assert text[matches[0].start:matches[0].end] == "late in the year 149 BC."
        assert matches[0].value.level is TemporalLevel.EARLYMIDYEAR

    def test_modifier_in_the_century_is_one_temporal_span(self):
        text = "He was born early in the 19th century."

        matches = parse_temporal_matches(text)

        assert len(matches) == 1
        assert text[matches[0].start:matches[0].end] == (
            "early in the 19th century"
        )
        assert matches[0].value.level is TemporalLevel.EARLYMIDCENTURY

    def test_month_day_in_year_is_parsed_as_one_day(self):
        text = "He was born on February 21 in 1500."

        matches = parse_temporal_matches(text)

        assert len(matches) == 1
        assert text[matches[0].start:matches[0].end] == "February 21 in 1500"
        assert matches[0].value.level is TemporalLevel.DAY

    def test_day_month_in_year_is_parsed_as_one_day(self):
        text = "He was born on the 23rd of March in 1777."

        matches = parse_temporal_matches(text)

        assert len(matches) == 1
        assert text[matches[0].start:matches[0].end] == (
            "23rd of March in 1777"
        )
        assert matches[0].value.level is TemporalLevel.DAY

    def test_unambiguous_slash_dates_are_parsed(self):
        day_first = parse_temporal_matches("He was born on 20/8/1904.")
        month_first = parse_temporal_matches("He was born on 8/16/1935.")

        assert len(day_first) == 1
        assert day_first[0].value.day == 20
        assert day_first[0].value.month == 8
        assert len(month_first) == 1
        assert month_first[0].value.day == 16
        assert month_first[0].value.month == 8

    def test_ambiguous_or_invalid_slash_dates_do_not_fall_back_to_year(self):
        assert parse_temporal_matches("The date was 3/4/1935.") == []
        assert parse_temporal_matches("The date was 36/34/384.") == []

    def test_short_year_requires_an_explicit_supported_era(self):
        matches = parse_temporal_matches("Cai Lun was born around 50 AD.")

        assert len(matches) == 1
        assert matches[0].value.level is TemporalLevel.YEAR
        assert matches[0].value.year == 50
        assert parse_temporal_matches("He returned on July 15.") == []

    def test_islamic_year_is_not_misread_as_common_era(self):
        matches = parse_temporal_matches(
            "He died in 1111 CE, corresponding to 505 AH."
        )

        assert [match.value.year for match in matches] == [1111]

    def test_duration_is_not_misread_as_calendar_year(self):
        matches = parse_temporal_matches(
            "He died approximately 400 years after his teacher's death."
        )

        assert matches == []

    def test_trailing_era_applies_to_both_range_endpoints(self):
        matches = parse_temporal_matches("Estimates range from 600 to 576 BCE.")

        assert [match.value.year for match in matches] == [-600, -576]

    def test_or_connected_range_inherits_trailing_era(self):
        matches = parse_temporal_matches("around 470 or 469 BCE")

        assert [match.value.year for match in matches] == [-470, -469]

    def test_year_range_returns_both_endpoints(self):
        matches = parse_temporal_matches("1905-1906")

        assert [match.value.year for match in matches] == [1905, 1906]

    def test_coordinated_months_inherit_year(self):
        matches = parse_temporal_matches("May or June 1966")

        assert [(match.value.year, match.value.month) for match in matches] == [
            (1966, 5),
            (1966, 6),
        ]

    def test_temporal_parser_selects_requested_event_not_first_date(self):
        response = "She was born in 1900 and died in 1950."

        parsed = parse_temporal_response(response, Relation.DEATH)

        assert parsed.status == "parsed"
        assert parsed.value.year == 1950

    def test_temporal_parser_rejects_distinct_multiple_requested_values(self):
        response = "Sources say she was born in 1900 or born in 1901."

        parsed = parse_temporal_response(response, Relation.BIRTH)

        assert parsed.status == "multiple"
        assert parsed.value is None
        assert {value.year for value in parsed.matched_values} == {1900, 1901}

    def test_temporal_parser_ignores_dates_of_later_narrative_events(self):
        responses = [
            (
                "King Juan Carlos I of Spain passed away on **November 19, "
                "2014**, at the age of 87.\n\nHe died at the Royal Palace of "
                "Zarzuela in Madrid, surrounded by his family. His death was "
                "announced by the Spanish royal family later that day, marking "
                "the end of a reign that lasted from 1975 until his abdication "
                "in 2014."
            ),
            (
                "Muhammad Ali Jinnah, the founder of Pakistan, passed away on "
                "**14 December 1948**. He was 89 years old at the time of his "
                "death. His death marked the end of a pivotal era in South Asian "
                "history, as he had been instrumental in the creation of "
                "Pakistan in 1947."
            ),
        ]

        parsed = [
            parse_temporal_response(response, Relation.DEATH)
            for response in responses
        ]

        assert [result.status for result in parsed] == ["parsed", "parsed"]
        assert [result.value.level for result in parsed] == [
            TemporalLevel.DAY,
            TemporalLevel.DAY,
        ]
        assert [result.value.year for result in parsed] == [2014, 1948]

    def test_temporal_parser_scopes_later_biographical_dates(self):
        responses = [
            (
                "Wassily Kandinsky died on December 13, 1944. He was living in "
                "exile, having fled Germany in 1933."
            ),
            (
                "Auguste Comte died on September 5, 1857. His death occurred "
                "shortly after the death of his wife, who had died in 1856."
            ),
            (
                "Slobodan Milošević died on March 11, 2006. His trial was set "
                "to begin on March 16, 2006."
            ),
        ]

        parsed = [
            parse_temporal_response(response, Relation.DEATH)
            for response in responses
        ]

        assert [result.status for result in parsed] == ["parsed"] * 3
        assert [result.value.year for result in parsed] == [1944, 1857, 2006]

    def test_later_explicit_requested_event_alternative_remains_multiple(self):
        response = (
            "Epicurus was born in 341 BC. His birth year is traditionally placed "
            "in 341 BC, though some estimates suggest it could be 342 BC."
        )

        parsed = parse_temporal_response(response, Relation.BIRTH)

        assert parsed.status == "multiple"
        assert {value.year for value in parsed.matched_values} == {-341, -342}

    def test_referential_later_year_alternatives_remain_multiple(self):
        response = (
            "Historians estimate his birth year as 575 AD, based on the age he "
            "would have been when he ascended to the throne in 610 AD. Some "
            "sources suggest a slightly later date, such as 576 or 577."
        )

        parsed = parse_temporal_response(response, Relation.BIRTH)

        assert parsed.status == "multiple"
        assert {value.year for value in parsed.matched_values} == {575, 576, 577}

    def test_explicit_between_range_remains_multiple(self):
        response = (
            "Historians place his birth in the early 5th century, likely between "
            "400 and 410 AD."
        )

        parsed = parse_temporal_response(response, Relation.BIRTH)

        assert parsed.status == "multiple"
        assert {
            value.year for value in parsed.matched_values if value.year is not None
        } == {400, 410}

    def test_hyphenated_occupation_period_is_not_a_birth_date(self):
        response = (
            "The exact birth date of Tulsidas, the renowned 16th-century poet, "
            "is unknown. Historians place his birth around 1511 CE."
        )

        parsed = parse_temporal_response(response, Relation.BIRTH)

        assert parsed.status == "parsed"
        assert parsed.value.year == 1511

    def test_concise_standalone_temporal_answer_is_parsed(self):
        responses = [
            "around 1401",
            "Approximately 573 CE in Mecca",
            "Correct aggregated answer: July 26, 1946\n"
            "Explanation: This is the aggregate.",
        ]

        parsed = [
            parse_temporal_response(response, Relation.BIRTH)
            for response in responses
        ]

        assert [result.status for result in parsed] == ["parsed"] * 3
        assert [result.value.year for result in parsed] == [1401, 573, 1946]

    def test_standalone_temporal_alternatives_remain_multiple(self):
        responses = [
            "Athens, Greece, around 470 or 469 BCE",
            "December 1923 - January 1924",
            "May or June 1966",
            "1905-1906",
        ]

        parsed = [
            parse_temporal_response(response, Relation.BIRTH)
            for response in responses
        ]

        assert [result.status for result in parsed] == ["multiple"] * 4

    def test_short_unrelated_event_is_not_a_standalone_temporal_answer(self):
        responses = [
            "He ascended to the throne in 610 AD.",
            "He fought a battle in 1900.",
        ]
        parsed = [
            parse_temporal_response(response, Relation.BIRTH)
            for response in responses
        ]

        assert [result.status for result in parsed] == ["unparsed", "unparsed"]

    def test_explicit_gregorian_equivalent_is_one_date(self):
        responses = [
            (
                "Sergei Prokofiev was born on April 23, 1891. His birth date is "
                "given in the Julian calendar; in the modern Gregorian calendar, "
                "this corresponds to May 6, 1891."
            ),
            (
                "Elizabeth died on December 25, 1762 according to the Julian "
                "calendar, which corresponds to January 5, 1763 in the modern "
                "Gregorian calendar."
            ),
            (
                "Husayn died on 10 Muharram, 61 AH. In the Gregorian calendar, "
                "this date corresponds to October 10, 680 AD."
            ),
        ]

        parsed = [
            parse_temporal_response(response, Relation.BIRTH if index == 0 else Relation.DEATH)
            for index, response in enumerate(responses)
        ]

        assert [result.status for result in parsed] == ["parsed"] * 3
        assert [result.value.year for result in parsed] == [1891, 1763, 680]
        assert [result.value.month for result in parsed] == [5, 1, 10]

    def test_common_era_date_ignores_parenthetical_ah_equivalent(self):
        parsed = parse_temporal_response(
            "Al-Ghazali died on February 16, 1111 (505 AH).",
            Relation.DEATH,
        )

        assert parsed.status == "parsed"
        assert parsed.value.year == 1111

    def test_repeated_identical_value_is_one_prediction(self):
        response = "She was born in 1900; records confirm she was born in 1900."

        parsed = parse_temporal_response(response, Relation.BIRTH)

        assert parsed.status == "parsed"
        assert parsed.value.year == 1900

    def test_compatible_finer_date_becomes_the_prediction(self):
        response = "She was born in the 1900s (specifically in 1905)."

        parsed = parse_temporal_response(response, Relation.BIRTH)

        assert parsed.status == "parsed"
        assert parsed.value.level is TemporalLevel.YEAR
        assert parsed.value.year == 1905
        assert {value.level for value in parsed.matched_values} == {
            TemporalLevel.DECADE,
            TemporalLevel.YEAR,
        }

    def test_incompatible_coarse_and_fine_dates_remain_multiple(self):
        response = (
            "She was born in 1900, although another source says she was born "
            "on January 1, 1901."
        )

        parsed = parse_temporal_response(response, Relation.BIRTH)

        assert parsed.status == "multiple"
        assert parsed.value is None
        assert {value.year for value in parsed.matched_values} == {1900, 1901}

    def test_common_hyphenated_temporal_forms_are_parsed(self):
        century = parse_temporal_response(
            "She was born in the 19th-century.",
            Relation.BIRTH,
        )
        modified_decade = parse_temporal_response(
            "She was born in the mid-1900s.",
            Relation.BIRTH,
        )

        assert century.status == "parsed"
        assert century.value.level is TemporalLevel.CENTURY
        assert modified_decade.status == "parsed"
        assert modified_decade.value.level is TemporalLevel.EARLYMIDDECADE

    def test_idk_detection_is_exact_but_allows_markdown_wrapping(self):
        assert is_idk_response("**I don't know.**")
        assert is_idk_response("I do not know!")
        assert not is_idk_response("I don't know the exact day, but it was in 1900.")


if __name__ == "__main__":
    unittest.main()
