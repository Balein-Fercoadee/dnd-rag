"""rule character helpers."""

import re

from typing import Any

from .rag_config import (
    CASTER_LEVEL_IN_QUESTION,
    CHARACTER_CLASS_LEVEL_PATTERN,
    FEAT_QUALIFICATION_QUESTION,
    FEAT_COUNT_QUESTION,
    FEAT_SELECTION_LEVEL_PATTERN,
    STARTING_MONEY_QUESTION,
)

from .rule_parsing import (
    citation,
    feat_name_from_question,
    ordinal_suffix,
)

CORE_CLASSES = (
    "Barbarian",
    "Bard",
    "Cleric",
    "Druid",
    "Fighter",
    "Monk",
    "Paladin",
    "Ranger",
    "Rogue",
    "Sorcerer",
    "Wizard",
)


def _merge_overlapping_page_passages(
    passages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    passages_by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    other_passages = []
    for passage in passages:
        source = passage.get("source")
        page = passage.get("page")
        if (
            passage.get("kind") == "page_text"
            and isinstance(source, str)
            and isinstance(page, int)
            and isinstance(passage.get("text"), str)
        ):
            passages_by_page.setdefault((source, page), []).append(passage)
        else:
            other_passages.append(passage)

    merged_passages = list(other_passages)
    for page_passages in passages_by_page.values():
        chunks = [str(passage["text"]) for passage in page_passages]
        while True:
            best_match: tuple[int, int, int] | None = None
            for left_index, left in enumerate(chunks):
                for right_index, right in enumerate(chunks):
                    if left_index == right_index:
                        continue
                    max_overlap = min(len(left), len(right), 256)
                    for overlap in range(max_overlap, 39, -1):
                        if left.endswith(right[:overlap]):
                            if best_match is None or overlap > best_match[2]:
                                best_match = (left_index, right_index, overlap)
                            break
            if best_match is None:
                break
            left_index, right_index, overlap = best_match
            chunks[left_index] += chunks[right_index][overlap:]
            del chunks[right_index]

        merged_passages.extend({**page_passages[0], "text": text} for text in chunks)
    return merged_passages


def feat_class_level_prerequisite_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if FEAT_QUALIFICATION_QUESTION.search(question) is None:
        return None
    feat_name = feat_name_from_question(question)
    if feat_name is None:
        return None

    character_levels = {
        match.group("class").strip().casefold(): int(match.group("level"))
        for match in CHARACTER_CLASS_LEVEL_PATTERN.finditer(question)
    }
    if not character_levels:
        return None

    character_level = sum(character_levels.values())
    selection_match = FEAT_SELECTION_LEVEL_PATTERN.search(question)
    selection_level = (
        int(selection_match.group("level"))
        if selection_match is not None
        else character_level
    )
    if selection_level < character_level:
        return None
    levels_available = selection_level - character_level

    feat_heading = re.compile(r"(?P<name>[A-Z][A-Z'’ -]*?)\s*\[[A-Z][A-Z ]*\]")
    next_feat_heading = re.compile(r"\b[A-Z][A-Z'’ -]{2,}\s+\[[A-Z ]+\]")
    prerequisite_pattern = re.compile(
        r"\bPrerequisites?\s*:\s*(?P<requirements>.*?)"
        r"(?=\.|\b(?:Benefit|Normal|Special)\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    class_requirement_pattern = re.compile(
        r"(?:^|[,;]\s*|\band\s+)"
        r"(?P<class>[a-z][a-z'’-]*(?:\s+[a-z][a-z'’-]*){0,2})\s+level\s+"
        r"(?P<level>\d{1,2})(?:st|nd|rd|th)?\b",
        re.IGNORECASE,
    )
    proficiency_requirement_pattern = re.compile(
        r"\bproficiency with (?:the )?selected weapon\b",
        re.IGNORECASE,
    )
    selected_feat_pattern = re.compile(
        r"\balready selected (?:the )?(?P<name>[\w][\w'’-]*(?:\s+[\w][\w'’-]*)*?)\s+feat\b",
        re.IGNORECASE,
    )
    selected_weapon_feat_pattern = re.compile(
        r"\b(?P<name>[A-Z][\w'’-]*\s+[A-Z][\w'’-]*"
        r"(?:\s+[A-Z][\w'’-]*){0,1})"
        r"\s+with selected weapon\b"
    )

    for passage in _merge_overlapping_page_passages(passages):
        text = passage.get("text")
        if not isinstance(text, str):
            continue
        heading_match = next(
            (
                match
                for match in feat_heading.finditer(text)
                if re.sub(r"\s+", " ", match.group("name").strip()).casefold()
                == feat_name.casefold()
            ),
            None,
        )
        if heading_match is None:
            continue
        entry = text[heading_match.end() :]
        next_heading = next_feat_heading.search(entry)
        if next_heading is not None:
            entry = entry[: next_heading.start()]
        prerequisite_match = prerequisite_pattern.search(entry)
        if prerequisite_match is None:
            continue

        requirements = (
            re.sub(
                r"\s+",
                " ",
                prerequisite_match.group("requirements"),
            )
            .strip()
            .rstrip(".")
        )
        caster_requirement = re.search(
            r"\bcaster\s+level\s*(?P<level>\d{1,2})(?:st|nd|rd|th)?\b",
            requirements,
            re.IGNORECASE,
        )
        if caster_requirement is not None:
            required_level = int(caster_requirement.group("level"))
            caster_level_match = CASTER_LEVEL_IN_QUESTION.search(question)
            if caster_level_match is None:
                return (
                    f"Cannot establish from the details given. {feat_name} requires "
                    f"caster level {required_level}{ordinal_suffix(required_level)}; "
                    "the stated class level and character level do not establish a "
                    f"caster level. {citation(passage)}"
                )
            caster_level = int(caster_level_match.group("level"))
            if caster_level < required_level:
                return (
                    f"No. {feat_name} requires caster level "
                    f"{required_level}{ordinal_suffix(required_level)}, but the "
                    f"stated caster level is {caster_level}. {citation(passage)}"
                )

        class_requirements = [
            match
            for match in class_requirement_pattern.finditer(requirements)
            if match.group("class").strip().casefold() != "caster"
        ]
        for requirement_match in class_requirements:
            required_class = requirement_match.group("class").strip().casefold()
            required_level = int(requirement_match.group("level"))
            current_level = character_levels.get(required_class, 0)
            maximum_level = current_level + levels_available
            if maximum_level >= required_level:
                continue

            described_levels = ", ".join(
                f"{level} {class_name.title()}{' level' if level == 1 else ' levels'}"
                for class_name, level in character_levels.items()
            )
            missing_levels = required_level - maximum_level
            listed_requirements = []
            if proficiency_requirement_pattern.search(requirements):
                listed_requirements.append("proficiency with the selected weapon")
            selected_feat_names = {
                match.group("name").strip()
                for match in selected_feat_pattern.finditer(entry)
            }
            selected_feat_names.update(
                match.group("name").strip()
                for match in selected_weapon_feat_pattern.finditer(requirements)
            )
            listed_requirements.extend(
                f"{name} feat" for name in sorted(selected_feat_names)
            )
            listed_requirements.extend(
                f"{match.group('class').strip().title()} level "
                f"{match.group('level')}{ordinal_suffix(int(match.group('level')))}"
                for match in class_requirements
            )
            requirement_summary = ", ".join(listed_requirements)
            other_requirements_note = (
                " The question does not establish whether the character meets the "
                "other listed prerequisites."
                if len(listed_requirements) > 1
                else ""
            )
            return (
                f"No. {feat_name} requires {requirement_summary}. The described "
                f"character has {described_levels}; by character level "
                f"{selection_level}, they could have at most {maximum_level} "
                f"{required_class.title()} "
                f"{'level' if maximum_level == 1 else 'levels'}, which is "
                f"{missing_levels} short of the required {required_class.title()} "
                f"level {required_level}.{other_requirements_note} "
                f"{citation(passage)}"
            )

        missing_requirements = []
        if (
            proficiency_requirement_pattern.search(requirements)
            and re.search(
                r"\b(?:proficient|proficiency)\b.{0,60}\bweapon\b",
                question,
                re.IGNORECASE,
            )
            is None
        ):
            missing_requirements.append("proficiency with the selected weapon")
        required_feats = {
            match.group("name").strip()
            for match in selected_weapon_feat_pattern.finditer(requirements)
        }
        required_feats.update(
            match.group("name").strip()
            for match in selected_feat_pattern.finditer(entry)
        )
        missing_requirements.extend(
            f"{name} feat"
            for name in sorted(required_feats)
            if re.search(rf"\b{re.escape(name)}\b", question, re.IGNORECASE) is None
        )
        if missing_requirements and class_requirements:
            satisfied_class_requirements = ", ".join(
                f"{match.group('class').strip().title()} level "
                f"{match.group('level')}{ordinal_suffix(int(match.group('level')))}"
                for match in class_requirements
            )
            return (
                f"Cannot establish whether the character can take {feat_name}. "
                f"The stated class level meets {satisfied_class_requirements}, "
                f"but the question does not establish "
                f"{', '.join(missing_requirements)}. {citation(passage)}"
            )
    return None


def _multiclass_feat_count_answer(
    class_levels: list[tuple[str, int]],
    passages: list[dict[str, Any]],
    *,
    is_human: bool,
) -> str | None:
    if is_human and not any(
        isinstance(passage.get("text"), str)
        and re.search(
            r"\b1 extra feat at 1st level\b|"
            r"\bhuman.{0,160}\b(?:one|1)\s+(?:extra|bonus) feat\b|"
            r"\b(?:extra|bonus) feat.{0,100}\bhuman\b",
            passage["text"],
            re.IGNORECASE | re.DOTALL,
        )
        for passage in passages
    ):
        return None

    class_bonus_counts: list[tuple[str, int]] = []
    citations_by_source: dict[str, list[str]] = {}
    for class_name, class_level in class_levels:
        single_class_question = (
            f"How many feats does a {class_level}{ordinal_suffix(class_level)} "
            f"level {class_name} get?"
        )
        answer = structured_feat_count_answer(single_class_question, passages)
        if answer is None:
            return None

        bonus_match = re.search(
            rf",\s*(?P<count>\d+)\s+{re.escape(class_name)} bonus feats?\b",
            answer,
            re.IGNORECASE,
        )
        if bonus_match is None:
            return None
        class_bonus_counts.append((class_name, int(bonus_match.group("count"))))

        citation_match = re.search(
            r"\[(?P<source>.*?), PDF pp?\. (?P<pages>[\d, ]+)\]",
            answer,
        )
        if citation_match is None:
            return None
        pages = citations_by_source.setdefault(citation_match.group("source"), [])
        pages.extend(
            page.strip()
            for page in citation_match.group("pages").split(",")
            if page.strip() not in pages
        )

    character_level = sum(level for _, level in class_levels)
    character_level_feats = 1 + character_level // 3
    human_bonus_feats = int(is_human)
    total_feats = (
        character_level_feats
        + sum(count for _, count in class_bonus_counts)
        + human_bonus_feats
    )
    breakdown = [
        f"{character_level_feats} character-level "
        f"{'feat' if character_level_feats == 1 else 'feats'}"
    ]
    breakdown.extend(
        f"{count} {class_name} bonus {'feat' if count == 1 else 'feats'}"
        for class_name, count in class_bonus_counts
    )
    if is_human:
        breakdown.append("1 Human bonus feat")
    citation_text = " ".join(
        f"[{source}, PDF pp. {', '.join(pages)}]"
        for source, pages in citations_by_source.items()
    )
    return (
        f"A {character_level}{ordinal_suffix(character_level)}-level "
        f"{'human ' if is_human else ''}multiclass character gets "
        f"{total_feats} feats: {', '.join(breakdown[:-1])}, and "
        f"{breakdown[-1]}. {citation_text}"
    )


def structured_feat_count_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if FEAT_COUNT_QUESTION.search(question) is None:
        return None

    requested_classes = [
        class_name
        for class_name in CORE_CLASSES
        if re.search(rf"\b{re.escape(class_name)}\b", question, re.IGNORECASE)
    ]
    if not requested_classes:
        return None
    if len(requested_classes) > 1:
        class_level_matches: list[tuple[int, str, int]] = []
        for class_name in requested_classes:
            class_pattern = re.escape(class_name)
            patterns = (
                re.compile(
                    r"\b(?P<level>\d{1,2})(?:st|nd|rd|th)?[\s-]+level\s+"
                    rf"(?:[a-z][a-z-]*\s+)?{class_pattern}\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    rf"\b{class_pattern}\s+(?:at\s+)?(?:class\s+)?level\s+"
                    r"(?P<level>\d{1,2})(?:st|nd|rd|th)?\b",
                    re.IGNORECASE,
                ),
            )
            match = next(
                (match for pattern in patterns if (match := pattern.search(question))),
                None,
            )
            if match is not None:
                class_level_matches.append(
                    (match.start(), class_name, int(match.group("level")))
                )
        class_levels = [
            (class_name, level) for _, class_name, level in sorted(class_level_matches)
        ]
        if (
            len(class_levels) != len(requested_classes)
            or len({class_name for class_name, _ in class_levels}) != len(class_levels)
            or any(level < 1 for _, level in class_levels)
        ):
            return None
        is_human = re.search(r"\bhuman\b", question, re.IGNORECASE) is not None
        return _multiclass_feat_count_answer(
            class_levels,
            passages,
            is_human=is_human,
        )
    class_name = requested_classes[0]
    class_pattern = re.escape(class_name)
    level_patterns = (
        re.compile(
            r"\b(?P<level>\d{1,2})(?:st|nd|rd|th)?[\s-]+level\s+"
            rf"(?:[a-z][a-z-]*\s+)?{class_pattern}\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:[a-z][a-z-]*\s+)?{class_pattern}\s+"
            r"(?:at\s+)?(?:class\s+)?level\s+"
            r"(?P<level>\d{1,2})(?:st|nd|rd|th)?\b",
            re.IGNORECASE,
        ),
    )
    level_match = next(
        (match for pattern in level_patterns if (match := pattern.search(question))),
        None,
    )
    if level_match is None:
        return None

    class_level = int(level_match.group("level"))
    if class_level < 1:
        return None
    is_human = re.search(r"\bhuman\b", question, re.IGNORECASE) is not None

    passages_by_source: dict[str, list[dict[str, Any]]] = {}
    for passage in passages:
        source = passage.get("source")
        if isinstance(source, str) and isinstance(passage.get("text"), str):
            passages_by_source.setdefault(source, []).append(passage)

    for source, source_passages in passages_by_source.items():
        compact_text = re.sub(
            r"\s+",
            " ",
            " ".join(str(passage["text"]) for passage in source_passages),
        )
        character_feat_table = re.search(
            r"\bTable 3[–-]2: Experience and Level-Dependent Benefits\b"
            r".{0,1200}\bFeats\b",
            compact_text,
            re.IGNORECASE,
        )
        has_character_feat_table = (
            character_feat_table is not None
            and re.search(
                rf"\b{class_level}{ordinal_suffix(class_level)}\b",
                compact_text,
                re.IGNORECASE,
            )
            is not None
        )
        has_general_progression = (
            re.search(
                r"\bgets every three levels\b|"
                r"\bfeats?\b.{0,80}\bevery three levels\b|"
                r"\bevery three levels\b.{0,80}\bfeats?\b|"
                r"\bfeats?\b.{0,80}\bevery third (?:character )?level\b|"
                r"\bevery third (?:character )?level\b.{0,80}\bfeats?\b",
                compact_text,
                re.IGNORECASE,
            )
            is not None
            or has_character_feat_table
        )
        has_first_character_feat = (
            re.search(
                r"\bany 1st-level character gets\b|"
                r"\bcharacters? (?:get|gets|gain|gains|receive|receives) "
                r"(?:one|a) feat at 1st level\b",
                compact_text,
                re.IGNORECASE,
            )
            is not None
            or has_character_feat_table
        )
        has_human_bonus = re.search(
            r"\bbonus\b.{0,300}\bgranted to a human character\b|"
            r"\b1 extra feat at 1st level\b|"
            r"\bhuman.{0,160}\b(?:one|1)\s+(?:extra|bonus) feat\b|"
            r"\b(?:extra|bonus) feat.{0,100}\bhuman\b",
            compact_text,
            re.IGNORECASE | re.DOTALL,
        )
        class_level_patterns = (
            re.compile(
                rf"\bat\s+(?P<level>\d{{1,2}})(?:st|nd|rd|th)?[\s-]+level\b"
                rf".{{0,120}}\b{class_pattern}\b.{{0,120}}\bbonus\b.{{0,40}}\bfeat\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\b{class_pattern}\b.{{0,120}}\bat\s+"
                r"(?P<level>\d{1,2})(?:st|nd|rd|th)?[\s-]+level\b"
                r".{0,120}\bbonus\b.{0,40}\bfeat\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\b{class_pattern}\b.{{0,120}}\bbonus\b.{{0,40}}\bfeat\b"
                rf".{{0,120}}\bat\s+(?P<level>\d{{1,2}})(?:st|nd|rd|th)?"
                r"[\s-]+level\b",
                re.IGNORECASE,
            ),
        )
        class_rule_sentences = [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+", compact_text)
            if re.search(rf"\b{class_pattern}\b", sentence, re.IGNORECASE)
            and re.search(r"\bbonus\b.{0,80}\bfeat\b", sentence, re.IGNORECASE)
        ]
        class_grant_levels = {
            int(match.group("level"))
            for pattern in class_level_patterns
            for sentence in class_rule_sentences
            for match in pattern.finditer(sentence)
        }
        wizard_table_levels: set[int] = set()
        if class_name == "Wizard":
            wizard_row_pattern = re.compile(
                r"\b(?P<level>\d{1,2})(?:st|nd|rd|th)\b"
                r"(?P<row>(?:(?!\b\d{1,2}(?:st|nd|rd|th)\b).){0,300})",
                re.IGNORECASE | re.DOTALL,
            )
            for passage in source_passages:
                passage_text = str(passage["text"])
                table_match = re.search(
                    r"\bTable 3[–-]18: The Wizard\b",
                    passage_text,
                    re.IGNORECASE,
                )
                if table_match is None:
                    continue
                table_text = passage_text[table_match.start() :]
                wizard_table_levels.update(
                    int(match.group("level"))
                    for match in wizard_row_pattern.finditer(table_text)
                    if re.search(
                        r"\bScribe Scroll\b|\bBonus feat\b",
                        match.group("row"),
                        re.IGNORECASE,
                    )
                )
            class_grant_levels.update(wizard_table_levels)
        has_wizard_class_table = (
            class_name == "Wizard"
            and bool(wizard_table_levels)
            and {
                1,
                5,
                10,
                15,
                20,
            }.issubset(wizard_table_levels)
        )

        number_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        interval_pattern = re.compile(
            rf"\bevery\s+(?P<interval>\d{{1,2}}|"
            rf"{'|'.join(number_words)})\b.{{0,80}}\blevels?\s+thereafter\b",
            re.IGNORECASE,
        )
        class_intervals = {
            (
                int(match.group("interval"))
                if match.group("interval").isdigit()
                else number_words[match.group("interval").casefold()]
            )
            for sentence in class_rule_sentences
            for match in interval_pattern.finditer(sentence)
        }

        if len(class_intervals) > 1:
            continue
        class_bonus_levels = set(class_grant_levels)
        class_interval = next(iter(class_intervals), None)
        if class_interval is not None and class_interval > 0:
            class_bonus_levels.update(
                range(class_interval, class_level + 1, class_interval)
            )
        class_progression_evidence = (
            class_interval is not None or has_wizard_class_table
        )

        if not all(
            (
                class_bonus_levels,
                class_progression_evidence,
                has_general_progression,
                has_first_character_feat,
            )
        ):
            continue
        if is_human and has_human_bonus is None:
            continue

        character_level_feats = 1 + class_level // 3
        class_bonus_feats = sum(
            bonus_level <= class_level for bonus_level in class_bonus_levels
        )
        human_bonus_feats = int(is_human)
        total_feats = character_level_feats + class_bonus_feats + human_bonus_feats
        breakdown = (
            f"{character_level_feats} character-level "
            f"{'feat' if character_level_feats == 1 else 'feats'}, "
            f"{class_bonus_feats} {class_name} bonus "
            f"{'feat' if class_bonus_feats == 1 else 'feats'}"
        )
        if is_human:
            breakdown += ", and 1 Human bonus feat"
        evidence_patterns = (
            *(pattern.pattern for pattern in class_level_patterns),
            interval_pattern.pattern,
            r"\bTable 3[–-]18: The Wizard\b",
            r"\bgets every three levels\b",
            r"\bfeats?\b.{0,80}\bevery three levels\b",
            r"\bevery three levels\b.{0,80}\bfeats?\b",
            r"\bfeats?\b.{0,80}\bevery third (?:character )?level\b",
            r"\bevery third (?:character )?level\b.{0,80}\bfeats?\b",
            r"\bany 1st-level character gets\b",
            r"\bcharacters? (?:get|gets|gain|gains|receive|receives) "
            r"(?:one|a) feat at 1st level\b",
            r"\bgranted to a human character\b",
            r"\b1 extra feat at 1st level\b",
            r"\bhuman.{0,100}\b(?:one )?(?:extra|bonus) feat\b",
            r"\bhuman.{0,160}\b(?:one|1)\s+(?:extra|bonus) feat\b",
            r"\b(?:extra|bonus) feat.{0,100}\bhuman\b",
            r"\bTable 3[–-]2: Experience and Level-Dependent Benefits\b"
            r".{0,1200}\bFeats\b",
        )
        pages = list(
            dict.fromkeys(
                str(passage["page"])
                for passage in source_passages
                if passage.get("page") is not None
                and any(
                    re.search(pattern, str(passage["text"]), re.IGNORECASE)
                    for pattern in evidence_patterns
                )
            )
        )
        page_citation = (
            f"[{source}, PDF p. {pages[0]}]"
            if len(pages) == 1
            else f"[{source}, PDF pp. {', '.join(pages)}]"
        )
        return (
            f"A {class_level}{ordinal_suffix(class_level)}-level "
            f"{'human ' if is_human else ''}{class_name} gets {total_feats} feats: "
            f"{breakdown}. {page_citation}"
        )

    return (
        "Cannot establish the total feat count because the retrieved passages "
        "do not show the character-level and class bonus-feat progressions"
        + (" and the Human bonus feat." if is_human else ".")
    )


structured_fighter_feat_count_answer = structured_feat_count_answer


def structured_starting_money_answer(
    question: str, passages: list[dict[str, Any]]
) -> str | None:
    if STARTING_MONEY_QUESTION.search(question) is None:
        return None

    requested_classes = [
        class_name
        for class_name in CORE_CLASSES
        if re.search(rf"\b{class_name}\b", question, re.IGNORECASE)
    ]
    if len(requested_classes) != 1:
        return None
    requested_class = requested_classes[0]
    class_row_pattern = re.compile(
        rf"\b{re.escape(requested_class)}\b[^\d\n]{{0,40}}"
        r"(?P<count>\d+)\s*d\s*(?P<sides>\d+)"
        r"(?:\s*[x×]\s*(?P<multiplier>\d+)\b)?",
        re.IGNORECASE,
    )

    for passage in passages:
        text = passage.get("text")
        if (
            not isinstance(text, str)
            or re.search(r"\bstarting\s+(?:gold|money)\b", text, re.IGNORECASE) is None
        ):
            continue
        match = class_row_pattern.search(text)
        if match is None:
            continue

        count = int(match.group("count"))
        sides = int(match.group("sides"))
        multiplier_match = match.group("multiplier")
        if multiplier_match is None:
            average_match = re.match(
                r"\s*\((?P<gp>\d+)\s*gp" r"(?:\s*,\s*(?P<sp>\d+)\s*sp)?\)",
                text[match.end() :],
                re.IGNORECASE,
            )
            if average_match is None:
                continue
            printed_average = int(average_match.group("gp")) + (
                int(average_match.group("sp") or 0) / 10
            )
            base_average = count * (sides + 1) / 2
            inferred_multiplier = printed_average / base_average
            if not inferred_multiplier.is_integer():
                continue
            multiplier = int(inferred_multiplier)
        else:
            multiplier = int(multiplier_match)

        minimum = count * multiplier
        maximum = count * sides * multiplier
        average = count * (sides + 1) * multiplier / 2
        dice_expression = f"{count}d{sides}"
        if multiplier != 1:
            dice_expression += f" × {multiplier}"
        return (
            f"{requested_class} starting money is {dice_expression} gp: "
            f"{minimum}–{maximum} gp "
            f"(average {average:g} gp). {citation(passage)}"
        )

    return (
        f"Cannot establish the starting money for a {requested_class}; "
        "the matching starting-gold table row was not retrieved."
    )
