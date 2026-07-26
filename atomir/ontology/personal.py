"""The `personal` ontology pack — an OPTIONAL, opinionated set of ~18 predicates
for personal-assistant memory. Select with ONTOLOGY_PACK=personal. This is an
example pack, not a default: atomir is general-purpose and seeds nothing unless a
pack is chosen. Copy this file to build a domain pack of your own.

Format: (canonical, state_template, object_type, [aliases...], description).
"""

from __future__ import annotations

PERSONAL_SEEDS = [
    ("works_at", "works at", "organization",
     ["joined", "left", "started at", "quit", "hired at", "worked for", "employed by",
      "resigned from", "works for"], "employment relationship"),
    ("lives_in", "lives in", "place",
     ["moved to", "relocated to", "resides in", "lives at"], "place of residence"),
    ("manager_is", "reports to", "person",
     ["manager is", "managed by", "reports to", "boss is"], "who the user reports to"),
    ("relationship_with", "is in a relationship with", "person",
     ["dating", "married to", "engaged to", "partner is", "broke up with", "divorced"],
     "romantic relationship"),
    ("has_pet", "has a pet", "animal",
     ["adopted", "got a pet", "has a dog", "has a cat"], "pet ownership"),
    ("plays", "plays", "object", ["plays", "plays the"], "plays an instrument or game"),
    ("practices", "practices", "activity",
     ["took up", "does", "trains", "picked up", "started doing", "practices"],
     "a hobby or regular activity"),
    ("drives", "drives", "object", ["drives", "owns a car"], "vehicle the user drives"),
    ("speaks", "speaks", "language",
     ["speaks", "learned", "is learning"], "a language the user speaks"),
    ("reads", "reads", "media",
     ["reads", "subscribes to", "subscribed to"], "publications the user reads"),
    ("owns", "owns", "object", ["owns", "bought", "purchased"], "something the user owns"),
    ("uses", "uses", "object", ["uses", "started using"], "a tool or product the user uses"),
    ("likes", "likes", "other",
     ["likes", "enjoys", "loves", "favorite"], "something the user likes"),
    ("dislikes", "dislikes", "other",
     ["dislikes", "hates", "avoids", "can't stand"], "something the user dislikes"),
    ("member_of", "is a member of", "organization",
     ["member of", "belongs to", "joined the club"], "membership in a group"),
    ("studied_at", "studied at", "organization",
     ["studied at", "graduated from", "attends", "enrolled at"], "education history"),
    ("health_condition", "has the condition", "condition",
     ["diagnosed with", "suffers from", "allergic to", "has"], "a health condition"),
    ("goal", "has the goal", "other",
     ["wants to", "plans to", "aims to", "goal is"], "a goal or intention"),
]
