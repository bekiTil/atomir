"""Episodic memory layer (additive, feature-flagged by EPISODIC_ENABLED).

Events (things that happened) live in an append-mostly log, organized into
per-entity, per-verb chronological branches. The atomic-fact store becomes a
projection of the event log. Everything here is inert unless the flag is on;
with it off, the package behaves exactly as the current release.
"""
