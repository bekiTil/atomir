"""`atomir` command-line entry point.

Currently one subcommand: `atomir migrate --backfill --user <id>` synthesizes
episodic events for a user's existing facts, so a store created
before the episodic layer starts showing timelines. Non-destructive.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atomir")
    sub = parser.add_subparsers(dest="command")

    mig = sub.add_parser("migrate", help="migrate/backfill an existing store")
    mig.add_argument("--backfill", action="store_true",
                     help="synthesize events for existing facts")
    mig.add_argument("--user", required=True, help="user_id to backfill")

    args = parser.parse_args(argv)

    if args.command == "migrate" and args.backfill:
        from atomir.assembly import build_memory_service

        service = build_memory_service()
        if service.episodic is None:
            print("EPISODIC_ENABLED must be true to backfill.", file=sys.stderr)
            return 1
        n = service.episodic.backfill(args.user)
        print(f"Backfilled {n} event(s) for user {args.user!r}.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
