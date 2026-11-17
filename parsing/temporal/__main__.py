"""Inspect temporal parsing for one answer."""
import argparse
from dataclasses import asdict
import json

from . import parse_temporal_text, parse_temporal_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', help='Generated answer to inspect')
    parser.add_argument('--relation', choices=('birth', 'death'), help='Select the requested event rather than all dates')
    parser.add_argument('--include-brackets', action='store_true', help='Include bracketed content when extracting all dates; event selection always includes it')
    args = parser.parse_args()
    result = (parse_temporal_response(args.text, args.relation) if args.relation
              else parse_temporal_text(args.text, include_brackets=args.include_brackets))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
