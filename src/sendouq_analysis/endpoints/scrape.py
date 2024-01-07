import argparse
from sendouq_analysis.ingest.scrape import scrape_matches_to_files
from sendouq_analysis.ingest.parse import parse

def main():
    parser = argparse.ArgumentParser(description='Scrape matches from sendou.ink and parse them.')
    parser.add_argument('--start-id', required=True, type=int, help='The id of the first match to scrape')
    parser.add_argument('--end-id', default=False, type=int, help='The id of the last match to scrape. If False, scrape until it errors out.')
    parser.add_argument('--save-path', default='/data', type=str, help='The path to save the matches to')
    args = parser.parse_args()

    scrape_matches_to_files(args.start_id, args.save_path, end_id=args.end_id)
    match_data = parse([args.save_path])
    print(match_data)

if __name__ == '__main__':
    main()


