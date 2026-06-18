from src.scraper import DatasetScraper
import argparse


def main():
    parser = argparse.ArgumentParser(description="CLI tool to interact and test implemented methods.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    
    parser_scraper = subparsers.add_parser("scrape", help="Fetch and store the dataset from the web.")
    parser_scraper.add_argument("--fetch",action="store_true",help="Fetch the database from the web.")
    parser_scraper.add_argument("--resolve",action="store_true",help="Collect all time series data from the web.")

    args = parser.parse_args()
    if args.command == "scrape":
        scraper = DatasetScraper()
        if args.fetch:
            scraper.fetch_database()
        if args.resolve:
            scraper.resolve_all_mat_data()

if __name__ == "__main__":
    main()
