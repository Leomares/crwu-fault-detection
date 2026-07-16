from src.scraper import DatasetScraper
from src.dataset import Dataloader, _Dataset 
import argparse


def main():
    parser = argparse.ArgumentParser(description="CLI tool to interact and test implemented methods.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    
    parser_scraper = subparsers.add_parser("scrape", help="Fetch and store the dataset from the web.")
    parser_scraper.add_argument("--fetch",action="store_true",help="Fetch the database from the web.")
    parser_scraper.add_argument("--resolve",action="store_true",help="Collect all time series data from the web.")

    parser_dataset = subparsers.add_parser("dataset", help="Build the dataset from the database.")
    parser_dataset.add_argument("--dataset",action="store_true",help="Generate the dataset with default config and features.")
    parser_dataset.add_argument("--loader",action="store_true",help="Test dataloader generator.")

    args = parser.parse_args()
    if args.command == "scrape":
        scraper = DatasetScraper()
        if args.fetch:
            scraper.fetch_database()
        if args.resolve:
            scraper.resolve_all_mat_data()
    elif args.command == "dataset":
        if args.dataset:
            dataset = _Dataset()
            data = dataset.get_dataset(time_per_datapoint_s=0.1, rebuild=True)
            print(f"Generated dataset with {data.shape} entries.")
            print(data.head())
        if args.loader:
            dataloader = Dataloader()
            data = dataloader.get_data()
            print(f"[1] Fetched all dataset from disk with {data.shape} entries")
            for batch in dataloader.get_batch(10):
                print(f"[2] Fetched batch with {len(batch)} entries")
                print(batch.head())
                break


if __name__ == "__main__":
    main()
