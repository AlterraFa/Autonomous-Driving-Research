import carla
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Name of the Town to switch", type=str)
parser.add_argument("--list", help="List out all available town name", action="store_true")
args = parser.parse_args()

if args.name or args.list:
    try:
        print("Connecting to CARLA server...")
        client = carla.Client("localhost", 2000)
        # Low timeout for the initial connection check
        client.set_timeout(5.0) 
        
        # Check if server is actually alive
        client.get_server_version()
        print("Connected!")

        # 1. Handle Listing first
        if args.list:
            print("Fetching map list...")
            maps = client.get_available_maps()
            print(f"There are {len(maps)} maps available:")
            for map_path in maps:
                # Clean up the path to show just the Town Name
                print(f" - {map_path.split('/')[-1]}")

        # 2. Handle Loading second (only if --name was provided)
        if args.name:
            print(f"Loading map: {args.name}...")
            # Use a longer timeout for loading maps
            client.set_timeout(60.0) 
            client.load_world(args.name)
            print("Map loaded successfully!")

    except RuntimeError as e:
        print(f"Error: {e}")
        print("Hint: Check if the CARLA server is running or if it's stuck in synchronous mode.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
else:
    parser.print_help()