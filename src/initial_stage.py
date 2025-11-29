import os


def main():
    sample_dir = os.path.join(os.path.dirname(__file__), "../data/samples")
    print(f"Checking sample data directory: {sample_dir}")
    if os.path.isdir(sample_dir):
        print("Directory exists.")
    else:
        print("Directory does not exist. Please create it and place some sample files.")


if __name__ == "__main__":
    main()

