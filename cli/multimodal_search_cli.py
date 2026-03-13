import argparse
from lib.multimodal_search import verify_image_embedding

def main():
    parser = argparse.ArgumentParser(description="describe image cli")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_image_embedding_parser.add_argument("image_path", type=str, help="Required image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            embedding = verify_image_embedding(args.image_path)
            print(f"Embedding shape: {embedding.shape[0]} dimensions")
        case _: 
            parser.print_help()

if __name__ == "__main__":
    main()