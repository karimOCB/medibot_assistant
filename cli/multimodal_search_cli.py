import argparse
from lib.multimodal_search import verify_image_embedding, image_search_commad

def main():
    parser = argparse.ArgumentParser(description="describe image cli")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_image_embedding_parser.add_argument("image_path", type=str, help="Required image path")

    image_search_parser = subparsers.add_parser("image_search", help="Multimodal search")
    image_search_parser.add_argument("image_path", type=str, help="Required image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            embedding = verify_image_embedding(args.image_path)
            print(f"Embedding sh ape: {embedding.shape[0]} dimensions")
        case "image_search":
            results = image_search_commad(args.image_path)
            for i, dr in enumerate(results, 1):
                print(f"{i}. Name: {dr["name"]}.(similarity: {dr["similarity_score"]:.4f})")
                print(f"Age: {dr["age"]}. Specialty: {dr["specialty"]}. Bio: {dr["bio"]}. Availability: {dr["availability"]}")
        case _: 
            parser.print_help()

if __name__ == "__main__":
    main()