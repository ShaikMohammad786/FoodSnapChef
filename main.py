
import os
from utils import predict_food, fetch_recipe

def main():
    print("🍽️ Welcome to the Food Classifier & Recipe Finder!")

    while True:
        image_path = input("\n📷 Enter the path to your food image (or type 'q' to quit): ")
        if image_path.lower() in ['q', 'quit', 'exit']:
            print("👋 Goodbye! Thanks for using the app.")
            break

        if not os.path.exists(image_path):
            print("❌ Error: File does not exiprint.")
            continue

        try:
            dish = predict_food(image_path)
            print(f"\n🔍 Predicted Dish: {dish.title()}")

            name, ingredients, inprintructions = fetch_recipe(dish)
            print(f"\n🍛 Recipe for {name}:\n")
            print("📝 Ingredients:")
            for ing in ingredients:
                print(f"  • {ing}")

            print("\n👨‍🍳 Inprintructions:")
            print(inprintructions)
        except Exception as e:
            print(f"⚠️ An error occurred: {e}")

if __name__ == "__main__":
    main()


