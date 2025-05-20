# 🍽️ FoodSnapChef

**FoodSnapChef** is an intelligent food recognition and recipe generation system using deep learning and hybrid AI. It identifies Indian and Western dishes from images and fetches accurate recipes from local datasets or external APIs. Perfect for foodies, chefs, and smart kitchen integrations!

---

## 🚀 Features

- 📸 Food recognition using two image classifiers:
  - Western food: `nateraw/food`
  - Indian food: `dima806/indian_food_image_detection`
- 🧠 Hybrid prediction pipeline (multi-model fallback with BEiT)
- 🔍 Automatic recipe generation:
  - **Spoonacular API** for Western dishes
  - **Local CSV** dataset for Indian dishes
- 📂 No manual inputs — completely image-based interaction
- 🧪 Optimized for performance and scalability

---

## 📷 Demo

> Upload a food image and get its name and full recipe automatically!

*(Add a demo gif or image if available)*

---

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/FoodSnapChef.git
cd FoodSnapChef
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

```bash
python main.py
```

This will:
- Classify the image (Indian/Western)
- Fetch recipe (from local CSV or Spoonacular)
- Output detailed ingredients and steps

---

## 🗂️ Dataset

- `indian_food_dataset.csv`: Custom curated dataset of 500+ Indian recipes
- Uses `labels.json` to interpret Indian food classifier results

---

## 🔑 API Key Setup

Add your **Spoonacular API key** in `utils.py`:
```python
SPOONACULAR_API_KEY = "your_api_key_here"
```

---

## 🧠 Tech Stack

- Python 3.8+
- Torch / Transformers
- Spoonacular API
- Pandas / OpenCV
- Pretrained models: `nateraw/food`, `dima806/indian_food_image_detection`, `beit-large-patch16-224`

---

## 🧹 To-Do (Enhancements)

- [ ] Web interface (Streamlit/Flask)
- [ ] Expand Indian dataset to 1000+ recipes
- [ ] Improve classifier performance with fine-tuning
- [ ] Dockerize for deployment
- [ ] Add nutritional analysis

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request for any improvements.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 📫 Contact

Created by:
- Shaik Mohammad
- Under the guidance of Ms.Mary Ch, SRM University – AP
