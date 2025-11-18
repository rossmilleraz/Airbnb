# Airbnb Review Quality Prediction  
This is a project I built to practice doing data science the “real” way — dealing with messy data, cleaning it up, engineering features, and training a model that actually gives you something useful.

The main idea behind the project is simple:

**Can we predict a listing’s overall review quality *before* guests leave reviews?**

InsideAirbnb has a ton of raw data about listings, hosts, neighborhoods, availability, etc.  
The reviews themselves come later. So this project focuses on using only "pre-review" information to estimate the final rating.

It’s basically a small, end-to-end ML pipeline that mirrors the stuff you’d expect to do in a junior data science job.

---

## ⭐ What the project actually does

### **1. extract.py**
A small helper that loads CSVs using `pathlib`, so the project works on anyone’s computer without hardcoding file paths.

### **2. clean.py**
Cleans the raw dataset and fixes everything that’s annoying:
- drops high-missing columns  
- lowercases text  
- converts things like `"85%"` → `0.85`  
- parses bathroom_text into clean numeric values  
- converts types (dates, booleans, floats, etc.)  
- saves `listings_cleaned.csv`

### **3. transform.py**
Does all the feature engineering that actually matters:
- host tenure (days/years)
- days since last review
- occupancy ratio
- review score mean / std
- beds per guest
- one-hot encodes room/property type
- frequency-encodes neighborhoods
- clips outliers
- saves `listings_transformed.csv`

### **4. model.py**
Trains a Random Forest on the engineered dataset and reports:
- MAE  
- RMSE  
- R²  

It also saves:
- the model (`pkl`)  
- the feature list used during training  


### **5. run_pipeline.py**
This is the "one-click" version of everything.

Run it once and it will:
- clean  
- transform  
- model  
- save results  
- print performance metrics  

Perfect for anyone who wants to replicate the results without touching the code.

---

## 🧠 Why I built it this way

I wanted to get comfortable with:
- writing modular code instead of giant notebooks
- structuring a project like something you’d actually see at work
- cleaning messy data where nothing is labeled nicely
- understanding how different features affect a model
- using Git/GitHub properly
- separating EDA, cleaning, transforming, and modeling
- working with Random Forests for tabular regression

This project forced me to make a lot of actual decisions (what to drop, what to keep, how to engineer certain features), which is the kind of experience I wanted.

---

## 📁 Project structure
<img width="277" height="540" alt="Screenshot 2025-11-17 170159" src="https://github.com/user-attachments/assets/355a1e4e-e0b1-492e-a95b-63d209d49d11" />
