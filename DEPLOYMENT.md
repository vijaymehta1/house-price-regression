# Deployment — Streamlit Community Cloud (free)

Ye project Streamlit Community Cloud ke free plan pe deploy hota hai.
Deploy se pehle saara code GitHub repo me hona chahiye.

## Kya-kya chahiye

- GitHub account: **vijaymehta1**
- Repo ka naam: `house-price-regression` (public)
- Repo me ye files honi chahiye (sab pehle se tracked hain):
  - `app/app.py` — main app file
  - `requirements.txt` — pinned versions
  - `models/house_price_model.pkl` — saved model (178 KB)
  - `data/train.csv` — defaults (median/mode) ke liye app ise padhta hai
  - `reports/model_comparison.json` — model info ke liye

## Steps

### 1. GitHub pe repo banao

github.com par jao → New repository → naam `house-price-regression`
→ Public → Create. Koi README/license wahan se add mat karo
(khaali repo chahiye, warna push me conflict hoga).

### 2. Code push karo (project folder se)

```bash
cd ~/workspace/house-price-regression
git remote add origin https://github.com/vijaymehta1/house-price-regression.git
git branch -M main
git push -u origin main
```

GitHub username/password poochhega — apne account se login karo.
(Pehli baar Personal Access Token maang sakta hai; password ki jagah
token daalna hota hai. Token: GitHub → Settings → Developer settings
→ Personal access tokens → Generate.)

### 3. Streamlit Cloud pe deploy karo

1. https://share.streamlit.io kholo → GitHub se sign in karo.
2. **New app** → **Use existing repo** chuno.
3. Repository: `vijaymehta1/house-price-regression`, Branch: `main`,
   Main file path: `app/app.py`.
4. **Deploy** dabao. Pehli baar 2-3 minute lagte hain (libraries
   install hoti hain).
5. Deploy hone ke baad ek public URL milega, jaise
   `https://house-price-regression.streamlit.app`.

### 4. Deploy ke baad test karo

- App kholo, ek typical ghar bharo (OverallQual 7, GrLivArea 2000,
  Neighborhood NridgHt) → price aani chahiye, lagbhag **$203,548**.
- Model Information section me R² 0.9127 aur RMSE $19,708 dikhna chahiye.
- Agar koi error aaye to Streamlit Cloud ke **Manage app → Logs**
  me dekho.

## Notes

- App ko chalane ke liye `data/train.csv` zaroori hai (defaults ke
  liye) — isliye ye file repo me tracked hai, `.gitignore` me nahi.
- Model file bhi repo me hai; dobara training ki zaroorat nahi.
- Free plan pe app sone (sleep) ho sakta hai jab koi use na kare —
  dobara kholne pe 30-60 second me jaag jata hai.
