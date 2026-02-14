# BreastEdge AI - Guide Enregistrement Vidéo Démo

## ✅ COMPLÉTÉ
- **5 images de démo sélectionnées** dans `~/breast_edge_ai/demo_images/`:
  - `demo_benign_1.png` - BENIGN 97.5% confidence
  - `demo_benign_2.png` - BENIGN 97.1% confidence
  - `demo_malignant_1.png` - MALIGNANT 99.9% confidence
  - `demo_malignant_2.png` - MALIGNANT 99.9% confidence
  - `demo_borderline.png` - MALIGNANT 69.6% confidence (cas limite)

## 🎬 PROCHAINES ÉTAPES (Manuel - sur DGX Spark directement)

### Option A: Enregistrement avec OBS Studio (Recommandé)
1. **Installe OBS Studio** (si pas déjà installé):
   ```bash
   # Depuis le terminal DGX Spark avec sudo
   sudo apt install -y obs-studio
   ```

2. **Configure OBS**:
   - Source: Capture d'écran
   - Résolution: 1920x1080
   - FPS: 30
   - Format: MP4
   - Qualité: High

3. **Enregistre la démo** (~3 minutes):
   - Ouvre Firefox: `firefox http://localhost:7860`
   - Lance l'enregistrement OBS
   - Montre l'interface vide (5 sec)
   - Upload `demo_benign_1.png` → attends résultat complet → pause 3 sec
   - Upload `demo_malignant_1.png` → attends résultat → pause 3 sec
   - Upload `demo_malignant_2.png` → attends résultat → pause 3 sec
   - Upload `demo_borderline.png` → attends résultat → pause 3 sec
   - Upload `demo_benign_2.png` → attends résultat → pause 3 sec
   - Arrête l'enregistrement
   - Sauvegarde: `~/breast_edge_ai/demo_raw.mp4`

### Option B: Enregistrement avec SimpleScreenRecorder
```bash
sudo apt install -y simplescreenrecorder
simplescreenrecorder
```
- Suis les mêmes étapes qu'avec OBS

### Option C: Enregistrement avec ffmpeg (si déjà installé)
```bash
# Vérifie si ffmpeg est installé
which ffmpeg

# Si oui, lance l'enregistrement:
ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i :0.0 \
  -c:v libx264 -preset ultrafast -crf 18 \
  ~/breast_edge_ai/demo_raw.mp4 &

# Note le PID
FFMPEG_PID=$!

# Fais la démo (3 min max)

# Arrête l'enregistrement
kill $FFMPEG_PID
```

## 📸 SCREENSHOTS (Manuel)
1. **Installe scrot** (si nécessaire):
   ```bash
   sudo apt install -y scrot
   ```

2. **Capture l'interface vide**:
   ```bash
   firefox http://localhost:7860 &
   sleep 5
   scrot ~/breast_edge_ai/screenshot_interface.png
   ```

3. **Capture chaque résultat de prédiction** (après chaque upload dans la démo):
   ```bash
   scrot ~/breast_edge_ai/screenshot_benign_1.png
   scrot ~/breast_edge_ai/screenshot_malignant_1.png
   scrot ~/breast_edge_ai/screenshot_borderline.png
   # etc.
   ```

## 🎥 POST-PRODUCTION (Automatique - une fois demo_raw.mp4 créé)

Une fois que tu as `~/breast_edge_ai/demo_raw.mp4`, lance ce script:

```bash
cd ~/breast_edge_ai && python3 post_production.py
```

Le script `post_production.py` va automatiquement:
1. Créer un titre de 5 secondes
2. Créer un outro de 5 secondes avec les métriques
3. Concaténer: titre + demo + outro
4. Vérifier la durée (≤ 3 min)
5. Générer `demo_final.mp4`

## ✅ LIVRAISON
- Fichier final: `~/breast_edge_ai/demo_final.mp4`
- Durée: ≤ 3 minutes
- Résolution: 1920x1080
- Format: MP4

---

**Note**: Les outils de capture (scrot, ffmpeg, OBS) nécessitent `sudo` pour l'installation. Lance les commandes d'installation directement depuis le terminal DGX Spark avec ton mot de passe sudo.
