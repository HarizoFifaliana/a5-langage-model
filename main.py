from gpt2_train import GPT2Trainer
from google.colab import drive
drive.mount('/content/drive')

# Nom du fichier CSV contenant les données
file_name = "/content/drive/MyDrive/rakitra.csv"

# Initialiser l'entraîneur GPT-2
trainer = GPT2Trainer(project_name="gpt2_csv_training")

# Charger et préparer les données directement à partir du fichier CSV
trainer.prepare_data(file_name)

# Lancer l'entraînement du modèle
trainer.train()

print("Entraînement terminé avec succès.")