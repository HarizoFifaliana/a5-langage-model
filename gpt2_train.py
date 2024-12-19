pip install datasets

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import wandb

class GPT2Trainer:
    def __init__(self, model_name="gpt2", output_dir="./results", logging_dir="./logs", project_name="gpt2_training"):
        """Initialise le modèle, le tokenizer et Weights & Biases."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.project_name = project_name

        # Définir le token de padding
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialiser W&B
        wandb.init(project=self.project_name)

    def prepare_data(self, file_path):
        """Charge et prépare les données."""
        dataset = load_dataset("csv", data_files=file_path)
        train_test_split = dataset["train"].train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]

        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def train(self, num_train_epochs=3, per_device_train_batch_size=8, eval_steps=500, save_steps=500):
        """Configure et lance l'entraînement."""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            save_total_limit=2,
            logging_dir=self.logging_dir,
            logging_steps=100,
            report_to=["wandb"],  # Activer le suivi W&B
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()

        # Sauvegarder le modèle et le tokenizer
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Modèle et tokenizer sauvegardés dans : {self.output_dir}")

        # Terminer W&B
        wandb.finish()
