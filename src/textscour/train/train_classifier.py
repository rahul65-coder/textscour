"""
Classifier Training Pipeline for TEXTSCOUR.
Trains multilingual sequence classification model (distilbert-base-multilingual-cased)
on clean negatives + injected promotional noise positives + hard negatives.
Saves model/, metrics.json, and triggers calibration.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from textscour.train.calibrate import calibrate_and_save
from textscour.train.synth_noise import BRANDS, fill_template, load_templates

logger = logging.getLogger(__name__)


# ============================================================================
# Built-in Authentic Clean Reference Texts (En, Hi, Hinglish) for Offline/Fallback
# ============================================================================

CLEAN_CORPUS_EN: list[str] = [
    "The solar system consists of the Sun and the objects that orbit it, whether they orbit it directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight planets.",
    "Photosynthesis is a biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy stored in glucose.",
    "The Industrial Revolution marked a period of rapid development and mechanization that transformed agrarian economies into industrial and urban ones in Britain and Europe.",
    "Albert Einstein developed the general theory of relativity, which remains one of the foundational pillars of modern physics alongside quantum mechanics.",
    "The Indian subcontinent has a long history of maritime trade, connecting ancient ports of Lothal and Muziris to Mesopotamia, Egypt, and Rome.",
    "Machine learning algorithms build mathematical models based on sample training data to make predictions or decisions without being explicitly programmed.",
    "Plate tectonics is the scientific theory that describes the large-scale motions of Earth's lithosphere divided into multiple continental and oceanic plates.",
    "DNA is composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions for the development and functioning of living organisms.",
    "The Reserve Bank of India regulates the monetary policy and currency circulation across the nation to maintain economic stability and financial growth.",
    "Software architecture refers to the fundamental structures of a software system and the discipline of creating such structures and systems.",
]

CLEAN_CORPUS_HI: list[str] = [
    "सौरमंडल सूर्य और उसकी परिक्रमा करने वाले खगोलीय पिंडों का समूह है। इसमें आठ प्रमुख ग्रह, उनके उपग्रह और क्षुद्रग्रह शामिल हैं।",
    "प्रकाश संश्लेषण वह जैव-रासायनिक प्रक्रिया है जिसमें हरे पौधे सूर्य के प्रकाश की उपस्थिति में कार्बन डाइऑक्साइड और जल से ग्लूकोज का निर्माण करते हैं।",
    "भारतीय संविधान विश्व का सबसे लंबा लिखित संविधान है जिसे संविधान सभा द्वारा छब्बीस नवंबर उन्नीस सौ उनचास को अंगीकृत किया गया था।",
    "हिमालय पर्वत श्रृंखला एशिया में स्थित है जो भारत, नेपाल, भूटान, चीन और पाकिस्तान की सीमाओं में फैली हुई एक नवीन वलित पर्वतमाला है।",
    "सिंधु घाटी सभ्यता कांस्य युगीन सभ्यता थी जो आधुनिक पाकिस्तान और पश्चिमोत्तर भारत के विस्तृत क्षेत्रों में विकसित हुई थी।",
    "कंप्यूटर विज्ञान में एल्गोरिदम किसी विशिष्ट समस्या के समाधान हेतु सुपरिभाषित चरणों की एक क्रमबद्ध श्रृंखला को कहा जाता है।",
    "महात्मा गांधी ने भारत के स्वतंत्रता संग्राम में सत्य और अहिंसा के सिद्धांतों पर आधारित सत्याग्रह आंदोलन का सफल नेतृत्व किया था।",
    "मौद्रिक नीति किसी देश के केंद्रीय बैंक द्वारा मुद्रा आपूर्ति और ब्याज दरों के नियंत्रण हेतु अपनाई जाने वाली आर्थिक नीति है।",
    "जल संरक्षण आज के युग में एक अत्यंत आवश्यक कदम है ताकि भावी पीढ़ियों के लिए शुद्ध पेयजल और कृषि जल की उपलब्धता बनी रहे।",
    "पर्यावरण प्रदूषण जैव विविधता और प्राकृतिक संतुलन के लिए एक गंभीर संकट बन चुका है जिसके समाधान के लिए वैश्विक प्रयास आवश्यक हैं।",
]

CLEAN_CORPUS_HINGLISH: list[str] = [
    "Indus Valley Civilisation me Harappa aur Mohenjo-daro jaise shaharon me behtareen urban planning thi.",
    "Harappa aur Mohenjo-daro jaise shaharon me behtareen urban planning, drainage system aur brick architecture paya gaya tha.",
    "Indus Valley Civilisation ek ancient bronze age civilisation thi jo modern Pakistan aur northwestern India ke regions me flourish hui thi.",
    "Is sabhyata ke log vyapar aur shilp kala me kafi nipun the aur unki lipi abhi tak padhi nahi ja saki hai.",
    "Photosynthesis ek aisi biological process hai jisme plants suraj ki roshni aur paani ka use karke glucose banate hain aur oxygen release karte hain.",
    "Is prakriya ke bina dharti par jeevan ka astitva sambhav nahi hai kyunki sabhi jeev oxygen aur bhojan ke liye ped paudhon par nirbhar karte hain.",
    "Indian Constitution duniya ka sabse lamba written constitution hai jo 26 January 1950 ko poore desh me lagu kiya gaya tha.",
    "Dr BR Ambedkar ki adhyakshata me bani drafting committee ne ise taiyar kiya tha jisme nagrikon ke maulik adhikar shamil hain.",
    "Solar system me total 8 planets hain jo sun ke around elliptical orbit me revolve karte hain, jisme Jupiter sabse bada planet hai.",
    "Surya ke guru-twaakarshan bal ke karan sabhi grah apni kasha me sthir rehte hain aur rituon ka chakra chalta hai.",
    "Computer architecture me CPU, memory unit, aur input output devices ka coordination system performance ko define karta hai.",
    "Modern processors me multiple cores aur cache memory se computation speed bohot tez ho gayi hai.",
    "Newton ke motion ke three laws classical mechanics ke fundamental principles hain jo objects par force aur motion ko explain karte hain.",
    "Machine learning models historical data par pattern recognition karke naye inputs par accurate predictions deliver karte hain.",
    "Reserve Bank of India desh ki monetary policy aur banking system ko regulate karta hai taaki mudrasfeeti niyantran me rahe.",
    "Climate change aur global warming ki wajah se arctic glaciers melt ho rahe hain aur paryavaran par bura asar pad raha hai.",
    "Renewable energy sources jaise solar aur wind power future energy requirements ko meet karne ke liye bohot zaroori hain.",
]

HARD_NEGATIVES: list[str] = [
    "According to research published in Nature (https://nature.com/articles/s41586-021-03819-2), global temperatures rose by 1.1 degrees Celsius over the pre-industrial average.",
    "For complete technical specifications and source code implementations, refer to the official repository at https://github.com/rust-lang/rust.",
    "The study methodology follows the guidelines established by the World Health Organization (https://who.int/publications).",
    "The camera produces exceptionally vibrant photographs in low-light environments, and the 5000 mAh battery provides over twenty-four hours of continuous usage. Build quality feels sturdy.",
    "This laptop features an aluminum unibody chassis, 32 gigabytes of high-speed memory, and a bright 4K display. Keyboard travel is comfortable for prolonged typing sessions.",
    "As observed by Smith et al. [14], linguistic representations capture structural hierarchies in multilingual models effectively.",
    "The complete dataset documentation and benchmark replication scripts are archived on arXiv (https://arxiv.org/abs/2103.00020).",
    "The soundstage is expansive with tight sub-bass response and balanced treble frequency curves. Acoustic isolation performs admirably during city commutes.",
]


class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = torch.tensor(label, dtype=torch.long)
        return item


def resolve_template_file(template_path: str) -> str | None:
    """Finds template file path across local dev and Kaggle input environments."""
    if os.path.exists(template_path):
        return template_path
    from pathlib import Path
    name = Path(template_path).name
    candidates = [
        Path(template_path),
        Path("configs/lexicons") / name,
        Path(__file__).resolve().parents[3] / "configs" / "lexicons" / name,
        Path("/kaggle/input/textscour-code/configs/lexicons") / name,
        Path("/tmp/textscour_build/configs/lexicons") / name,
    ]
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob(name):
            candidates.append(p)
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return None


def build_dataset_samples(
    template_path: str = "configs/lexicons/synth_templates_train.tsv",
    target_count: int = 2000,
) -> tuple[list[str], list[int]]:
    """
    Constructs balanced dataset:
    - Negatives (0): Clean corpus chunks + hard negatives (URLs/citations/product reviews without CTA)
    - Positives (1): Synthetic promotional and marketing noise injected via train templates
    """
    templates = []
    resolved_path = resolve_template_file(template_path)
    if resolved_path and os.path.exists(resolved_path):
        templates = load_templates(resolved_path)
        logger.info("Loaded %d training templates from %s", len(templates), resolved_path)
    else:
        logger.warning("No templates found at %s or candidate paths; falling back to basic synth", template_path)

    rng = random.Random(42)
    texts: list[str] = []
    labels: list[int] = []

    # 1. Clean Negatives (label 0)
    base_clean = CLEAN_CORPUS_EN + CLEAN_CORPUS_HI + CLEAN_CORPUS_HINGLISH
    n_neg = target_count // 2

    # Mix in clean text and hard negatives
    for i in range(n_neg):
        if i % 4 == 0:
            text = rng.choice(HARD_NEGATIVES)
        else:
            text = rng.choice(base_clean)
        texts.append(text)
        labels.append(0)

    # 2. Positives (label 1)
    n_pos = target_count - n_neg
    for i in range(n_pos):
        if templates:
            tmpl = rng.choice(templates)
            pos_text = fill_template(tmpl.template, rng)
        else:
            brand = rng.choice(BRANDS)
            pos_text = f"Special offer! Buy {brand} today and get 50% discount with promo code SAVE50."
        texts.append(pos_text)
        labels.append(1)

    # Shuffle dataset
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    shuffled_texts, shuffled_labels = zip(*combined)

    return list(shuffled_texts), list(shuffled_labels)


def train_classifier(
    output_dir: str = "model",
    model_name: str = "distilbert-base-multilingual-cased",
    batch_size: int = 32,
    lr: float = 3e-5,
    epochs: int = 2,
    max_train_seconds: int = 10800,
    sample_count: int = 1200,
    device_override: str | None = None,
) -> dict[str, Any]:
    """
    Trains sequence classification model with PyTorch, saves model/ and metrics.json,
    and runs calibration on validation logits.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Initializing classifier training on %s (model: %s)", device, model_name)

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Check multi-GPU DataParallel
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        logger.info("Enabling DataParallel across %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)

    # Build Samples
    texts, labels = build_dataset_samples(target_count=sample_count)
    n_total = len(texts)

    # 90 / 5 / 5 Split
    n_train = int(n_total * 0.90)
    n_val = int(n_total * 0.05)

    train_texts, train_labels = texts[:n_train], labels[:n_train]
    val_texts, val_labels = texts[n_train : n_train + n_val], labels[n_train : n_train + n_val]
    test_texts, test_labels = texts[n_train + n_val :], labels[n_train + n_val :]

    logger.info("Dataset splits: Train=%d, Val=%d, Test=%d", len(train_texts), len(val_texts), len(test_texts))

    train_ds = TextDataset(train_texts, train_labels, tokenizer, max_length=256)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, max_length=256)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    use_cuda_amp = (device.type == "cuda")

    best_val_f1 = 0.0
    val_logits_collector: list[float] = []
    val_labels_collector: list[int] = []

    for epoch in range(1, epochs + 1):
        if time.time() - start_time >= max_train_seconds:
            logger.warning("Reached max_train_seconds limit (%ds). Breaking training.", max_train_seconds)
            break

        model.train()
        train_loss_total = 0.0
        train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)

            if use_cuda_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, target)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, target)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_total / max(1, train_batches)

        # Validation evaluation
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        all_preds: list[int] = []
        all_targets: list[int] = []
        val_logits_collector.clear()
        val_labels_collector.clear()

        with torch.inference_mode():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, target)
                val_loss_total += loss.item()
                val_batches += 1

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                targets = target.cpu().numpy()

                all_preds.extend(preds.tolist())
                all_targets.extend(targets.tolist())

                # Collect logit difference / positive logit for calibration
                pos_logits = (logits[:, 1] - logits[:, 0]).cpu().numpy()
                val_logits_collector.extend(pos_logits.tolist())
                val_labels_collector.extend(targets.tolist())

        avg_val_loss = val_loss_total / max(1, val_batches)

        # Compute F1
        tp = sum(1 for p, t in zip(all_preds, all_targets) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(all_preds, all_targets) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(all_preds, all_targets) if p == 0 and t == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        val_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(
            f"[Epoch {epoch}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Val F1: {val_f1:.4f}",
            flush=True,
        )
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f, Precision: %.4f, Recall: %.4f, Val F1: %.4f",
            epoch, epochs, avg_train_loss, avg_val_loss, precision, recall, val_f1,
        )

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            # Save best checkpoint
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    total_duration = time.time() - start_time

    # Run temperature calibration
    calib_meta = calibrate_and_save(
        val_logits=val_logits_collector,
        val_labels=val_labels_collector,
        out_path=os.path.join(output_dir, "calibration.json"),
        model_name=model_name,
    )

    metrics = {
        "model_name": model_name,
        "epochs_completed": epoch,
        "best_val_f1": round(best_val_f1, 4),
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "test_samples": len(test_texts),
        "total_training_seconds": round(total_duration, 2),
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "calibration": calib_meta,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Classifier training finished in %.1fs. Model and metrics saved to %s", total_duration, output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TEXTSCOUR Ambiguous Chunk Classifier")
    parser.add_argument("--output-dir", type=str, default="model", help="Directory to save model artifacts")
    parser.add_argument("--model", type=str, default="distilbert-base-multilingual-cased", help="Base model identifier")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for AdamW")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--max-train-seconds", type=int, default=10800, help="Maximum allowed training wall clock time")
    parser.add_argument("--samples", type=int, default=600, help="Number of training samples to synthesize")
    args = parser.parse_args()

    train_classifier(
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        max_train_seconds=args.max_train_seconds,
        sample_count=args.samples,
    )


if __name__ == "__main__":
    main()
