import torch
from sklearn.metrics import accuracy_score, f1_score


def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    valid_steps = 0

    model_device = next(model.parameters()).device

    for batch in dataloader:
        batch = {k: v.to(model_device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        if loss is None or not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss detected: {loss}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        valid_steps += 1

    if valid_steps == 0:
        raise ValueError("No valid training steps were executed.")

    return total_loss / valid_steps


def validate_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    valid_steps = 0

    model_device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model_device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            if loss is None or not torch.isfinite(loss):
                raise ValueError(f"Non-finite validation loss detected: {loss}")

            total_loss += loss.item()
            valid_steps += 1

    if valid_steps == 0:
        raise ValueError("No valid validation steps were executed.")

    return total_loss / valid_steps


def parse_label(decoded_text: str, valid_labels: list[str], fallback_label: str | None = None) -> str:
    normalized = decoded_text.strip().lower()

    for label in valid_labels:
        if normalized == label.lower():
            return label

    first_line = normalized.splitlines()[0].strip() if normalized else ""
    for label in valid_labels:
        if first_line == label.lower():
            return label

    for label in valid_labels:
        if label.lower() in normalized:
            return label

    if fallback_label is not None:
        return fallback_label

    return valid_labels[-1] # defalut label returns the last one

def validate_generation(model, dataset, tokenizer, valid_labels, max_new_tokens=8):
    model.eval()

    preds = []
    labels = []

    model_device = next(model.parameters()).device

    for sample in dataset.samples:
        prompt_text = dataset.build_inference_text(sample)

        encoding = tokenizer(
            prompt_text,
            truncation=True,
            max_length=dataset.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(model_device)
        attention_mask = encoding["attention_mask"].to(model_device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[0][input_ids.shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

        pred_label = parse_label(decoded, valid_labels)
        true_label = sample["target"]

        preds.append(pred_label)
        labels.append(true_label)

    val_acc = accuracy_score(labels, preds)
    val_f1 = f1_score(labels, preds, average="weighted")

    return val_acc, val_f1