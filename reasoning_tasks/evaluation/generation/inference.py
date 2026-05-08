"""
evaluation/generation/inference.py

Generation inference for Task 4 post-validation.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from common.dataset import ResponseGenerationDataset, serialize_response_target


def run_generation_inference(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: ResponseGenerationDataset,
    max_new_tokens: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Generate validation responses for Task 4.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer used by the model.
        dataset: Response generation validation dataset.
        max_new_tokens: Maximum number of new tokens to generate.
        device: Target torch device.

    Returns:
        Full generation records.
    """
    model.eval()
    records: List[Dict[str, Any]] = []

    for idx, sample in enumerate(dataset.samples):
        prompt_text = dataset.build_inference_text(sample)

        forced_prefix = ""
        if dataset.sys_reaction == "verbal":
            forced_prefix = "\nutterance_list:\n- "
            prompt_text = prompt_text + forced_prefix
            
        old_truncation_side = tokenizer.truncation_side
        tokenizer.truncation_side = "right"

        encoding = tokenizer(
            prompt_text,
            truncation=True,
            max_length=dataset.max_length,
            return_tensors="pt",
        )

        tokenizer.truncation_side = old_truncation_side
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

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

        output_ids = generated_ids[0]
        prompt_len = input_ids.shape[1]

        # Robust handling:
        # Case 1: generate() returns prompt + continuation
        # Case 2: generate() returns continuation only
        if output_ids.shape[0] >= prompt_len and torch.equal(output_ids[:prompt_len], input_ids[0]):
            new_tokens = output_ids[prompt_len:]
        else:
            new_tokens = output_ids

        generated_content = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        if dataset.sys_reaction == "verbal":
            prediction = "utterance_list:\n- " + generated_content
        else:
            prediction = generated_content

        reference = serialize_response_target(
            sample["target"],
            sys_reaction=dataset.sys_reaction,
        )

        ############### debug ################
        # if idx < 3:
        #     print("\n" + "=" * 60)
        #     print(f"[GEN DEBUG SAMPLE {idx}]")

        #     print("\n[PROMPT LEN]", prompt_len)
        #     print("[OUTPUT LEN]", output_ids.shape[0])
        #     print("[OUTPUT STARTS WITH PROMPT]",
        #         torch.equal(output_ids[:prompt_len], input_ids[0])
        #         if output_ids.shape[0] >= prompt_len else False)

        #     print("\n[PROMPT FILE MODE]", dataset.sys_reaction)

        #     print("\n[PROMPT TEXT HEAD]")
        #     print(dataset.prompt_text[:500])

        #     print("\n[REFERENCE TARGET FORMAT]")
        #     print(repr(reference[:300]))

        #     print("\n[PREDICTION HEAD]")
        #     print(repr(prediction[:300]))


        #     print("\n[GENERATED CONTENT HEAD]")
        #     print(repr(generated_content[:300]))

        #     print("\n[FORMAT CHECK]")
        #     print("Starts with 'utterance_list:' ->", prediction.strip().startswith("utterance_list:"))

        #     print("=" * 60)
        ############### debug ################

        # Teacher-forcing NLL/PPL on the gold reference target.
        # This uses dataset.__getitem__(), where labels mask the prompt with -100.
        tf_batch = dataset[idx]

        tf_input_ids = tf_batch["input_ids"].unsqueeze(0).to(device)
        tf_attention_mask = tf_batch["attention_mask"].unsqueeze(0).to(device)
        tf_labels = tf_batch["labels"].unsqueeze(0).to(device)

        with torch.no_grad():
            tf_outputs = model(
                input_ids=tf_input_ids,
                attention_mask=tf_attention_mask,
                labels=tf_labels,
            )

        nll = float(tf_outputs.loss.item())
        ppl = float(torch.exp(tf_outputs.loss).sum().item())

        info = dataset.additional_info(idx)

        records.append(
            {
                "record_index": idx,
                "sample_id": info["sample_id"],
                "prediction": prediction,
                "reference": reference,
                "input": info["raw_input"],
                "task_fields": sample.get("fields", {}),
                "nll": nll,
                "ppl": ppl,
            }
        )
    return records
