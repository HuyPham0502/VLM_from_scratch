import math
from collections import Counter
from tqdm import tqdm
import torch
import nltk
from rouge_score import rouge_scorer


def compute_cider(predictions: list, references: list) -> float:
    """Compute CIDEr (Consensus-based Image Description Evaluation) score.

    Uses TF-IDF weighted n-gram matching across 1-gram to 4-gram.

    Args:
        predictions: List of generated answer strings.
        references: List of ground truth answer strings (or list of lists).

    Returns:
        float: CIDEr score (higher is better).
    """
    if not predictions or not references:
        return 0.0

    # Ensure reference format is list of lists
    refs = [[r] if isinstance(r, str) else r for r in references]
    n_samples = len(predictions)

    # Compute document frequency for each n-gram across references
    doc_freq = [Counter() for _ in range(4)]
    for ref_list in refs:
        seen = [set() for _ in range(4)]
        for ref in ref_list:
            words = ref.lower().split()
            for n in range(1, 5):
                for i in range(len(words) - n + 1):
                    ngram = tuple(words[i:i + n])
                    seen[n - 1].add(ngram)
        for n in range(4):
            for ngram in seen[n]:
                doc_freq[n][ngram] += 1

    # Compute TF-IDF cosine similarity for each prediction vs reference
    scores = []
    for pred, ref_list in zip(predictions, refs):
        pred_words = pred.lower().split()
        sample_scores = []
        for n in range(1, 5):
            pred_ngrams = Counter([tuple(pred_words[i:i + n]) for i in range(len(pred_words) - n + 1)])
            ref_ngrams = Counter()
            for r in ref_list:
                rw = r.lower().split()
                ref_ngrams.update([tuple(rw[i:i + n]) for i in range(len(rw) - n + 1)])

            # Compute TF-IDF vectors
            all_ngrams = set(pred_ngrams.keys()) | set(ref_ngrams.keys())
            if not all_ngrams:
                continue

            dot_product = 0.0
            pred_norm = 0.0
            ref_norm = 0.0
            for ng in all_ngrams:
                df = doc_freq[n - 1].get(ng, 0)
                idf = math.log(max(1.0, float(n_samples) / (df + 1e-6)))
                p_val = pred_ngrams.get(ng, 0) * idf
                r_val = (ref_ngrams.get(ng, 0) / max(1, len(ref_list))) * idf
                dot_product += p_val * r_val
                pred_norm += p_val * p_val
                ref_norm += r_val * r_val

            norm = (math.sqrt(pred_norm) * math.sqrt(ref_norm))
            if norm > 0:
                sample_scores.append(dot_product / norm)

        if sample_scores:
            scores.append(sum(sample_scores) / len(sample_scores) * 10.0)

    return sum(scores) / max(1, len(scores)) if scores else 0.0


class VLMEvaluator:
    """Comprehensive evaluation suite for Vietnamese Vision-Language Models.

    Calculates:
        1. BLEU-4: N-gram precision with brevity penalty.
        2. ROUGE-L: Longest Common Subsequence F1-score.
        3. METEOR: Word alignment considering stems and synonyms.
        4. BERTScore: Semantic contextual similarity via multilingual BERT.
        5. CIDEr: Consensus TF-IDF weighted n-gram similarity.
    """

    def __init__(self, model, tokenizer, device=None):
        """Initialize VLMEvaluator.

        Args:
            model: VisionLanguageModel instance.
            tokenizer: Hugging Face tokenizer instance.
            device: Compute device.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        # Download nltk resources if missing
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            pass

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        max_samples: int = None,
        use_bert_score: bool = True,
    ) -> dict:
        """Run inference across the evaluation dataset and compute all 5 metrics.

        Args:
            dataloader: PyTorch DataLoader yielding batched samples.
            max_samples: Optional limit on the number of evaluation samples.
            use_bert_score: Whether to compute BERTScore (requires model download).

        Returns:
            dict: Dictionary containing calculated metric scores.
        """
        self.model.eval()
        predictions = []
        references = []

        total_batches = len(dataloader)
        sample_count = 0

        print("\n[Evaluator] Generating predictions on evaluation set...")
        for batch in tqdm(dataloader, total=total_batches, desc="Evaluating"):
            images = batch["images"].to(self.device)
            questions = batch["questions"]
            answers = batch["answers"]

            B = images.shape[0]
            for i in range(B):
                pred_text = self.model.generate(
                    image=images[i],
                    question=questions[i],
                    max_new_tokens=48,
                    temperature=0.0,  # Greedy decoding for consistent evaluation
                )
                predictions.append(pred_text.strip())
                references.append(answers[i].strip())

                sample_count += 1
                if max_samples and sample_count >= max_samples:
                    break
            if max_samples and sample_count >= max_samples:
                break

        print(f"[Evaluator] Collected {len(predictions)} predictions. Computing metrics...")

        # Metric 1: BLEU-4
        bleu_scores = []
        smooth = nltk.translate.bleu_score.SmoothingFunction().method1
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = [ref.lower().split()]
            score = nltk.translate.bleu_score.sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth,
            )
            bleu_scores.append(score)
        bleu4 = (sum(bleu_scores) / max(1, len(bleu_scores))) * 100.0

        # Metric 2: ROUGE-L
        rouge_scores = []
        for pred, ref in zip(predictions, references):
            res = self.rouge.score(ref, pred)
            rouge_scores.append(res["rougeL"].fmeasure)
        rouge_l = (sum(rouge_scores) / max(1, len(rouge_scores))) * 100.0

        # Metric 3: METEOR
        meteor_scores = []
        for pred, ref in zip(predictions, references):
            try:
                m_score = nltk.translate.meteor_score.meteor_score([ref.split()], pred.split())
                meteor_scores.append(m_score)
            except Exception:
                meteor_scores.append(0.0)
        meteor = (sum(meteor_scores) / max(1, len(meteor_scores))) * 100.0

        # Metric 4: CIDEr
        cider = compute_cider(predictions, references)

        # Metric 5: BERTScore
        bert_f1 = 0.0
        if use_bert_score:
            try:
                import bert_score
                P, R, F1 = bert_score.score(
                    predictions,
                    references,
                    lang="vi",
                    verbose=False,
                    rescale_with_baseline=False,
                )
                bert_f1 = F1.mean().item() * 100.0
            except Exception as e:
                print(f"[Evaluator] BERTScore computation skipped: {e}")

        results = {
            "BLEU-4": round(bleu4, 2),
            "ROUGE-L": round(rouge_l, 2),
            "METEOR": round(meteor, 2),
            "CIDEr": round(cider, 2),
            "BERTScore-F1": round(bert_f1, 2),
            "samples_evaluated": len(predictions),
        }

        # Print clean results table
        print("\n" + "=" * 50)
        print("           EVALUATION RESULTS SUMMARY")
        print("=" * 50)
        for metric, val in results.items():
            print(f"  {metric:<20}: {val}")
        print("=" * 50 + "\n")

        # Show first 3 sample predictions
        print("[Sample Qualitative Predictions]:")
        for i in range(min(3, len(predictions))):
            print(f"  [{i+1}] Ref : {references[i]}")
            print(f"      Pred: {predictions[i]}")

        return results
