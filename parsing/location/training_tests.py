"""Training data, target masking, fixed loss weights and dev-only calibration."""

import importlib.util
import math
import unittest
from types import SimpleNamespace

from parsing.location.filter.calibration import select_threshold
from parsing.location.train import Collator, encode_example, load_data, person, training_text, weight, weighted_loss

HAS_TORCH = importlib.util.find_spec("torch") is not None


class DataTests(unittest.TestCase):
    def test_selected_data_and_weights(self):
        expected = {"extractor": (535, 133, 235, 1.439252336448598),
                    "filter": (560, 140, 163, 1.582142857142857), "rewriter": (486, 156, 0, 1.0)}
        for component, (train_count, dev_count, weighted_count, mean) in expected.items():
            configuration, splits = load_data(component)
            self.assertEqual((len(splits["train"]), len(splits["dev"])), (train_count, dev_count))
            self.assertFalse({person(row) for row in splits["train"]} & {person(row) for row in splits["dev"]})
            values = [weight(row, configuration) for row in splits["train"]]
            self.assertEqual(sum(value != 1 for value in values), weighted_count)
            self.assertAlmostEqual(sum(values)/len(values), mean)

    def test_target_mask_and_explicit_eos(self):
        tokenizer = SimpleNamespace(eos_token_id=9, encode=lambda text, **kwargs: [3, 4],
                                    apply_chat_template=lambda messages, **kwargs: [1, 2])
        encoded = encode_example(tokenizer, "input", "target", 5)
        self.assertEqual(encoded["input_ids"], [1, 2, 3, 4, 9])
        self.assertEqual(encoded["labels"], [-100, -100, 3, 4, 9])
        with self.assertRaises(ValueError):
            encode_example(tokenizer, "input", "target", 4)

    def test_filter_target_not_in_prompt(self):
        configuration, splits = load_data("filter")
        row = {**splits["train"][0], "expected_output": "SECRET_REFERENCE_LABEL"}
        self.assertNotIn("SECRET_REFERENCE_LABEL", training_text("filter", row))

    def test_overlapping_features_do_not_multiply_weights(self):
        configuration, _ = load_data("extractor")
        self.assertEqual(weight({"difficulty_categories": configuration["focus_categories"]}, configuration), 2)
        self.assertEqual(weight({"difficulty_categories": ["multi_sentence"]}, configuration), 1)

    def test_dev_calibration_respects_safety_and_direction(self):
        rows = [{"expected_output": "Single", "multiple_log_odds": score} for score in (-5, -4)]
        rows += [{"expected_output": "Multiple", "multiple_log_odds": score} for score in (-3.375, 2)]
        threshold, metrics = select_threshold(rows)
        self.assertEqual(threshold, -3.375)
        self.assertEqual(metrics["single_recall"], 1)
        self.assertEqual(metrics["multiple_recall"], 1)
        with self.assertRaises(ValueError):
            select_threshold(rows[:2])
        with self.assertRaises(ValueError):
            select_threshold([{**row, "multiple_log_odds": math.nan} for row in rows])


@unittest.skipUnless(HAS_TORCH, "Install --extra generation for real tensor tests")
class TensorTests(unittest.TestCase):
    def test_weighting_survives_microbatch_one_and_backpropagates(self):
        import torch
        torch.manual_seed(42)
        logits = torch.randn(1, 4, 8, requires_grad=True)
        labels = torch.tensor([[-100, -100, 3, 4]])
        base = weighted_loss(logits, labels, torch.tensor([1.]), 1.5)
        focused = weighted_loss(logits, labels, torch.tensor([2.]), 1.5)
        self.assertTrue(torch.allclose(focused, 2*base))
        focused.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(logits.grad[:, 0], torch.zeros_like(logits.grad[:, 0])))
        self.assertGreater(logits.grad[:, 1:3].abs().sum().item(), 0)

    def test_collator_masks_padding_and_keeps_weights(self):
        collate = Collator(0)
        batch = collate([{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2], "loss_weight": 3.},
                         {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3], "loss_weight": 1.}])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, -100], [-100, 2, 3]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 0], [1, 1, 1]])
        self.assertEqual(batch["loss_weight"].tolist(), [3., 1.])


if __name__ == "__main__":
    unittest.main()
