#!/usr/bin/env python3
"""
다양성 유지 공진화 (Diversity-Preserving Coevolution)

목표: Phase2와 상호보완적인 모델 유지
- FN 최소화 특화 (독성 놓치지 않기)
- FP는 Phase2가 보완하므로 허용
- Phase2와 예측이 다른 샘플에 집중

핵심 전략:
1. FN 가중치 3.0 (FP 가중치 1.0)
2. Phase2가 놓친 샘플 집중 학습
3. 공격 성공률이 높아야 학습 (다양성 유지)
4. 긴 학습 시간 (에폭 증가, 샘플 증가)
"""

import argparse
import logging
import os
import sys
import time
import json
import random
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, confusion_matrix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class WeightedToxicDataset(Dataset):
    """가중치가 적용된 독성 데이터셋"""

    def __init__(self, texts, labels, weights, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "weight": torch.tensor(self.weights[idx], dtype=torch.float),
        }


class WeightedTrainer(Trainer):
    """FN 가중치가 높은 Trainer"""

    def __init__(self, fn_weight=3.0, fp_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        weights = inputs.pop("weight", None)
        outputs = model(**inputs)
        logits = outputs.logits

        # 기본 cross entropy
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, labels)

        # 가중치 적용
        if weights is not None:
            loss = loss * weights

        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss


class DiversityCoevolution:
    """다양성 유지 공진화 시스템"""

    def __init__(
        self,
        coevo_model_path: str = "models/coevolution-latest",
        phase2_model_path: str = "models/phase2-combined/best_model",
        data_dir: str = "data/korean",
        fn_weight: float = 3.0,
        fp_weight: float = 1.0,
        min_evasion_for_train: float = 0.05,  # 5% 이상이어야 학습
        samples_per_cycle: int = 500,  # 사이클당 샘플 수 증가
        epochs_per_retrain: int = 3,  # 재학습 에폭 증가
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.coevo_path = Path(coevo_model_path)
        self.phase2_path = Path(phase2_model_path)
        self.data_dir = Path(data_dir)
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
        self.min_evasion = min_evasion_for_train
        self.samples_per_cycle = samples_per_cycle
        self.epochs_per_retrain = epochs_per_retrain

        # 통계
        self.stats = {
            "cycles": 0,
            "retrains": 0,
            "total_samples": 0,
            "evasion_history": [],
            "fn_count_history": [],
            "diversity_score_history": [],
        }

        # 공격 전략들 (더 다양하게)
        self.attack_strategies = [
            "typo",
            "jamo",
            "chosung",
            "leet",
            "unicode",
            "spacing",
            "phonetic",
            "mixed_light",
            "mixed_heavy",
            "slang",
            "context_inject",
            "char_repeat",
            "vowel_change",
            "consonant_sub",
            "reverse_partial",
        ]

        self._load_models()
        self._load_data()
        self._load_slang_dict()

    def _load_models(self):
        """모델 로드"""
        logger.info("Loading models...")

        # Coevolution 모델
        self.coevo_tokenizer = AutoTokenizer.from_pretrained(self.coevo_path)
        self.coevo_model = AutoModelForSequenceClassification.from_pretrained(
            self.coevo_path
        ).to(self.device)

        # Phase2 모델 (비교용)
        self.phase2_tokenizer = AutoTokenizer.from_pretrained(self.phase2_path)
        self.phase2_model = AutoModelForSequenceClassification.from_pretrained(
            self.phase2_path
        ).to(self.device)
        self.phase2_model.eval()

        logger.info(f"Device: {self.device}")

    def _load_data(self):
        """학습/검증 데이터 로드"""
        logger.info("Loading data...")

        # 학습 데이터
        train_texts, train_labels = [], []

        # KOTOX train
        kotox_train = self.data_dir / "KOTOX/data/KOTOX_classification/total/train.csv"
        if kotox_train.exists():
            df = pd.read_csv(kotox_train)
            toxic_df = df[df["label"] == 1]
            train_texts.extend(toxic_df["text"].tolist())
            train_labels.extend(toxic_df["label"].tolist())

        # 슬랭 데이터
        slang_file = self.data_dir / "slang_toxic.csv"
        if slang_file.exists():
            df = pd.read_csv(slang_file)
            if "text" in df.columns and "label" in df.columns:
                toxic_df = df[df["label"] == 1]
                train_texts.extend(toxic_df["text"].tolist())
                train_labels.extend(toxic_df["label"].tolist())

        self.train_texts = train_texts
        self.train_labels = train_labels
        logger.info(f"Loaded {len(train_texts)} toxic training samples")

        # 검증 데이터
        val_texts, val_labels = [], []

        # KOTOX valid
        kotox_valid = self.data_dir / "KOTOX/data/KOTOX_classification/total/valid.csv"
        if kotox_valid.exists():
            df = pd.read_csv(kotox_valid)
            val_texts.extend(df["text"].tolist())
            val_labels.extend(df["label"].tolist())

        # BEEP
        beep_dev = self.data_dir / "beep_dev.tsv"
        if beep_dev.exists():
            df = pd.read_csv(beep_dev, sep="\t")
            df["label"] = df["hate"].apply(lambda x: 0 if x == "none" else 1)
            val_texts.extend(df["comments"].tolist())
            val_labels.extend(df["label"].tolist())

        # UnSmile
        unsmile_valid = self.data_dir / "unsmile_valid.tsv"
        if unsmile_valid.exists():
            df = pd.read_csv(unsmile_valid, sep="\t")
            df["label"] = 1 - df["clean"]
            val_texts.extend(df["문장"].tolist())
            val_labels.extend(df["label"].tolist())

        self.val_texts = val_texts
        self.val_labels = val_labels
        logger.info(f"Loaded {len(val_texts)} validation samples")

    def _load_slang_dict(self):
        """슬랭 사전 로드"""
        self.slang_dict = {}
        slang_file = self.data_dir / "slang_dict.json"
        if slang_file.exists():
            with open(slang_file) as f:
                self.slang_dict = json.load(f)
        logger.info(f"Loaded {len(self.slang_dict)} slang mappings")

    def _apply_attack(self, text: str, strategy: str) -> str:
        """공격 전략 적용"""
        if strategy == "typo":
            return self._typo_attack(text)
        elif strategy == "jamo":
            return self._jamo_attack(text)
        elif strategy == "chosung":
            return self._chosung_attack(text)
        elif strategy == "leet":
            return self._leet_attack(text)
        elif strategy == "unicode":
            return self._unicode_attack(text)
        elif strategy == "spacing":
            return self._spacing_attack(text)
        elif strategy == "phonetic":
            return self._phonetic_attack(text)
        elif strategy == "mixed_light":
            return self._mixed_attack(text, intensity=0.3)
        elif strategy == "mixed_heavy":
            return self._mixed_attack(text, intensity=0.7)
        elif strategy == "slang":
            return self._slang_attack(text)
        elif strategy == "context_inject":
            return self._context_inject(text)
        elif strategy == "char_repeat":
            return self._char_repeat(text)
        elif strategy == "vowel_change":
            return self._vowel_change(text)
        elif strategy == "consonant_sub":
            return self._consonant_sub(text)
        elif strategy == "reverse_partial":
            return self._reverse_partial(text)
        return text

    def _typo_attack(self, text: str) -> str:
        """오타 삽입"""
        chars = list(text)
        if len(chars) < 2:
            return text
        idx = random.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    def _jamo_attack(self, text: str) -> str:
        """자모 분리"""
        result = []
        for char in text:
            if "가" <= char <= "힣":
                code = ord(char) - 0xAC00
                cho = code // 588
                jung = (code % 588) // 28
                jong = code % 28
                cho_chars = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
                jung_chars = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
                jong_chars = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
                result.append(cho_chars[cho])
                result.append(jung_chars[jung])
                if jong > 0:
                    result.append(jong_chars[jong])
            else:
                result.append(char)
        return "".join(result)

    def _chosung_attack(self, text: str) -> str:
        """초성만 추출"""
        result = []
        cho_chars = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
        for char in text:
            if "가" <= char <= "힣":
                code = ord(char) - 0xAC00
                cho = code // 588
                result.append(cho_chars[cho])
            else:
                result.append(char)
        return "".join(result)

    def _leet_attack(self, text: str) -> str:
        """리트스피크"""
        leet_map = {
            "ㅏ": "ㅑ", "ㅓ": "ㅕ", "ㅗ": "ㅛ", "ㅜ": "ㅠ",
            "a": "4", "e": "3", "i": "1", "o": "0", "s": "5",
        }
        return "".join(leet_map.get(c, c) for c in text)

    def _unicode_attack(self, text: str) -> str:
        """유니코드 변형"""
        variants = {"ㅅ": "ㅆ", "ㅂ": "ㅃ", "ㄱ": "ㄲ", "ㅈ": "ㅉ", "ㄷ": "ㄸ"}
        return "".join(variants.get(c, c) for c in text)

    def _spacing_attack(self, text: str) -> str:
        """띄어쓰기 조작"""
        # 랜덤하게 띄어쓰기 제거 또는 추가
        if random.random() > 0.5:
            return text.replace(" ", "")
        else:
            chars = list(text)
            for i in range(len(chars) - 1, 0, -1):
                if random.random() < 0.2:
                    chars.insert(i, " ")
            return "".join(chars)

    def _phonetic_attack(self, text: str) -> str:
        """발음 기반 변형"""
        phonetic_map = {"시": "씨", "바": "빠", "새": "쌔", "개": "걔"}
        result = text
        for k, v in phonetic_map.items():
            result = result.replace(k, v)
        return result

    def _mixed_attack(self, text: str, intensity: float = 0.5) -> str:
        """복합 공격"""
        strategies = random.sample(
            ["typo", "spacing", "leet", "phonetic"],
            k=int(len(["typo", "spacing", "leet", "phonetic"]) * intensity) + 1,
        )
        result = text
        for strategy in strategies:
            result = self._apply_attack(result, strategy)
        return result

    def _slang_attack(self, text: str) -> str:
        """슬랭 치환"""
        result = text
        for standard, slang_list in self.slang_dict.items():
            if standard in result and slang_list:
                result = result.replace(standard, random.choice(slang_list))
        return result

    def _context_inject(self, text: str) -> str:
        """맥락 주입"""
        prefixes = ["ㅋㅋ ", "ㅎㅎ ", "아 ", "그냥 ", "진짜 "]
        suffixes = [" ㅋㅋ", " ㅎㅎ", " ㄹㅇ", " ㅇㅇ", ""]
        return random.choice(prefixes) + text + random.choice(suffixes)

    def _char_repeat(self, text: str) -> str:
        """글자 반복"""
        chars = list(text)
        if chars:
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = chars[idx] * random.randint(2, 3)
        return "".join(chars)

    def _vowel_change(self, text: str) -> str:
        """모음 변형"""
        vowel_map = {"ㅏ": "ㅑ", "ㅓ": "ㅕ", "ㅗ": "ㅛ", "ㅜ": "ㅠ", "ㅡ": "ㅢ"}
        return "".join(vowel_map.get(c, c) for c in text)

    def _consonant_sub(self, text: str) -> str:
        """자음 치환"""
        cons_map = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ", "ㅅ": "ㅆ"}
        return "".join(cons_map.get(c, c) for c in text)

    def _reverse_partial(self, text: str) -> str:
        """부분 역순"""
        if len(text) < 4:
            return text
        mid = len(text) // 2
        return text[:mid][::-1] + text[mid:]

    def predict(self, model, tokenizer, texts: list) -> tuple:
        """예측 수행"""
        model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = texts[i : i + 32]
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs[:, 1].cpu().tolist())

        return all_preds, all_probs

    def generate_adversarial_samples(self) -> tuple:
        """적대적 샘플 생성"""
        samples = []
        labels = []
        weights = []

        # 독성 샘플에서 랜덤 선택
        indices = random.sample(
            range(len(self.train_texts)),
            min(self.samples_per_cycle, len(self.train_texts)),
        )

        for idx in indices:
            text = self.train_texts[idx]
            strategy = random.choice(self.attack_strategies)
            attacked = self._apply_attack(text, strategy)
            samples.append(attacked)
            labels.append(1)  # 독성
            weights.append(self.fn_weight)  # FN 방지 가중치

        return samples, labels, weights

    def calculate_diversity_score(self) -> float:
        """Phase2와의 다양성 점수 계산"""
        # 검증 데이터에서 샘플링
        sample_size = min(500, len(self.val_texts))
        indices = random.sample(range(len(self.val_texts)), sample_size)
        texts = [self.val_texts[i] for i in indices]
        labels = [self.val_labels[i] for i in indices]

        # 양쪽 모델 예측
        coevo_preds, _ = self.predict(self.coevo_model, self.coevo_tokenizer, texts)
        phase2_preds, _ = self.predict(self.phase2_model, self.phase2_tokenizer, texts)

        # 다양성 = 예측이 다른 비율
        diff_count = sum(1 for c, p in zip(coevo_preds, phase2_preds) if c != p)
        diversity = diff_count / len(texts)

        # Coevo의 FN 계산 (독성인데 정상으로 예측)
        fn_count = sum(
            1 for pred, label in zip(coevo_preds, labels) if label == 1 and pred == 0
        )

        return diversity, fn_count

    def run_attack_cycle(self) -> dict:
        """공격 사이클 실행"""
        # 적대적 샘플 생성
        samples, labels, weights = self.generate_adversarial_samples()

        # Coevo 모델로 예측
        preds, probs = self.predict(self.coevo_model, self.coevo_tokenizer, samples)

        # 회피율 계산 (독성인데 정상으로 예측된 비율)
        evasions = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        evasion_rate = evasions / len(samples) if samples else 0

        # 회피 성공한 샘플 수집 (Hard Negative)
        hard_samples = []
        hard_labels = []
        hard_weights = []

        for i, (pred, label, prob) in enumerate(zip(preds, labels, probs)):
            if pred == 0 and label == 1:  # FN (회피 성공)
                hard_samples.append(samples[i])
                hard_labels.append(label)
                hard_weights.append(self.fn_weight * 1.5)  # 추가 가중치
            elif pred == 1 and label == 1 and prob < 0.7:  # 경계 케이스
                hard_samples.append(samples[i])
                hard_labels.append(label)
                hard_weights.append(self.fn_weight)

        return {
            "evasion_rate": evasion_rate,
            "evasions": evasions,
            "total": len(samples),
            "hard_samples": hard_samples,
            "hard_labels": hard_labels,
            "hard_weights": hard_weights,
        }

    def retrain(self, samples: list, labels: list, weights: list):
        """모델 재학습"""
        if not samples:
            return

        logger.info(f"Retraining with {len(samples)} samples...")

        dataset = WeightedToxicDataset(
            samples, labels, weights, self.coevo_tokenizer, max_length=128
        )

        training_args = TrainingArguments(
            output_dir="./tmp_trainer",
            num_train_epochs=self.epochs_per_retrain,
            per_device_train_batch_size=16,
            learning_rate=1e-5,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            report_to="none",
        )

        trainer = WeightedTrainer(
            fn_weight=self.fn_weight,
            fp_weight=self.fp_weight,
            model=self.coevo_model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()
        self.stats["retrains"] += 1
        self.stats["total_samples"] += len(samples)

    def evaluate(self) -> dict:
        """전체 검증 세트 평가"""
        preds, probs = self.predict(
            self.coevo_model, self.coevo_tokenizer, self.val_texts
        )
        cm = confusion_matrix(self.val_labels, preds)
        f1 = f1_score(self.val_labels, preds, average="weighted")

        return {
            "f1": f1,
            "fp": cm[0][1] if cm.shape[0] > 1 else 0,
            "fn": cm[1][0] if cm.shape[0] > 1 else 0,
            "accuracy": (cm[0][0] + cm[1][1]) / sum(sum(cm)),
        }

    def save_model(self, tag: str = ""):
        """모델 저장"""
        save_path = self.coevo_path
        self.coevo_model.save_pretrained(save_path)
        self.coevo_tokenizer.save_pretrained(save_path)

        # 버전 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = Path("models/coevolution/versions") / f"v_{timestamp}_diversity{tag}"
        version_dir.mkdir(parents=True, exist_ok=True)
        self.coevo_model.save_pretrained(version_dir)
        self.coevo_tokenizer.save_pretrained(version_dir)

        # 메타데이터 저장
        meta = {
            "timestamp": timestamp,
            "stats": self.stats,
            "fn_weight": self.fn_weight,
            "fp_weight": self.fp_weight,
        }
        with open(version_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"Model saved to {save_path} and {version_dir}")

    def run(self, max_cycles: int = 1000, max_hours: float = 3.0, target_fn: int = 30):
        """다양성 유지 공진화 실행"""
        start_time = time.time()
        max_seconds = max_hours * 3600

        logger.info("=" * 60)
        logger.info("DIVERSITY-PRESERVING COEVOLUTION")
        logger.info("=" * 60)
        logger.info(f"Max cycles: {max_cycles}")
        logger.info(f"Max hours: {max_hours}")
        logger.info(f"Target FN: {target_fn}")
        logger.info(f"FN weight: {self.fn_weight}")
        logger.info(f"Samples per cycle: {self.samples_per_cycle}")
        logger.info(f"Epochs per retrain: {self.epochs_per_retrain}")
        logger.info("=" * 60)

        # 초기 평가
        eval_result = self.evaluate()
        diversity, fn_count = self.calculate_diversity_score()
        logger.info(
            f"[Initial] F1={eval_result['f1']:.4f}, FP={eval_result['fp']}, "
            f"FN={eval_result['fn']}, Diversity={diversity:.2%}"
        )

        accumulated_samples = []
        accumulated_labels = []
        accumulated_weights = []
        retrain_threshold = 100  # 100개 모이면 재학습

        for cycle in range(1, max_cycles + 1):
            elapsed = time.time() - start_time

            # 시간 체크
            if elapsed > max_seconds:
                logger.info(f"Time limit reached: {elapsed/3600:.2f} hours")
                break

            # 공격 사이클
            result = self.run_attack_cycle()
            evasion_rate = result["evasion_rate"]

            self.stats["cycles"] = cycle
            self.stats["evasion_history"].append(evasion_rate)

            # Hard sample 축적
            accumulated_samples.extend(result["hard_samples"])
            accumulated_labels.extend(result["hard_labels"])
            accumulated_weights.extend(result["hard_weights"])

            # 주기적 로깅
            if cycle % 10 == 0:
                diversity, fn_count = self.calculate_diversity_score()
                self.stats["diversity_score_history"].append(diversity)
                self.stats["fn_count_history"].append(fn_count)

                elapsed_min = elapsed / 60
                logger.info(
                    f"[Cycle {cycle:4d}] evasion={evasion_rate:.1%} | "
                    f"accumulated={len(accumulated_samples)} | "
                    f"diversity={diversity:.1%} | "
                    f"retrains={self.stats['retrains']} | "
                    f"elapsed={elapsed_min:.1f}min"
                )

            # 재학습 조건
            should_retrain = False

            # 1. 샘플이 충분히 쌓였을 때
            if len(accumulated_samples) >= retrain_threshold:
                should_retrain = True

            # 2. 회피율이 높을 때 (다양성 유지를 위해)
            if evasion_rate >= self.min_evasion and len(accumulated_samples) >= 50:
                should_retrain = True

            if should_retrain:
                self.retrain(accumulated_samples, accumulated_labels, accumulated_weights)
                accumulated_samples = []
                accumulated_labels = []
                accumulated_weights = []

                # 재학습 후 평가
                eval_result = self.evaluate()
                diversity, fn_count = self.calculate_diversity_score()
                logger.info(
                    f"[Retrain {self.stats['retrains']}] "
                    f"F1={eval_result['f1']:.4f}, FP={eval_result['fp']}, "
                    f"FN={eval_result['fn']}, Diversity={diversity:.2%}"
                )

                # 목표 FN 달성 체크
                if eval_result["fn"] <= target_fn:
                    logger.info(f"Target FN ({target_fn}) achieved!")

            # 주기적 저장 (30분마다)
            if cycle % 500 == 0:
                self.save_model(f"_c{cycle}")

        # 최종 저장
        self.save_model("_final")

        # 최종 평가
        eval_result = self.evaluate()
        diversity, fn_count = self.calculate_diversity_score()
        total_time = (time.time() - start_time) / 60

        logger.info("")
        logger.info("=" * 60)
        logger.info("DIVERSITY COEVOLUTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total cycles: {self.stats['cycles']}")
        logger.info(f"Total time: {total_time:.1f} minutes")
        logger.info(f"Total retrains: {self.stats['retrains']}")
        logger.info(f"Total samples trained: {self.stats['total_samples']}")
        logger.info(f"Final F1: {eval_result['f1']:.4f}")
        logger.info(f"Final FP: {eval_result['fp']}")
        logger.info(f"Final FN: {eval_result['fn']}")
        logger.info(f"Final Diversity: {diversity:.2%}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Diversity-Preserving Coevolution")
    parser.add_argument("--max-cycles", type=int, default=5000, help="Max cycles")
    parser.add_argument("--max-hours", type=float, default=3.0, help="Max hours to run")
    parser.add_argument("--target-fn", type=int, default=30, help="Target FN count")
    parser.add_argument("--fn-weight", type=float, default=3.0, help="FN loss weight")
    parser.add_argument(
        "--samples-per-cycle", type=int, default=500, help="Samples per cycle"
    )
    parser.add_argument(
        "--epochs-per-retrain", type=int, default=3, help="Epochs per retrain"
    )
    args = parser.parse_args()

    coevo = DiversityCoevolution(
        fn_weight=args.fn_weight,
        samples_per_cycle=args.samples_per_cycle,
        epochs_per_retrain=args.epochs_per_retrain,
    )

    coevo.run(
        max_cycles=args.max_cycles,
        max_hours=args.max_hours,
        target_fn=args.target_fn,
    )


if __name__ == "__main__":
    main()
