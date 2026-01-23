"""CNN-Enhanced Transformer Classifier for Korean Toxic Text Detection.

Architecture:
    Transformer (KcELECTRA) → CNN Layers → Combined Classification

The CNN layers capture local n-gram patterns (curse words, slang) while
the Transformer captures global context. This hybrid approach improves
detection of obfuscated toxic expressions.

Reference:
    - Nature 2025: Adaptive ensemble techniques for hate speech detection
    - arXiv 2025: Three-Layer LoRA-Tuned BERTweet Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
)


class CNNEnhancedClassifier(nn.Module):
    """Transformer + CNN hybrid model for toxic text classification.

    Architecture:
        Input Text
            ↓
        Transformer Encoder (KcELECTRA, 12 layers)
            ↓
        [CLS] Token + All Hidden States
            ↓
        Multi-Scale CNN (kernel 2,3,4,5)
            ↓
        MaxPool + Concatenate
            ↓
        Combined Features (768 + 512)
            ↓
        Classification Head
            ↓
        Output (clean/toxic)

    Benefits:
        - CNN captures local n-gram patterns (욕설, 비속어)
        - Transformer captures global context
        - Multi-scale kernels detect various pattern lengths
    """

    def __init__(
        self,
        model_name: str = "beomi/KcELECTRA-base-v2022",
        num_labels: int = 2,
        hidden_size: int = 768,
        cnn_filters: int = 128,
        kernel_sizes: List[int] = None,
        dropout: float = 0.3,
        freeze_transformer_layers: int = 0,
    ):
        """Initialize CNN-Enhanced Classifier.

        Args:
            model_name: Pretrained transformer model name
            num_labels: Number of output classes (2 for binary)
            hidden_size: Transformer hidden size (768 for base models)
            cnn_filters: Number of filters per CNN kernel
            kernel_sizes: List of CNN kernel sizes (default: [2,3,4,5])
            dropout: Dropout probability
            freeze_transformer_layers: Number of transformer layers to freeze (0=none)
        """
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]

        self.kernel_sizes = kernel_sizes
        self.num_labels = num_labels

        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config

        # Optionally freeze some transformer layers
        if freeze_transformer_layers > 0:
            for layer in self.transformer.encoder.layer[:freeze_transformer_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Multi-scale CNN layers for n-gram pattern detection
        # Each kernel size captures different pattern lengths:
        # - kernel=2: bigram (시발, ㅅㅂ)
        # - kernel=3: trigram (씨발놈, 개새끼)
        # - kernel=4: 4-gram (longer patterns)
        # - kernel=5: 5-gram (complex expressions)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=cnn_filters,
                kernel_size=k,
                padding=k // 2,  # same padding
            )
            for k in kernel_sizes
        ])

        # Batch normalization for CNN outputs
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(cnn_filters) for _ in kernel_sizes
        ])

        # Combined feature size: CLS (768) + CNN (128 * 4 kernels)
        cnn_output_size = cnn_filters * len(kernel_sizes)
        combined_size = hidden_size + cnn_output_size

        # Classification head with deeper layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_size, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_labels),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize CNN and classifier weights."""
        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Token type IDs [batch, seq_len]
            labels: Ground truth labels [batch] (optional, for training)

        Returns:
            Dict with 'logits' and optionally 'loss'
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        # [CLS] token representation (global context)
        cls_output = transformer_outputs.last_hidden_state[:, 0, :]  # [batch, 768]

        # All token representations for CNN
        # [batch, seq_len, hidden_size]
        sequence_output = transformer_outputs.last_hidden_state

        # Transpose for Conv1d: [batch, hidden_size, seq_len]
        sequence_output = sequence_output.transpose(1, 2)

        # Apply multi-scale CNN
        cnn_outputs = []
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            # Conv1d → BatchNorm → ReLU → MaxPool
            x = conv(sequence_output)  # [batch, filters, seq_len]
            x = bn(x)
            x = F.relu(x)
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # [batch, filters]
            cnn_outputs.append(x)

        # Concatenate CNN outputs
        cnn_features = torch.cat(cnn_outputs, dim=1)  # [batch, filters * num_kernels]

        # Combine CLS and CNN features
        combined = torch.cat([cls_output, cnn_features], dim=1)  # [batch, 768 + 512]

        # Classification
        logits = self.classifier(combined)  # [batch, num_labels]

        output = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            output["loss"] = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return output

    def get_feature_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Get feature importance scores for interpretability.

        Returns CNN activation patterns to show which n-grams are important.
        """
        with torch.no_grad():
            transformer_outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            sequence_output = transformer_outputs.last_hidden_state.transpose(1, 2)

            activations = {}
            for i, (conv, kernel_size) in enumerate(zip(self.conv_layers, self.kernel_sizes)):
                x = conv(sequence_output)
                x = F.relu(x)
                # Get max activation position
                max_vals, max_indices = x.max(dim=2)
                activations[f"kernel_{kernel_size}"] = {
                    "max_values": max_vals,
                    "max_positions": max_indices,
                }

            return activations


class CNNEnhancedInference:
    """Inference wrapper for CNN-Enhanced Classifier."""

    def __init__(
        self,
        model_path: str = None,
        model_name: str = "beomi/KcELECTRA-base-v2022",
        device: str = None,
    ):
        """Initialize inference.

        Args:
            model_path: Path to trained model weights (None for pretrained only)
            model_name: Base transformer model name
            device: Device to use
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create model
        self.model = CNNEnhancedClassifier(model_name=model_name)

        # Load trained weights if provided
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")

        self.model.to(self.device)
        self.model.eval()
        print(f"CNN-Enhanced Classifier ready on {self.device}")

    def predict(
        self,
        texts: Union[str, List[str]],
        max_length: int = 256,
        threshold: float = 0.5,
    ) -> Union[Dict, List[Dict]]:
        """Predict toxicity.

        Args:
            texts: Input text(s)
            max_length: Max sequence length
            threshold: Classification threshold

        Returns:
            Prediction dict(s)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = F.softmax(outputs["logits"], dim=-1)
            toxic_probs = probs[:, 1].cpu().numpy()

        results = []
        for i, toxic_prob in enumerate(toxic_probs):
            label = 1 if toxic_prob > threshold else 0
            results.append({
                "label": label,
                "label_text": "toxic" if label == 1 else "clean",
                "confidence": round(float(toxic_prob if label == 1 else 1 - toxic_prob), 4),
                "toxic_prob": round(float(toxic_prob), 4),
            })

        return results[0] if single_input else results


def create_cnn_model(
    model_name: str = "beomi/KcELECTRA-base-v2022",
    **kwargs,
) -> CNNEnhancedClassifier:
    """Factory function to create CNN-Enhanced model.

    Args:
        model_name: Base transformer model
        **kwargs: Additional arguments for CNNEnhancedClassifier

    Returns:
        CNNEnhancedClassifier instance
    """
    return CNNEnhancedClassifier(model_name=model_name, **kwargs)


# Quick test
if __name__ == "__main__":
    print("Creating CNN-Enhanced Classifier...")
    model = CNNEnhancedClassifier()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cnn_params = sum(p.numel() for p in model.conv_layers.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"\nModel Architecture:")
    print(f"  Transformer: KcELECTRA-base-v2022 (12 layers, 768 hidden)")
    print(f"  CNN Kernels: {model.kernel_sizes}")
    print(f"  CNN Filters: 128 per kernel")
    print(f"  Combined Features: 768 (CLS) + 512 (CNN) = 1280")
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  CNN Layers: {cnn_params:,}")
    print(f"  Classifier: {classifier_params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randint(0, 1000, (2, 64))
    dummy_mask = torch.ones(2, 64)

    with torch.no_grad():
        output = model(dummy_input, dummy_mask)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output logits shape: {output['logits'].shape}")
    print(f"  Output logits: {output['logits']}")
    print("\n✅ CNN-Enhanced Classifier ready!")
