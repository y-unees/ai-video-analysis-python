from __future__ import annotations

ENCODER_REGISTRY = {
    "CLIP-16": {
        "encoder_name": "CLIP-16",
        "provider": "huggingface",
        "model_identifier": "openai/clip-vit-base-patch16",
        "implementation_class": "transformers.CLIPVisionModel",
        "weight_variant": "from_pretrained default",
        "input_size": [224, 224],
        "normalization_profile": "imagenet",
        "asset_loader": "CLIPVisionModel.from_pretrained",
        "upstream_mapping_status": "verified",
        "support_status": "implemented_unverified",
    },
    "CLIP-32": {
        "encoder_name": "CLIP-32",
        "provider": "huggingface",
        "model_identifier": "openai/clip-vit-base-patch32",
        "implementation_class": "transformers.CLIPVisionModel",
        "weight_variant": "from_pretrained default",
        "input_size": [224, 224],
        "normalization_profile": "imagenet",
        "asset_loader": "CLIPVisionModel.from_pretrained",
        "upstream_mapping_status": "verified",
        "support_status": "implemented_unverified",
    },
    "XCLIP-16": {
        "encoder_name": "XCLIP-16",
        "provider": "huggingface",
        "model_identifier": "microsoft/xclip-base-patch16",
        "implementation_class": "transformers.XCLIPVisionModel",
        "weight_variant": "from_pretrained default",
        "input_size": [224, 224],
        "normalization_profile": "imagenet",
        "asset_loader": "XCLIPVisionModel.from_pretrained",
        "upstream_mapping_status": "verified",
        "support_status": "implemented_unverified",
    },
    "XCLIP-32": {
        "encoder_name": "XCLIP-32",
        "provider": "huggingface",
        "model_identifier": "microsoft/xclip-base-patch32",
        "implementation_class": "transformers.XCLIPVisionModel",
        "weight_variant": "from_pretrained default",
        "input_size": [224, 224],
        "normalization_profile": "imagenet",
        "asset_loader": "XCLIPVisionModel.from_pretrained",
        "upstream_mapping_status": "verified",
        "support_status": "implemented_unverified",
    },
    "DINO-base": {
        "encoder_name": "DINO-base",
        "provider": "huggingface",
        "model_identifier": "facebook/dinov2-base",
        "implementation_class": "transformers.AutoModel",
        "weight_variant": "from_pretrained default",
        "input_size": [224, 224],
        "normalization_profile": "imagenet",
        "asset_loader": "AutoModel.from_pretrained",
        "upstream_mapping_status": "verified",
        "support_status": "implemented_unverified",
    },
    "DINO-large": {
        "encoder_name": "DINO-large",
        "provider": "huggingface",
        "model_identifier": "facebook/dinov2-large",
        "implementation_class": "transformers.AutoModel",
        "weight_variant": "from_pretrained default",
        "input_size": [224, 224],
        "normalization_profile": "imagenet",
        "asset_loader": "AutoModel.from_pretrained",
        "upstream_mapping_status": "verified",
        "support_status": "implemented_unverified",
    },
    "ResNet-18": {
        "encoder_name": "ResNet-18",
        "provider": "torchvision",
        "model_identifier": "torchvision/resnet18",
        "implementation_class": "torchvision.models.resnet18",
        "weight_variant": "DEFAULT when downloads are allowed, None otherwise",
        "input_size": [224, 224],
        "normalization_profile": "imagenet",
        "asset_loader": "torchvision.models.resnet18",
        "upstream_mapping_status": "verified",
        "support_status": "implemented_unverified",
    },
    "VGG-16": {
        "encoder_name": "VGG-16",
        "provider": "torchvision",
        "model_identifier": "torchvision/vgg16",
        "implementation_class": "torchvision.models.vgg16",
        "weight_variant": "DEFAULT when downloads are allowed, None otherwise",
        "input_size": [224, 224],
        "normalization_profile": "imagenet",
        "asset_loader": "torchvision.models.vgg16",
        "upstream_mapping_status": "verified",
        "support_status": "implemented_unverified",
    },
    "EfficientNet-b4": {
        "encoder_name": "EfficientNet-b4",
        "provider": "torchvision",
        "model_identifier": "torchvision/efficientnet_b4",
        "implementation_class": "torchvision.models.efficientnet_b4",
        "weight_variant": "DEFAULT when downloads are allowed, None otherwise",
        "input_size": [224, 224],
        "normalization_profile": "imagenet",
        "asset_loader": "torchvision.models.efficientnet_b4",
        "upstream_mapping_status": "verified",
        "support_status": "implemented_unverified",
    },
    "MobileNet-v3": {
        "encoder_name": "MobileNet-v3",
        "provider": "timm",
        "model_identifier": "mobilenetv3_large_100",
        "implementation_class": "timm.create_model",
        "weight_variant": "pretrained=True when downloads are allowed, False otherwise",
        "input_size": [224, 224],
        "normalization_profile": "imagenet",
        "asset_loader": "timm.create_model",
        "upstream_mapping_status": "verified",
        "support_status": "implemented_unverified",
    },
}

SUPPORTED_ENCODERS = {name: item["model_identifier"] for name, item in ENCODER_REGISTRY.items()}
TRANSFORMER_ENCODERS = {"CLIP-16", "CLIP-32", "XCLIP-16", "XCLIP-32", "DINO-base", "DINO-large"}
SUPPORTED_DISTANCES = {"l2", "cos"}

D3_UPSTREAM_REPOSITORY = "https://github.com/Zig-HS/D3"
D3_UPSTREAM_COMMIT = "c798fbc57fe0c4198d63a73732c2c0f9e4b4816c"
D3_UPSTREAM_COMMIT_DATE = "2026-06-23T16:44:02+08:00"
D3_UPSTREAM_LICENSE = "MIT"
D3_PAPER_REFERENCE = "arXiv:2508.00701"
D3_DETECTOR_VERSION = "0.8.1-upstream-c798fbc-adapter"
D3_RESULT_SCHEMA_VERSION = "0.8"

METHOD_VERIFICATION = {
    "upstream_commit": D3_UPSTREAM_COMMIT,
    "preprocessing_parity": "adapted",
    "mathematical_parity": "verified_synthetic_tensor",
    "runtime_parity": "not_verified",
    "score_direction_status": "conflicting",
    "deviations": [
        "Local single-video preprocessing decodes directly with OpenCV instead of upstream ffmpeg-to-folder plus cv2.imread.",
        "Actual pretrained XCLIP-16 runtime parity has not been executed in this environment.",
        "Upstream AP target uses 1 - label after real=0/fake=1 CSV labels, so score direction remains neutral in this application.",
    ],
}

SCORE_DIRECTION_RECORD = {
    "status": "conflicting",
    "higher_score_indicates": "unknown",
    "evidence": [
        {
            "source": "upstream_file",
            "file": "utils/folder2csv.py",
            "description": "Real videos are assigned label 0 and AI videos are assigned label 1.",
        },
        {
            "source": "upstream_file",
            "file": "eval.py",
            "description": "average_precision_score is called with 1 - y_true as the AP target and batch_dis_std as the prediction.",
        },
    ],
    "notes": [
        "The upstream evaluation target therefore treats higher scores as positive for real videos under its CSV labels.",
        "The project docs and paper motivation discuss generated-video temporal artifacts, so this adapter does not expose a verified synthetic/real direction.",
    ],
}
