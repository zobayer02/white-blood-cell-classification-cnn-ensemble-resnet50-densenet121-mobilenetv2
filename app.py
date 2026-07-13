# ===============================
# WBC Classification Flask App
# Phase E-1: Ensemble Prediction
# ===============================

import io
import os
import textwrap
import uuid
from threading import Lock
from typing import Any, cast

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image

from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from lime import lime_image
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import mark_boundaries

from tf_keras import layers, models
from tf_keras.applications.densenet import preprocess_input as densenet_preprocess
from tf_keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tf_keras.applications.resnet50 import preprocess_input as resnet_preprocess

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===============================
# Flask Setup
# ===============================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PORTABLE_MODEL_DIR = os.path.join(
    BASE_DIR,
    "_portable_build",
    "WBC_Flask_App_Portable",
    "models"
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")
LIME_FOLDER = os.path.join(BASE_DIR, "static", "lime")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
os.makedirs(LIME_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER
app.config["LIME_FOLDER"] = LIME_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ===============================
# Model Paths
# ===============================

def resolve_model_path(env_var_name, filename):
    override_path = os.getenv(env_var_name)
    candidate_paths = [
        override_path,
        os.path.join(MODEL_DIR, filename),
        os.path.join(PORTABLE_MODEL_DIR, filename)
    ]

    for candidate_path in candidate_paths:
        if candidate_path and os.path.exists(candidate_path):
            return candidate_path

    raise FileNotFoundError(
        f"Missing model file '{filename}'. "
        f"Place it in '{MODEL_DIR}' or set '{env_var_name}'."
    )


RESNET_MODEL_PATH = resolve_model_path(
    "RESNET_MODEL_PATH",
    "resnet50_wbc_final.keras"
)

DENSENET_MODEL_PATH = resolve_model_path(
    "DENSENET_MODEL_PATH",
    "densenet121_wbc_retuned_final.keras"
)

MOBILENET_MODEL_PATH = resolve_model_path(
    "MOBILENET_MODEL_PATH",
    "mobilenetv2_wbc_retune_v3_final.keras"
)


# ===============================
# Class Names
# ===============================

CLASS_NAMES = [
    "Basophil",
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil"
]


# ===============================
# Ensemble Weights
# ===============================

W_RESNET = 0.50
W_DENSENET = 0.35
W_MOBILENET = 0.15


# ===============================
# Load Models Once
# ===============================

print("Loading models...")

resnet_model: Any = models.load_model(RESNET_MODEL_PATH, compile=False)
densenet_model: Any = models.load_model(DENSENET_MODEL_PATH, compile=False)
mobilenet_model: Any = models.load_model(MOBILENET_MODEL_PATH, compile=False)

print("All models loaded successfully.")


# ===============================
# Helper Functions
# ===============================

IMG_SIZE = (224, 224)
LIME_NUM_SAMPLES = 300
JOB_PROGRESS = {}
JOB_PROGRESS_LOCK = Lock()
REPORT_CACHE = {}
REPORT_CACHE_LOCK = Lock()
MAX_CACHED_REPORTS = 20


def update_job_progress(job_id, percentage, message, step):
    if not job_id:
        return

    with JOB_PROGRESS_LOCK:
        JOB_PROGRESS[job_id] = {
            "percentage": max(0, min(100, int(percentage))),
            "message": message,
            "step": step
        }


def get_job_progress(job_id):
    with JOB_PROGRESS_LOCK:
        return JOB_PROGRESS.get(
            job_id,
            {
                "percentage": 0,
                "message": "Waiting for processing to start...",
                "step": 0
            }
        ).copy()


def cache_report_data(report_data):
    report_id = uuid.uuid4().hex

    with REPORT_CACHE_LOCK:
        if len(REPORT_CACHE) >= MAX_CACHED_REPORTS:
            oldest_report_id = next(iter(REPORT_CACHE))
            REPORT_CACHE.pop(oldest_report_id, None)

        REPORT_CACHE[report_id] = report_data

    return report_id


def get_cached_report(report_id):
    with REPORT_CACHE_LOCK:
        return REPORT_CACHE.get(report_id)


def wrap_text_lines(text, width=70):
    return textwrap.wrap(str(text), width=width) or [""]


def draw_pdf_image(fig, rect, image_path, title):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_facecolor("#f8fbfd")
    ax.text(
        0,
        1.03,
        title,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="#102a43"
    )

    if image_path and os.path.exists(image_path):
        with Image.open(image_path) as image:
            ax.imshow(np.array(image.convert("RGB")))
    else:
        ax.text(
            0.5,
            0.5,
            "Image unavailable",
            ha="center",
            va="center",
            fontsize=11,
            color="#627d98"
        )


def add_pdf_metric(fig, x, y, label, value):
    fig.text(x, y, label, fontsize=10, color="#627d98")
    fig.text(x, y - 0.03, value, fontsize=18, fontweight="bold", color="#102a43")


def build_report_pdf(report_data):
    results = report_data["results"]
    reliability = report_data["reliability"]
    weights = report_data["weights"]
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(
            0.07,
            0.965,
            "EXPLAINABLE ENSEMBLE ANALYSIS",
            fontsize=11,
            fontweight="bold",
            color="#1769aa"
        )
        fig.text(
            0.07,
            0.93,
            "WBC Classification Report",
            fontsize=25,
            fontweight="bold",
            color="#102a43"
        )

        draw_pdf_image(
            fig,
            [0.07, 0.60, 0.34, 0.23],
            report_data["image_path"],
            "Uploaded WBC Image"
        )

        fig.text(
            0.46,
            0.83,
            "Weighted Soft Voting Result",
            fontsize=13,
            fontweight="bold",
            color="#102a43"
        )
        fig.text(
            0.46,
            0.75,
            results["ensemble"]["pred_class"],
            fontsize=31,
            fontweight="bold",
            color="#0f4f82"
        )
        fig.text(
            0.46,
            0.70,
            f'Confidence: {results["ensemble"]["confidence_percent"]}%',
            fontsize=15,
            fontweight="bold",
            color="#243b53"
        )
        fig.text(
            0.46,
            0.665,
            f'Reliability Flag: {reliability["flag"]}',
            fontsize=13,
            color="#243b53"
        )
        fig.text(
            0.46,
            0.635,
            f'Model Agreement: {reliability["agreement_label"]}',
            fontsize=13,
            color="#243b53"
        )

        reason_y = 0.595
        for line in wrap_text_lines(
            f'Reliability Reason: {reliability["reason"]}',
            width=46
        ):
            fig.text(0.46, reason_y, line, fontsize=11, color="#627d98")
            reason_y -= 0.022

        fig.text(
            0.07,
            0.52,
            "Individual Model Predictions",
            fontsize=15,
            fontweight="bold",
            color="#102a43"
        )
        table_ax = fig.add_axes([0.07, 0.25, 0.86, 0.22])
        table_ax.axis("off")
        model_table = table_ax.table(
            cellText=[
                [
                    "ResNet50",
                    results["resnet"]["pred_class"],
                    f'{results["resnet"]["confidence"] * 100:.2f}%',
                    f'{weights["ResNet50"]:.2f}'
                ],
                [
                    "DenseNet121",
                    results["densenet"]["pred_class"],
                    f'{results["densenet"]["confidence"] * 100:.2f}%',
                    f'{weights["DenseNet121"]:.2f}'
                ],
                [
                    "MobileNetV2",
                    results["mobilenet"]["pred_class"],
                    f'{results["mobilenet"]["confidence"] * 100:.2f}%',
                    f'{weights["MobileNetV2"]:.2f}'
                ]
            ],
            colLabels=[
                "Model",
                "Predicted Class",
                "Confidence",
                "Ensemble Weight"
            ],
            loc="center",
            cellLoc="left"
        )
        model_table.auto_set_font_size(False)
        model_table.set_fontsize(10)
        model_table.scale(1, 1.8)
        for (row, col), cell in model_table.get_celld().items():
            cell.set_edgecolor("#d9e2ec")
            if row == 0:
                cell.set_facecolor("#edf4f8")
                cell.set_text_props(weight="bold", color="#102a43")
            else:
                cell.set_facecolor("#ffffff")
                cell.set_text_props(color="#243b53")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(
            0.07,
            0.95,
            "Class Probability Distribution",
            fontsize=18,
            fontweight="bold",
            color="#102a43"
        )

        probability_ax = fig.add_axes([0.11, 0.58, 0.78, 0.25])
        probabilities = results["ensemble"]["class_probabilities"]
        labels = [item["class_name"] for item in probabilities][::-1]
        values = [float(item["percentage"]) for item in probabilities][::-1]
        probability_ax.barh(labels, values, color="#1769aa")
        probability_ax.set_xlim(0, 100)
        probability_ax.set_xlabel("Probability (%)")
        probability_ax.set_facecolor("#f8fbfd")
        probability_ax.grid(axis="x", linestyle="--", alpha=0.25)
        probability_ax.spines["top"].set_visible(False)
        probability_ax.spines["right"].set_visible(False)

        fig.text(
            0.07,
            0.46,
            "QEEF Reliability Analysis",
            fontsize=18,
            fontweight="bold",
            color="#102a43"
        )
        add_pdf_metric(
            fig,
            0.08,
            0.40,
            "QEEF Trust Score",
            (
                f'{reliability["qeef_trust_percentage"]}%'
                if reliability["qeef_trust_percentage"] is not None
                else "Unavailable"
            )
        )
        add_pdf_metric(
            fig,
            0.39,
            0.40,
            "Ensemble Confidence",
            f'{reliability["ensemble_confidence"] * 100:.2f}%'
        )
        add_pdf_metric(
            fig,
            0.68,
            0.40,
            "Model Agreement",
            f'{reliability["agreement_label"]} ({reliability["agreement_score"]:.2f})'
        )
        add_pdf_metric(
            fig,
            0.08,
            0.28,
            "Grad-CAM Cosine Similarity",
            (
                f'{reliability["average_cosine_similarity"]:.4f}'
                if reliability["average_cosine_similarity"] is not None
                else "Unavailable"
            )
        )
        add_pdf_metric(
            fig,
            0.39,
            0.28,
            "Grad-CAM SSIM Similarity",
            (
                f'{reliability["average_ssim_similarity"]:.4f}'
                if reliability["average_ssim_similarity"] is not None
                else "Unavailable"
            )
        )
        add_pdf_metric(
            fig,
            0.68,
            0.28,
            "Final Reliability Flag",
            reliability["flag"]
        )

        fig.text(
            0.07,
            0.14,
            "Reason",
            fontsize=13,
            fontweight="bold",
            color="#102a43"
        )
        reason_block_ax = fig.add_axes([0.07, 0.05, 0.86, 0.08])
        reason_block_ax.axis("off")
        reason_block_ax.set_facecolor("#f5f8fa")
        reason_block_ax.text(
            0.02,
            0.72,
            "\n".join(wrap_text_lines(reliability["reason"], width=92)),
            fontsize=11,
            color="#334e68",
            va="top"
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(
            0.07,
            0.95,
            "Grad-CAM and LIME Explainability",
            fontsize=18,
            fontweight="bold",
            color="#102a43"
        )
        draw_pdf_image(
            fig,
            [0.07, 0.60, 0.25, 0.23],
            report_data["gradcam_paths"].get("resnet"),
            "ResNet50 Grad-CAM"
        )
        draw_pdf_image(
            fig,
            [0.375, 0.60, 0.25, 0.23],
            report_data["gradcam_paths"].get("densenet"),
            "DenseNet121 Grad-CAM"
        )
        draw_pdf_image(
            fig,
            [0.68, 0.60, 0.25, 0.23],
            report_data["gradcam_paths"].get("mobilenet"),
            "MobileNetV2 Grad-CAM"
        )

        if report_data.get("lime_path"):
            draw_pdf_image(
                fig,
                [0.18, 0.18, 0.64, 0.28],
                report_data["lime_path"],
                "LIME Explanation"
            )
        else:
            lime_ax = fig.add_axes([0.12, 0.20, 0.76, 0.18])
            lime_ax.axis("off")
            lime_ax.text(
                0.5,
                0.5,
                "LIME explanation could not be generated for this report.",
                ha="center",
                va="center",
                fontsize=12,
                color="#b42318"
            )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer


def build_report_png(report_data):
    results = report_data["results"]
    reliability = report_data["reliability"]
    weights = report_data["weights"]
    buffer = io.BytesIO()
    fig = plt.figure(figsize=(12, 26), facecolor="white")

    fig.text(
        0.06,
        0.975,
        "EXPLAINABLE ENSEMBLE ANALYSIS",
        fontsize=14,
        fontweight="bold",
        color="#1769aa"
    )
    fig.text(
        0.06,
        0.95,
        "WBC Classification Report",
        fontsize=32,
        fontweight="bold",
        color="#102a43"
    )

    draw_pdf_image(
        fig,
        [0.06, 0.77, 0.35, 0.14],
        report_data["image_path"],
        "Uploaded WBC Image"
    )
    fig.text(
        0.47,
        0.89,
        "Weighted Soft Voting Result",
        fontsize=16,
        fontweight="bold",
        color="#102a43"
    )
    fig.text(
        0.47,
        0.85,
        results["ensemble"]["pred_class"],
        fontsize=40,
        fontweight="bold",
        color="#0f4f82"
    )
    fig.text(
        0.47,
        0.82,
        f'Confidence: {results["ensemble"]["confidence_percent"]}%',
        fontsize=18,
        fontweight="bold",
        color="#243b53"
    )
    fig.text(
        0.47,
        0.795,
        f'Reliability: {reliability["flag"]}',
        fontsize=16,
        color="#243b53"
    )
    fig.text(
        0.47,
        0.775,
        f'Model Agreement: {reliability["agreement_label"]}',
        fontsize=15,
        color="#243b53"
    )
    fig.text(
        0.47,
        0.735,
        "\n".join(wrap_text_lines(reliability["reason"], width=55)),
        fontsize=13,
        color="#627d98"
    )

    fig.text(
        0.06,
        0.69,
        "Individual Model Predictions",
        fontsize=20,
        fontweight="bold",
        color="#102a43"
    )
    table_ax = fig.add_axes([0.06, 0.59, 0.88, 0.08])
    table_ax.axis("off")
    model_table = table_ax.table(
        cellText=[
            [
                "ResNet50",
                results["resnet"]["pred_class"],
                f'{results["resnet"]["confidence"] * 100:.2f}%',
                f'{weights["ResNet50"]:.2f}'
            ],
            [
                "DenseNet121",
                results["densenet"]["pred_class"],
                f'{results["densenet"]["confidence"] * 100:.2f}%',
                f'{weights["DenseNet121"]:.2f}'
            ],
            [
                "MobileNetV2",
                results["mobilenet"]["pred_class"],
                f'{results["mobilenet"]["confidence"] * 100:.2f}%',
                f'{weights["MobileNetV2"]:.2f}'
            ]
        ],
        colLabels=["Model", "Predicted Class", "Confidence", "Ensemble Weight"],
        loc="center",
        cellLoc="left"
    )
    model_table.auto_set_font_size(False)
    model_table.set_fontsize(12)
    model_table.scale(1, 2.1)
    for (row, col), cell in model_table.get_celld().items():
        cell.set_edgecolor("#d9e2ec")
        if row == 0:
            cell.set_facecolor("#edf4f8")
            cell.set_text_props(weight="bold", color="#102a43")

    fig.text(
        0.06,
        0.55,
        "Class Probability Distribution",
        fontsize=20,
        fontweight="bold",
        color="#102a43"
    )
    probability_ax = fig.add_axes([0.12, 0.41, 0.76, 0.12])
    probabilities = results["ensemble"]["class_probabilities"]
    probability_ax.barh(
        [item["class_name"] for item in probabilities][::-1],
        [float(item["percentage"]) for item in probabilities][::-1],
        color="#1769aa"
    )
    probability_ax.set_xlim(0, 100)
    probability_ax.set_xlabel("Probability (%)")
    probability_ax.grid(axis="x", linestyle="--", alpha=0.25)
    probability_ax.spines["top"].set_visible(False)
    probability_ax.spines["right"].set_visible(False)

    fig.text(
        0.06,
        0.37,
        "QEEF Reliability Analysis",
        fontsize=20,
        fontweight="bold",
        color="#102a43"
    )
    add_pdf_metric(
        fig,
        0.07,
        0.34,
        "QEEF Trust Score",
        (
            f'{reliability["qeef_trust_percentage"]}%'
            if reliability["qeef_trust_percentage"] is not None
            else "Unavailable"
        )
    )
    add_pdf_metric(
        fig,
        0.38,
        0.34,
        "Ensemble Confidence",
        f'{reliability["ensemble_confidence"] * 100:.2f}%'
    )
    add_pdf_metric(
        fig,
        0.68,
        0.34,
        "Model Agreement",
        f'{reliability["agreement_label"]} ({reliability["agreement_score"]:.2f})'
    )
    add_pdf_metric(
        fig,
        0.07,
        0.29,
        "Grad-CAM Cosine Similarity",
        (
            f'{reliability["average_cosine_similarity"]:.4f}'
            if reliability["average_cosine_similarity"] is not None
            else "Unavailable"
        )
    )
    add_pdf_metric(
        fig,
        0.38,
        0.29,
        "Grad-CAM SSIM Similarity",
        (
            f'{reliability["average_ssim_similarity"]:.4f}'
            if reliability["average_ssim_similarity"] is not None
            else "Unavailable"
        )
    )
    add_pdf_metric(fig, 0.68, 0.29, "Final Reliability Flag", reliability["flag"])

    fig.text(
        0.06,
        0.235,
        "Grad-CAM Explainability",
        fontsize=20,
        fontweight="bold",
        color="#102a43"
    )
    draw_pdf_image(
        fig,
        [0.06, 0.11, 0.26, 0.10],
        report_data["gradcam_paths"].get("resnet"),
        "ResNet50 Grad-CAM"
    )
    draw_pdf_image(
        fig,
        [0.37, 0.11, 0.26, 0.10],
        report_data["gradcam_paths"].get("densenet"),
        "DenseNet121 Grad-CAM"
    )
    draw_pdf_image(
        fig,
        [0.68, 0.11, 0.26, 0.10],
        report_data["gradcam_paths"].get("mobilenet"),
        "MobileNetV2 Grad-CAM"
    )
    draw_pdf_image(
        fig,
        [0.37, 0.015, 0.26, 0.07],
        report_data.get("lime_path"),
        "LIME Explanation"
    )

    fig.savefig(
        buffer,
        format="png",
        dpi=240,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.35
    )
    plt.close(fig)
    buffer.seek(0)
    return buffer


def load_image_for_prediction(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype(np.float32)
    return img_array


def preprocess_for_model(img_array, model_type):
    batch = np.expand_dims(img_array.copy(), axis=0)

    if model_type == "resnet":
        return resnet_preprocess(batch)

    if model_type == "densenet":
        return densenet_preprocess(batch)

    if model_type == "mobilenet":
        return mobilenet_preprocess(batch)

    raise ValueError("Invalid model_type")


def lime_preprocess_batch(images, model_type):
    images = images.astype(np.float32)

    if model_type == "resnet":
        return resnet_preprocess(images.copy())

    if model_type == "densenet":
        return densenet_preprocess(images.copy())

    if model_type == "mobilenet":
        return mobilenet_preprocess(images.copy())

    raise ValueError("Invalid model_type")


def ensemble_predict_for_lime(images):
    resnet_input = lime_preprocess_batch(images, "resnet")
    densenet_input = lime_preprocess_batch(images, "densenet")
    mobilenet_input = lime_preprocess_batch(images, "mobilenet")

    resnet_p = resnet_model.predict(resnet_input, verbose=0)
    densenet_p = densenet_model.predict(densenet_input, verbose=0)
    mobilenet_p = mobilenet_model.predict(mobilenet_input, verbose=0)

    return (
        W_RESNET * resnet_p +
        W_DENSENET * densenet_p +
        W_MOBILENET * mobilenet_p
    )


def generate_lime_explanation(
    image_path,
    ensemble_pred_idx,
    output_filename_prefix,
    progress_callback=None
):
    explainer = lime_image.LimeImageExplainer(random_state=42)
    img_array = load_image_for_prediction(image_path).astype(np.uint8)
    completed_samples = 0

    def classifier_fn(images):
        nonlocal completed_samples
        predictions = ensemble_predict_for_lime(images)
        completed_samples = min(
            LIME_NUM_SAMPLES,
            completed_samples + len(images)
        )

        if progress_callback:
            progress_callback(completed_samples, LIME_NUM_SAMPLES)

        return predictions

    explanation = explainer.explain_instance(
        image=img_array,
        classifier_fn=classifier_fn,
        top_labels=5,
        hide_color=0,
        num_samples=LIME_NUM_SAMPLES
    )

    temp, mask = explanation.get_image_and_mask(
        label=ensemble_pred_idx,
        positive_only=True,
        num_features=8,
        hide_rest=False
    )

    lime_overlay = mark_boundaries(temp / 255.0, mask)
    lime_filename = f"{output_filename_prefix}_lime.png"
    lime_path = os.path.join(app.config["LIME_FOLDER"], lime_filename)

    plt.figure(figsize=(5, 5))
    plt.imshow(cast(Any, lime_overlay))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        lime_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()

    return lime_filename


def is_gradcam_candidate_layer(layer):
    output_shape = getattr(layer, "output_shape", None)

    if not isinstance(output_shape, tuple) or len(output_shape) != 4:
        return False

    height, width = output_shape[1], output_shape[2]
    return height not in (None, 1) and width not in (None, 1)


def find_last_spatial_feature_layer(model):
    preferred_layers = {
        "resnet50": ["conv5_block3_out", "conv5_block3_add"],
        "densenet121": ["conv5_block16_2_conv", "relu"],
        "mobilenetv2_1.00_224": ["Conv_1", "out_relu"]
    }

    for model_name, layer_names in preferred_layers.items():
        if model.name == model_name:
            for layer_name in layer_names:
                try:
                    layer = model.get_layer(layer_name)
                except ValueError:
                    continue

                if is_gradcam_candidate_layer(layer):
                    return layer.name

    for layer in reversed(model.layers):
        if is_gradcam_candidate_layer(layer):
            return layer.name

    conv_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, layers.Conv2D)
    ]

    if conv_layers:
        return conv_layers[-1].name

    raise ValueError("No spatial feature layer found for Grad-CAM.")


def find_nested_base_and_last_conv(model):
    for layer_index in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[layer_index]

        if isinstance(layer, models.Model):
            return layer_index, layer, find_last_spatial_feature_layer(layer)

    return None, model, find_last_spatial_feature_layer(model)


def make_gradcam_heatmap(model, processed_img_batch, pred_index=None):
    base_model_index, base_model, conv_layer_name = (
        find_nested_base_and_last_conv(model)
    )
    conv_layer = base_model.get_layer(conv_layer_name)

    base_grad_model = models.Model(
        inputs=base_model.input,
        outputs=[
            cast(Any, conv_layer).output,
            cast(Any, base_model).output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, x = base_grad_model(
            processed_img_batch,
            training=False
        )

        if base_model_index is not None:
            for layer in model.layers[base_model_index + 1:]:
                x = layer(x, training=False)
            predictions = x
        else:
            predictions = model(processed_img_batch, training=False)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are None.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    if max_val.numpy() > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_heatmap_on_image(img_array, heatmap, alpha=0.45):
    img_uint8 = img_array.astype("uint8")
    heatmap_resized = cv2.resize(
        heatmap,
        (img_uint8.shape[1], img_uint8.shape[0])
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(
        cast(Any, heatmap_uint8),
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(
        img_uint8,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )


def generate_gradcam_for_model(
    image_path,
    model,
    model_type,
    model_display_name,
    pred_idx,
    output_filename_prefix
):
    img_array = load_image_for_prediction(image_path)
    processed_img_batch = preprocess_for_model(img_array, model_type)
    heatmap = make_gradcam_heatmap(model, processed_img_batch, pred_idx)
    overlay = overlay_heatmap_on_image(img_array, heatmap)

    safe_model_name = model_display_name.lower().replace(" ", "_")
    output_filename = (
        f"{output_filename_prefix}_{safe_model_name}_gradcam.png"
    )
    output_path = os.path.join(app.config["GRADCAM_FOLDER"], output_filename)
    Image.fromarray(overlay).save(output_path)

    return heatmap, output_filename


def predict_single_model(model, img_array, model_type):
    processed = preprocess_for_model(img_array, model_type)
    probs = model.predict(processed, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return {
        "pred_idx": pred_idx,
        "pred_class": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "probabilities": probs
    }


def predict_ensemble(image_path):
    img_array = load_image_for_prediction(image_path)

    resnet_result = predict_single_model(
        resnet_model,
        img_array,
        "resnet"
    )

    densenet_result = predict_single_model(
        densenet_model,
        img_array,
        "densenet"
    )

    mobilenet_result = predict_single_model(
        mobilenet_model,
        img_array,
        "mobilenet"
    )

    ensemble_probs = (
        W_RESNET * resnet_result["probabilities"] +
        W_DENSENET * densenet_result["probabilities"] +
        W_MOBILENET * mobilenet_result["probabilities"]
    )

    ensemble_pred_idx = int(np.argmax(ensemble_probs))
    ensemble_confidence = float(np.max(ensemble_probs))

    class_probabilities = []

    for idx, class_name in enumerate(CLASS_NAMES):
        class_probabilities.append({
            "class_name": class_name,
            "probability": float(ensemble_probs[idx]),
            "percentage": round(float(ensemble_probs[idx]) * 100, 2)
        })

    class_probabilities = sorted(
        class_probabilities,
        key=lambda x: x["probability"],
        reverse=True
    )

    return {
        "resnet": resnet_result,
        "densenet": densenet_result,
        "mobilenet": mobilenet_result,
        "ensemble": {
            "pred_idx": ensemble_pred_idx,
            "pred_class": CLASS_NAMES[ensemble_pred_idx],
            "confidence": ensemble_confidence,
            "confidence_percent": round(ensemble_confidence * 100, 2),
            "probabilities": ensemble_probs,
            "class_probabilities": class_probabilities
        }
    }


def calculate_model_agreement(prediction_results):
    preds = [
        prediction_results["resnet"]["pred_class"],
        prediction_results["densenet"]["pred_class"],
        prediction_results["mobilenet"]["pred_class"]
    ]

    unique_preds = set(preds)

    if len(unique_preds) == 1:
        return "Full Agreement", 1.0

    if len(unique_preds) == 2:
        return "Partial Agreement", 0.66

    return "Disagreement", 0.33


def normalize_heatmap(heatmap):
    heatmap = heatmap.astype(np.float32)
    min_val = np.min(heatmap)
    max_val = np.max(heatmap)

    if max_val - min_val == 0:
        return np.zeros_like(heatmap)

    return (heatmap - min_val) / (max_val - min_val)


def heatmap_cosine_similarity(h1, h2):
    h1_flat = normalize_heatmap(h1).reshape(1, -1)
    h2_flat = normalize_heatmap(h2).reshape(1, -1)
    return float(cosine_similarity(h1_flat, h2_flat)[0][0])


def heatmap_ssim_similarity(h1, h2):
    h1_norm = normalize_heatmap(h1)
    h2_norm = normalize_heatmap(h2)
    similarity = ssim(h1_norm, h2_norm, data_range=1.0)
    return float(cast(Any, similarity))


def compute_heatmap_consistency(
    resnet_heatmap,
    densenet_heatmap,
    mobilenet_heatmap
):
    target_size = IMG_SIZE
    resized_heatmaps = [
        cv2.resize(heatmap, target_size)
        for heatmap in (
            resnet_heatmap,
            densenet_heatmap,
            mobilenet_heatmap
        )
    ]
    pairs = [
        (resized_heatmaps[0], resized_heatmaps[1]),
        (resized_heatmaps[0], resized_heatmaps[2]),
        (resized_heatmaps[1], resized_heatmaps[2])
    ]

    cosine_scores = [
        heatmap_cosine_similarity(a, b)
        for a, b in pairs
    ]
    ssim_scores = [
        heatmap_ssim_similarity(a, b)
        for a, b in pairs
    ]

    return {
        "average_cosine": float(np.mean(cosine_scores)),
        "average_ssim": float(np.mean(ssim_scores)),
        "cosine_scores": cosine_scores,
        "ssim_scores": ssim_scores
    }


def calculate_qeef_reliability(prediction_results, gradcam_heatmaps):
    ensemble_confidence = prediction_results["ensemble"]["confidence"]
    agreement_label, agreement_score = calculate_model_agreement(
        prediction_results
    )
    consistency = compute_heatmap_consistency(
        gradcam_heatmaps["resnet"],
        gradcam_heatmaps["densenet"],
        gradcam_heatmaps["mobilenet"]
    )

    average_cosine = consistency["average_cosine"]
    average_ssim = consistency["average_ssim"]
    consistency_score = 0.5 * average_cosine + 0.5 * average_ssim
    qeef_trust_score = (
        0.40 * ensemble_confidence +
        0.30 * consistency_score +
        0.30 * agreement_score
    )

    reasons = []

    if qeef_trust_score < 0.55:
        reasons.append("Low QEEF trust score")
    if ensemble_confidence < 0.70:
        reasons.append("Low ensemble confidence")
    if average_cosine < 0.50:
        reasons.append("Low Grad-CAM cosine similarity")
    if average_ssim < 0.50:
        reasons.append("Low Grad-CAM SSIM similarity")
    if agreement_score < 0.66:
        reasons.append("Strong model disagreement")
    elif agreement_score < 1.0:
        reasons.append("Partial model disagreement")

    if (
        qeef_trust_score >= 0.75
        and ensemble_confidence >= 0.85
        and agreement_score == 1.0
    ):
        flag = "Reliable"
        reason = "High confidence and consistent prediction"
    elif (
        qeef_trust_score < 0.55
        or ensemble_confidence < 0.70
        or agreement_score < 0.66
    ):
        flag = "Suspicious"
        reason = "; ".join(reasons)
    else:
        flag = "Review Needed"
        reason = "; ".join(reasons) or "Prediction requires expert review"

    return {
        "flag": flag,
        "reason": reason,
        "ensemble_confidence": ensemble_confidence,
        "agreement_label": agreement_label,
        "agreement_score": agreement_score,
        "average_cosine_similarity": average_cosine,
        "average_ssim_similarity": average_ssim,
        "qeef_trust_score": qeef_trust_score,
        "qeef_trust_percentage": round(qeef_trust_score * 100, 2)
    }


def calculate_basic_reliability(prediction_results):
    ensemble_confidence = prediction_results["ensemble"]["confidence"]
    agreement_label, agreement_score = calculate_model_agreement(
        prediction_results
    )

    if ensemble_confidence >= 0.90 and agreement_score == 1.0:
        flag = "Reliable"
        reason = "High ensemble confidence and full model agreement."
    elif ensemble_confidence < 0.70 or agreement_score < 0.66:
        flag = "Suspicious"
        reason = "Low confidence or strong disagreement among models."
    else:
        flag = "Review Needed"
        reason = "Prediction is acceptable but requires expert review."

    return {
        "flag": flag,
        "reason": reason,
        "ensemble_confidence": ensemble_confidence,
        "agreement_label": agreement_label,
        "agreement_score": agreement_score,
        "average_cosine_similarity": None,
        "average_ssim_similarity": None,
        "qeef_trust_score": None,
        "qeef_trust_percentage": None
    }


# ===============================
# Routes
# ===============================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/progress/<job_id>", methods=["GET"])
def progress(job_id):
    return jsonify(get_job_progress(job_id))


@app.route("/predict", methods=["POST"])
def predict():
    job_id = request.form.get("job_id", "")
    update_job_progress(
        job_id,
        0,
        "Validating and saving uploaded image...",
        0
    )

    if "file" not in request.files:
        return "No file uploaded.", 400

    file = request.files["file"]

    if not file.filename:
        return "No selected file.", 400

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_ext:
        return "Invalid file type. Please upload jpg, jpeg, png, or bmp.", 400

    unique_filename = f"{uuid.uuid4().hex}{ext}"
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(image_path)

    update_job_progress(
        job_id,
        0,
        "Running CNN ensemble and class prediction...",
        1
    )
    prediction_results = predict_ensemble(image_path)
    filename_prefix = os.path.splitext(unique_filename)[0]

    update_job_progress(
        job_id,
        0,
        "Generating model-specific Grad-CAM explanations...",
        2
    )
    gradcam_outputs = {
        "resnet": generate_gradcam_for_model(
            image_path,
            resnet_model,
            "resnet",
            "resnet",
            prediction_results["resnet"]["pred_idx"],
            filename_prefix
        ),
        "densenet": generate_gradcam_for_model(
            image_path,
            densenet_model,
            "densenet",
            "densenet",
            prediction_results["densenet"]["pred_idx"],
            filename_prefix
        ),
        "mobilenet": generate_gradcam_for_model(
            image_path,
            mobilenet_model,
            "mobilenet",
            "mobilenet",
            prediction_results["mobilenet"]["pred_idx"],
            filename_prefix
        )
    }
    gradcam_heatmaps = {
        model_type: output[0]
        for model_type, output in gradcam_outputs.items()
    }
    gradcam_filenames = {
        model_type: output[1]
        for model_type, output in gradcam_outputs.items()
    }
    gradcam_urls = {
        model_type: url_for(
            "static",
            filename=f"gradcam/{filename}"
        )
        for model_type, filename in gradcam_filenames.items()
    }

    lime_url = None
    lime_filename = None

    try:
        update_job_progress(
            job_id,
            0,
            f"Generating LIME explanation: 0/{LIME_NUM_SAMPLES}",
            3
        )

        def report_lime_progress(completed, total):
            percentage = round((completed / total) * 100)
            update_job_progress(
                job_id,
                percentage,
                f"Generating LIME explanation: {completed}/{total}",
                3
            )

        lime_filename = generate_lime_explanation(
            image_path,
            prediction_results["ensemble"]["pred_idx"],
            filename_prefix,
            progress_callback=report_lime_progress
        )
        lime_url = url_for(
            "static",
            filename=f"lime/{lime_filename}"
        )
    except Exception:
        app.logger.exception("LIME explanation generation failed.")

    try:
        update_job_progress(
            job_id,
            100,
            "Computing Grad-CAM consistency and QEEF reliability...",
            4
        )
        reliability = calculate_qeef_reliability(
            prediction_results,
            gradcam_heatmaps
        )
    except Exception:
        app.logger.exception(
            "QEEF reliability calculation failed. Using basic reliability."
        )
        reliability = calculate_basic_reliability(prediction_results)

    update_job_progress(
        job_id,
        100,
        "Applying abnormality flagging...",
        5
    )

    uploaded_image_url = url_for(
        "static",
        filename=f"uploads/{unique_filename}"
    )

    update_job_progress(
        job_id,
        100,
        "Preparing final explainability report...",
        6
    )

    weights = {
        "ResNet50": W_RESNET,
        "DenseNet121": W_DENSENET,
        "MobileNetV2": W_MOBILENET
    }
    return render_template(
        "result.html",
        image_url=uploaded_image_url,
        results=prediction_results,
        reliability=reliability,
        gradcam_urls=gradcam_urls,
        lime_url=lime_url,
        weights=weights
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
