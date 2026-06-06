from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

CLASS_NAMES = ['Normal', 'Tuberculosis', 'Pneumonia', 'COVID-19']
RAW_MODEL_CLASSES = ['Covid', 'Normal', 'Pneumonia', 'Tuberculosis']
RAW_TO_DISPLAY = {
    'Covid': 'COVID-19',
    'Normal': 'Normal',
    'Pneumonia': 'Pneumonia',
    'Tuberculosis': 'Tuberculosis',
}
TARGET_LAYER_HINTS = {
    'densenet': ['conv5_block16_concat', 'relu'],
    'resnet50': ['conv5_block3_out'],
    'efficientnet': ['top_activation', 'top_conv'],
    'mobilenetv2': ['Conv_1', 'out_relu'],
    'vgg16': ['block5_conv3'],
}


def infer_model_kind(model_path: Path) -> str:
    name = model_path.name.lower()
    if 'resnet' in name:
        return 'resnet50'
    if 'efficientnet' in name:
        return 'efficientnet'
    if 'densenet' in name:
        return 'densenet'
    if 'mobilenet' in name:
        return 'mobilenetv2'
    if 'vgg16' in name:
        return 'vgg16'
    return 'custom'


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def preprocess_for_model(image: Image.Image, model_kind: str, target_size: tuple[int, int]) -> np.ndarray:
    rgb_image = image.convert('RGB').resize(target_size)
    batch = np.expand_dims(np.asarray(rgb_image, dtype=np.float32), axis=0)
    if model_kind == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input

        return preprocess_input(batch)
    if model_kind == 'efficientnet':
        from tensorflow.keras.applications.efficientnet import preprocess_input

        return preprocess_input(batch)
    if model_kind == 'densenet':
        from tensorflow.keras.applications.densenet import preprocess_input

        return preprocess_input(batch)
    if model_kind == 'mobilenetv2':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        return preprocess_input(batch)
    if model_kind == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input

        return preprocess_input(batch)
    return batch / 255.0


def align_probabilities(raw_probs: np.ndarray) -> np.ndarray:
    mapped = {label: 0.0 for label in CLASS_NAMES}
    for raw_label, score in zip(RAW_MODEL_CLASSES, raw_probs.tolist()):
        display_label = RAW_TO_DISPLAY.get(raw_label)
        if display_label in mapped:
            mapped[display_label] = float(score)
    aligned = np.array([mapped[label] for label in CLASS_NAMES], dtype=np.float32)
    if aligned.sum() <= 0:
        return np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES), dtype=np.float32)
    return aligned / aligned.sum()


def build_compat_model(model_path: Path):
    import tempfile
    import tensorflow as tf

    model_kind = infer_model_kind(model_path)
    builders = {
        'densenet': lambda: tf.keras.applications.DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3), name='densenet121'),
        'resnet50': lambda: tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3), name='resnet50'),
        'efficientnet': lambda: tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3), name='efficientnetb0'),
        'mobilenetv2': lambda: tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3), name='mobilenetv2'),
        'vgg16': lambda: tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(224, 224, 3), name='vgg16'),
    }
    if model_kind not in builders:
        raise ValueError(f'No compatibility builder is defined for {model_path.name}.')

    with tempfile.TemporaryDirectory() as td:
        with ZipFile(model_path, 'r') as archive:
            if 'model.weights.h5' not in archive.namelist():
                raise ValueError(f'{model_path.name} does not contain model.weights.h5.')
            archive.extract('model.weights.h5', path=td)
        weights_path = Path(td) / 'model.weights.h5'
        base_model = builders[model_kind]()
        base_model.trainable = False
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(224, 224, 3), name='input_layer_1'),
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d'),
                tf.keras.layers.BatchNormalization(name='batch_normalization'),
                tf.keras.layers.Dense(256, activation='relu', name='dense'),
                tf.keras.layers.Dropout(0.5, name='dropout'),
                tf.keras.layers.Dense(4, activation='softmax', name='dense_1'),
            ],
            name='sequential',
        )
        model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
        model.load_weights(weights_path)
        setattr(model, '_brave_loader_mode', 'compat')
        return model


def load_model(model_path: Path):
    import tensorflow as tf

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        setattr(model, '_brave_loader_mode', 'standard')
        return model
    except Exception:
        return build_compat_model(model_path)


def find_last_conv_layer(model):
    import tensorflow as tf

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            nested = find_last_conv_layer(layer)
            if nested is not None:
                return nested
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        try:
            shape = layer.output_shape
        except Exception:
            shape = None
        if isinstance(shape, list):
            shape = shape[0] if shape else None
        if shape is not None and hasattr(shape, '__len__') and len(shape) == 4 and 'conv' in layer.name.lower():
            return layer
    return None


def resolve_gradcam_target_layer(model, model_kind: str):
    for layer_name in TARGET_LAYER_HINTS.get(model_kind, []):
        try:
            return model.get_layer(layer_name)
        except Exception:
            pass
        for layer in model.layers:
            if hasattr(layer, 'get_layer'):
                try:
                    return layer.get_layer(layer_name)
                except Exception:
                    pass
    return find_last_conv_layer(model)


def run_gradcam_forward(model, target_layer, batch):
    import tensorflow as tf

    for layer in model.layers:
        if not isinstance(layer, tf.keras.Model):
            continue
        try:
            nested_target = layer.get_layer(target_layer.name)
        except Exception:
            continue
        feature_model = tf.keras.models.Model(layer.inputs, [nested_target.output, layer.outputs[0]])
        with tf.GradientTape() as tape:
            conv_output, features = feature_model(batch, training=False)
            after_backbone = False
            x = features
            for outer_layer in model.layers:
                if outer_layer is layer:
                    after_backbone = True
                    continue
                if not after_backbone:
                    continue
                x = outer_layer(x, training=False)
            predictions = x
            pred_index = int(tf.argmax(predictions[0]).numpy())
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_output)
        return conv_output, predictions, grads, pred_index

    grad_model = tf.keras.models.Model(model.inputs, [target_layer.output, model.outputs[0]])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(batch, training=False)
        pred_index = int(tf.argmax(predictions[0]).numpy())
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_output)
    return conv_output, predictions, grads, pred_index


def render_gradcam_overlay(image: Image.Image, normalized_heatmap: np.ndarray, display_label: str) -> np.ndarray:
    def thoracic_focus_prior(height: int, width: int) -> np.ndarray:
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        x_mid = width / 2.0
        y_mid = height * 0.56
        thorax = np.exp(-((((xs - x_mid) / (width * 0.33)) ** 2) + (((ys - y_mid) / (height * 0.36)) ** 2)) * 1.6)

        left_shoulder = np.exp(-((((xs - width * 0.16) / (width * 0.18)) ** 2) + (((ys - height * 0.16) / (height * 0.14)) ** 2)) * 2.2)
        right_shoulder = np.exp(-((((xs - width * 0.84) / (width * 0.18)) ** 2) + (((ys - height * 0.16) / (height * 0.14)) ** 2)) * 2.2)
        left_axilla = np.exp(-((((xs - width * 0.18) / (width * 0.15)) ** 2) + (((ys - height * 0.31) / (height * 0.12)) ** 2)) * 2.0)
        right_axilla = np.exp(-((((xs - width * 0.82) / (width * 0.15)) ** 2) + (((ys - height * 0.31) / (height * 0.12)) ** 2)) * 2.0)

        prior = thorax * (1.0 - 0.72 * np.clip(left_shoulder + right_shoulder, 0, 1))
        prior *= (1.0 - 0.58 * np.clip(left_axilla + right_axilla, 0, 1))
        prior = cv2.GaussianBlur(prior.astype(np.float32), (0, 0), 17)
        return np.clip(prior, 0.12, 1.0)

    gray = np.asarray(image.convert('L'))
    base_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    smoothed = cv2.GaussianBlur(np.clip(normalized_heatmap, 0, 1).astype(np.float32), (0, 0), 11)
    center_prior = thoracic_focus_prior(smoothed.shape[0], smoothed.shape[1])
    focused_map = smoothed * (0.35 + 0.65 * center_prior)
    heat_uint8 = np.uint8(np.clip(focused_map, 0, 1) * 255)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base_rgb, 0.56, heat_rgb, 0.44, 0)

    threshold = max(0.46, float(np.percentile(focused_map, 89)))
    mask = np.uint8(focused_map >= threshold) * 255
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        dark_mask = gray < np.percentile(gray, 58)
        ranked: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 120:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            contour_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            region = contour_mask.astype(bool)
            if not np.any(region):
                continue
            heat_score = float(focused_map[region].mean())
            dark_overlap = float(dark_mask[region].mean())
            center_score = float(center_prior[region].mean())
            score = heat_score * (0.45 + dark_overlap) * (0.5 + center_score) * np.sqrt(area)
            ranked.append((score, contour, (x, y, w, h)))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:2]
        for _, contour, _ in selected:
            cv2.drawContours(overlay, [contour], -1, (255, 255, 255), 2)

        if selected:
            _, contour, (x, y, w, h) = selected[0]
            focus_x = x + w // 2
            focus_y = y + h // 2
            label = f'{display_label} focus'
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.55, min(0.82, overlay.shape[1] / 900.0))
            thickness = 2
            text_w, text_h = cv2.getTextSize(label, font, scale, thickness)[0]
            box_x = int(np.clip(x, 12, max(12, overlay.shape[1] - text_w - 24)))
            box_y = max(text_h + 18, y - 18)
            top_left = (box_x - 8, box_y - text_h - 10)
            bottom_right = (box_x + text_w + 8, box_y + 8)
            cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), -1)
            cv2.putText(overlay, label, (box_x, box_y), font, scale, (38, 38, 38), thickness, cv2.LINE_AA)
            cv2.arrowedLine(overlay, (box_x + text_w // 2, bottom_right[1]), (focus_x, focus_y), (255, 255, 255), 2, tipLength=0.12)
            cv2.circle(overlay, (focus_x, focus_y), 6, (255, 255, 255), 2)
    return overlay


def generate_gradcam(model_path: Path, image_path: Path, output_path: Path) -> dict[str, object]:
    import tensorflow as tf

    model = load_model(model_path)
    image = Image.open(image_path).convert('RGB')
    input_shape = getattr(model, 'input_shape', (None, 224, 224, 3))
    height = int(input_shape[1] or 224)
    width = int(input_shape[2] or 224)
    model_kind = infer_model_kind(model_path)
    batch = preprocess_for_model(image, model_kind, (width, height))
    model(batch, training=False)
    target_layer = resolve_gradcam_target_layer(model, model_kind)
    if target_layer is None:
        raise ValueError(f'No convolutional layer was found for {model_path.name}.')

    conv_output, predictions, grads, pred_index = run_gradcam_forward(model, target_layer, batch)
    if grads is None:
        raise ValueError(f'Gradients were unavailable for {model_path.name}.')

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    cam = tf.reduce_sum(conv_output * pooled_grads[tf.newaxis, tf.newaxis, :], axis=-1)
    cam = tf.maximum(cam, 0)
    cam_max = float(tf.reduce_max(cam).numpy())
    if cam_max <= 0:
        raise ValueError(f'Grad-CAM produced an empty map for {model_path.name}.')

    cam = cam / cam_max
    cam = tf.image.resize(cam[..., tf.newaxis], (image.size[1], image.size[0]), method='bilinear').numpy().squeeze()

    raw_output = np.asarray(predictions[0].numpy(), dtype=np.float32).flatten()
    probs = raw_output if np.isclose(raw_output.sum(), 1.0, atol=0.05) else softmax(raw_output)
    probs = align_probabilities(probs)
    display_label = CLASS_NAMES[int(np.argmax(probs))]
    overlay = render_gradcam_overlay(image, cam, display_label)
    Image.fromarray(overlay).save(output_path)

    loader_mode = getattr(model, '_brave_loader_mode', 'standard')
    return {
        'probs': probs.tolist(),
        'mode': 'real-external-compat' if loader_mode == 'compat' else 'real-external',
        'engine': model_path.name,
        'detail': f'Grad-CAM from {target_layer.name}',
        'predicted_class': display_label,
        'overlay_path': str(output_path),
    }


def predict(model_path: Path, image_path: Path) -> dict[str, object]:
    model = load_model(model_path)
    image = Image.open(image_path).convert('RGB')
    input_shape = getattr(model, 'input_shape', (None, 224, 224, 3))
    height = int(input_shape[1] or 224)
    width = int(input_shape[2] or 224)
    model_kind = infer_model_kind(model_path)
    batch = preprocess_for_model(image, model_kind, (width, height))
    raw_output = np.asarray(model.predict(batch, verbose=0)[0], dtype=np.float32).flatten()
    probs = raw_output if np.isclose(raw_output.sum(), 1.0, atol=0.05) else softmax(raw_output)
    probs = align_probabilities(probs)
    loader_mode = getattr(model, '_brave_loader_mode', 'standard')
    return {
        'probs': probs.tolist(),
        'mode': 'real-external-compat' if loader_mode == 'compat' else 'real-external',
        'engine': model_path.name,
        'detail': f"{model_kind} {'compatibility ' if loader_mode == 'compat' else ''}external inference".strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--gradcam-output')
    args = parser.parse_args()

    if args.gradcam_output:
        payload = generate_gradcam(Path(args.model), Path(args.image), Path(args.gradcam_output))
    else:
        payload = predict(Path(args.model), Path(args.image))
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
