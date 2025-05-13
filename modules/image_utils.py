from imgutils.generic.yolo import YOLOModel
from PIL import Image, ImageDraw
from imgutils.edge.lineart import edge_image_with_lineart

from imgutils.segment import segment_rgba_with_isnetis
from imgutils.tagging import get_camie_tags

def detect_and_draw_boxes(image_path, output_path):
    model = YOLOModel("deepghs/booru_yolo")
    image = Image.open(image_path).convert("RGB")
    detections = model.predict(image, "yolov8m_as02")
    draw = ImageDraw.Draw(image)
    for box, label, confidence in detections:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{label} {confidence:.2f}", fill="red")
    image.save(output_path)
    return output_path, detections

def extract_lineart(image_path, output_path, coarse=False, detect_resolution=512, backcolor='white', forecolor=None):
    img = edge_image_with_lineart(
        image=image_path,
        coarse=coarse,
        detect_resolution=detect_resolution,
        backcolor=backcolor,
        forecolor=forecolor
    )
    img.save(output_path)
    return output_path

# 分類モデル推論
def classify_image(image_path, repo_id="deepghs/anime_aesthetic", model_name="swinv2pv3_v0_448_ls0.2_x"):
    from imgutils.generic.classify import ClassifyModel
    image = Image.open(image_path).convert("RGB")
    classifier = ClassifyModel(repo_id)
    label = classifier.predict(image, model_name)
    return label

# Camieタグ分類
def classify_image_camie(image_path):
    rating, tags, chars = get_camie_tags(image_path)
    return {
        "rating": rating,
        "tags": tags,
        "chars": chars
    }

def segment_image_with_isnetis(image_path, output_path):
    """
    imgutilsのISNetISモデルで画像をセグメントし、透明PNGとして保存
    :param image_path: 入力画像パス
    :param output_path: 出力画像パス
    :return: 出力画像パス
    """
    mask, image = segment_rgba_with_isnetis(image_path)
    image.save(output_path)
    return output_path