DOCUMENT_EXTRACT_SCHEMA = {
    "name": "document_extract",
    "description": (
        "Safely inspect and OCR the latest PDF uploaded in the current Feishu session. "
        "The runtime binds the source; never pass a source path. Always check metadata/text_layer first. "
        "For scanned pages, use render_pages (direct PDF rasterization with auto-rotation/deskew), then "
        "detect_regions or tile_image/crop_image before ocr_image on dense layouts. Generated image paths "
        "may also be passed to vision_analyze for blind verification. Never use a browser PDF viewer fallback."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "metadata", "text_layer", "render_pages", "detect_regions",
                    "tile_image", "crop_image", "ocr_image", "cleanup",
                ],
            },
            "first_page": {"type": "integer", "minimum": 1, "maximum": 500},
            "last_page": {"type": "integer", "minimum": 1, "maximum": 500},
            "dpi": {"type": "integer", "minimum": 150, "maximum": 300, "default": 300},
            "image_path": {
                "type": "string",
                "description": (
                    "Exact generated PNG path returned by this tool for the current source. "
                    "Arbitrary images and paths outside the source work directory are rejected."
                ),
            },
            "max_regions": {"type": "integer", "minimum": 1, "maximum": 50, "default": 30},
            "create_crops": {"type": "boolean", "default": True},
            "rows": {"type": "integer", "minimum": 1, "maximum": 6, "default": 2},
            "columns": {"type": "integer", "minimum": 1, "maximum": 6, "default": 2},
            "overlap": {"type": "number", "minimum": 0.0, "maximum": 0.2, "default": 0.08},
            "x": {"type": "integer", "minimum": 0},
            "y": {"type": "integer", "minimum": 0},
            "width": {"type": "integer", "minimum": 16},
            "height": {"type": "integer", "minimum": 16},
            "engine": {
                "type": "string", "enum": ["tesseract", "paddle"], "default": "tesseract",
            },
            "psm": {"type": "integer", "enum": [6, 11], "default": 11},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}
