def build_vision_tower():
    vision_tower = 'openai/clip-vit-large-patch14-336'
    return CLIPVisionTower(vision_tower)
