from .q_glb_material_modifier import QManualGLBMaterialModifier, QPresetGLBMaterialModifier

NODE_CLASS_MAPPINGS = {
    "QManualGLBMaterialModifier": QManualGLBMaterialModifier,
    "QPresetGLBMaterialModifier": QPresetGLBMaterialModifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QManualGLBMaterialModifier": "Q Manual GLB Material Modifier",
    "QPresetGLBMaterialModifier": "Q Preset GLB Material Modifier",
}