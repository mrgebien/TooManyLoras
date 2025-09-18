from nodes import LoraLoader
import folder_paths


class TooManyLoras:
    NAME = "TooManyLoras"
    CATEGORY = "loaders"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            }
        }

        # Dynamically create 10 lora inputs
        for i in range(1, 11):
            inputs["required"][f"lora_{i:02d}"] = (['None'] + folder_paths.get_filename_list("loras"), )
            inputs["required"][f"strength_{i:02d}"] = ("FLOAT", {
                "default": 1.0,
                "min": -10.0,
                "max": 10.0,
                "step": 0.01
            })

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"

    def load_loras(self, model, clip, **kwargs):
        for i in range(1, 11):
            lora_name = kwargs.get(f"lora_{i:02d}")
            strength = kwargs.get(f"strength_{i:02d}", 0)
            if lora_name and lora_name != "None" and strength != 0:
                model, clip = LoraLoader().load_lora(model, clip, lora_name, strength, strength)

        return (model, clip)


# Register the node
NODE_CLASS_MAPPINGS = {
    "TooManyLoras": TooManyLoras
}
