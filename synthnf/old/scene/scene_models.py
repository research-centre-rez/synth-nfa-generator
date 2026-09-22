from uuid import uuid4


class SceneModel:
    def __init__(self, scene_dict={}):
        self._scene_dict = scene_dict
        self._items_categorized = {}

    def add_sensor(self, sensor, sensor_name=None):
        self._add_item(sensor, "sensor", item_name=sensor_name)

    def add_integrator(self, integrator, integrator_name=None):
        self._add_item(integrator, "integrator", item_name=integrator_name)

    def add_light(self, light, light_name=None):
        self._add_item(light, "light", item_name=light_name)

    def add_shape(self, shape, shape_name=None):
        self._add_item(shape, "shape", item_name=shape_name)

    def _add_item(self, item, item_category, item_name=None):
        if item_name is None:
            item_name = f"{str(uuid4())}"

        item_model_key = f"{item_category}_{item_name}"

        self._scene_dict[item_name] = item
        category_dict = self._items_categorized.setdefault(item_category, {})
        category_dict[item_name] = item_model_key

    def to_dictionary(self):
        return self._scene_dict
