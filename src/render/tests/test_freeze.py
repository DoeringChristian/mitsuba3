from typing import Any, Callable, List
import pytest
import drjit as dr
import mitsuba as mi
import glob
from os.path import join

from mitsuba.scalar_rgb.test.util import find_resource

@pytest.mark.parametrize("scene_name", ["cornell_box"])
def test01_forward(variants_vec_rgb, scene_name):
    
    w = 16
    h = 16

    n = 5
    
    k = "light.emitter.radiance.value"
    
    def func(scene: mi.Scene, x) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    def run(scene: mi.Scene, n: int, func: Callable[mi.Scene, Any]) -> List[mi.TensorXf]:

        params = mi.traverse(scene)
        value = mi.Float(params[k].x)
        
        images = []
        for i in range(n):
            params[k].x = value + 10.0 * i
            params.update()
            
            img = func(scene, params[k].x)
            dr.eval(img)

            images.append(img)
            
        return images
            
    def load_scene(name: str):
        
        if name == "cornell_box":
            scene = mi.cornell_box()
            scene["sensor"]["film"]["width"] = w
            scene["sensor"]["film"]["height"] = h
            scene = mi.load_dict(scene)
        return scene

    scene = load_scene(scene_name)
    images_ref = run(scene, n, func)
    scene = load_scene(scene_name)
    images_frozen = run(scene, n, dr.freeze(func))

    for (ref, frozen) in zip(images_ref, images_frozen):
        assert dr.allclose(ref, frozen)
        
