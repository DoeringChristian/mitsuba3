from typing import Any, Callable, List
import pytest
import drjit as dr
import mitsuba as mi

def test01_cornell_box(variants_vec_rgb):
    print(f"{variants_vec_rgb=}")
    
    w = 128
    h = 128

    n = 5
    
    k = "light.emitter.radiance.value"
    
    def func(scene: mi.Scene, x) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    def run(n: int, func: Callable[mi.Scene, Any]) -> List[mi.TensorXf]:

        scene = mi.cornell_box()
        scene["sensor"]["film"]["width"] = w
        scene["sensor"]["film"]["height"] = h
        scene = mi.load_dict(scene)
        
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
            
    images_ref = run(n, func)
    images_frozen = run(n, dr.freeze(func))

    for (ref, frozen) in zip(images_ref, images_frozen):
        print(f"{ref=}")
        print(f"{frozen=}")
        assert dr.allclose(ref, frozen)
        
